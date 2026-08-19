import numpy as np
import torch
import networkx as nx

from torch_geometric.data import Data


def safe_float(val):
    try:
        f = float(val)
        if np.isnan(f):
            return 0.0
        return f
    except:
        return 0.0


def has_consecutive_block(labels, block_size=5):
    count = 0
    for val in labels:
        if val == 1:
            count += 1
            if count >= block_size:
                return True
        else:
            count = 0
    return False

def create_cluster_edges(G):
    """
    클러스터 그래프 내에서 edge 연결 알고리즘:
    - 각 클러스터에는 3개의 트랙 (track_id)이 존재하고, 각 트랙은 10개의 노드를 가짐
    - 동일 트랙 내에서는 과거 → 미래 노드들로 fully connected (i < j)
    - 트랙 A → B, A → C와 같은 경우: 현재 노드 i → 다른 트랙의 노드 k (k >= i)
    → 최종적으로 155 edge * 3 = 465개 생성
    """
    edges = []
    track_ids = sorted(set(nx.get_node_attributes(G, 'track_id').values()))
    track_to_nodes = {tid: [] for tid in track_ids}

    # 시간 순으로 정렬된 노드 저장
    for n, attr in G.nodes(data=True):
        tid = attr.get("track_id")
        track_to_nodes[tid].append((n, attr.get("time")))
    for tid in track_ids:
        track_to_nodes[tid] = sorted(track_to_nodes[tid], key=lambda x: x[1])

    # 동일 트랙 내 edge: i < j
    for tid in track_ids:
        nodes = [n for n, _ in track_to_nodes[tid]]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append((nodes[i], nodes[j]))

    # 트랙 간 edge: A → B, A → C, B → C (i → k, k ≥ i)
    for i, tid_from in enumerate(track_ids):
        nodes_from = [n for n, _ in track_to_nodes[tid_from]]
        for j, tid_to in enumerate(track_ids):
            if tid_from == tid_to:
                continue
            nodes_to = [n for n, _ in track_to_nodes[tid_to]]
            for i_idx in range(len(nodes_from)):
                for k_idx in range(i_idx, len(nodes_to)):
                    edges.append((nodes_from[i_idx], nodes_to[k_idx]))

    return edges

# block_size: consecutive anomaly block 개수
def cluster_to_pyg_data(G, block_size):
    if G is None:
        return None

    # 1) 노드를 track_id별로 모으기
    #   예: { 'A': [노드들], 'B': [노드들], 'C': [노드들] }
    track_map = {}
    for n in G.nodes():
        tid = G.nodes[n].get("track_id")
        if tid not in track_map:
            track_map[tid] = []
        track_map[tid].append(n)

    # 2) track_id별로 정렬하고, feats/anoms 등을 쌓기
    all_nodes_sorted = []
    feats = []
    anoms = []
    track_id_per_node = []

    for tid, nodes_in_track in track_map.items():
        # 시간 기준 정렬
        nodes_in_track = sorted(nodes_in_track, key=lambda n: G.nodes[n]["time"])

        for n in nodes_in_track:
            attr = G.nodes[n]
            lon = safe_float(attr.get("LON"))
            lat = safe_float(attr.get("LAT"))
            spd = safe_float(attr.get("SPEED"))
            s   = safe_float(attr.get("COURSE_SIN"))
            c   = safe_float(attr.get("COURSE_COS"))

            feats.append([lon, lat, spd, s, c])
            anoms.append(1 if attr.get("ANOMALY", False) else 0)
            track_id_per_node.append(tid)   # 노드가 어떤 경로(A/B/C)에 속하는지 기록

            all_nodes_sorted.append(n)

    # 3) 이제 feats/anoms는 전체 노드 순서 (A노드들 → B노드들 → C노드들)로 쌓임
    x = torch.tensor(feats, dtype=torch.float)

    # 4) "경로별" 라벨을 만들기
    #    예: 각 track에 대해 anoms 리스트를 한 번 더 보고, has_consecutive_block(...)을 적용
    route_labels = []
    for tid, nodes_in_track in track_map.items():
        # 노드 순서 동일하게 맞춰서 anoms를 뽑으려면, 다시 sorted 해서 인덱스를 찾거나,
        # track_id_per_node와 anoms를 직접 매핑해야 함
        nodes_in_track_sorted = sorted(nodes_in_track, key=lambda n: G.nodes[n]["time"])
        # 이 노드들이 feats/anoms 내에서 어디 위치하는지 찾기
        sub_anoms = []
        for node_i in nodes_in_track_sorted:
            idx = all_nodes_sorted.index(node_i) 
            sub_anoms.append(anoms[idx])

        # sub_anoms 내에 연속 block_size 이상이 있으면 1, 없으면 0
        val = 1.0 if has_consecutive_block(sub_anoms, block_size) else 0.0
        route_labels.append(val)

    route_labels = torch.tensor(route_labels, dtype=torch.float)

    # 5) edge_index 구성
    mapping = { all_nodes_sorted[i] : i for i in range(len(all_nodes_sorted)) }
    edges = []
    for s, t in G.edges():
        if s in mapping and t in mapping:
            edges.append([mapping[s], mapping[t]])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 6) data.y = route_labels  ==> shape: [3] 가 될 것 (A,B,C 순)
    data = Data(x=x, edge_index=edge_index, y=route_labels)
    return data