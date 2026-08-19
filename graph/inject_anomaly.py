import random
import math
import numpy as np
import networkx as nx
from math import sin, cos, radians
from tqdm import tqdm
from collections import defaultdict

from graph.convert import create_cluster_edges


""" MARK - 클러스터 그래프에서 route 단위로 분리 """
def extract_route_graphs(nx_graph_list):
    """
    각 cluster-level Nx 그래프에서 route 단위 (Track_ID 별) 서브그래프를 추출하여 반환
    """
    route_graphs = []
    for cluster_idx, G in enumerate(nx_graph_list): 
        track_map = {}
        for n, attr in G.nodes(data=True):
            tid = attr.get("track_id")
            if tid not in track_map:
                track_map[tid] = []
            track_map[tid].append(n)
        for tid, node_list in track_map.items():
            if len(node_list) < 1:
                continue
            H = nx.DiGraph()
            for n in node_list:
                H.add_node(n, **G.nodes[n])
            for s, t in G.edges():
                if s in node_list and t in node_list:
                    H.add_edge(s, t)
            H.graph["cluster_id"] = cluster_idx  # ✅ 클러스터 번호 저장
            route_graphs.append(H)
    return route_graphs


""" MARK - 	주어진 route에서 속도/회전 변화율 통계 계산 """
def compute_motion_statistics(H, node_list):
    dt_list, da_list, domega_list = [], [], []
    for j in range(1, len(node_list)):
        t_prev = H.nodes[node_list[j - 1]].get("time")
        t_curr = H.nodes[node_list[j]].get("time")
        if t_prev is None or t_curr is None:
            continue
        dt_sec = (t_curr - t_prev).total_seconds()
        if dt_sec <= 0:
            continue
        spd_prev = H.nodes[node_list[j - 1]].get("SPEED", 0.0)
        spd_curr = H.nodes[node_list[j]].get("SPEED", 0.0)
        crs_prev = H.nodes[node_list[j - 1]].get("COURSE", 0.0)
        crs_curr = H.nodes[node_list[j]].get("COURSE", 0.0)
        da_list.append((spd_curr - spd_prev) / dt_sec)
        domega_list.append((crs_curr - crs_prev) / dt_sec)
    return np.mean(da_list), np.std(da_list), np.mean(domega_list), np.std(domega_list)


""" MARK - 특정 시작점부터 노드 블록에 이상치 주입 """
def apply_anomaly_block(H, node_list, start_idx, block_size,
                        mu_a, sigma_a, mu_omega, sigma_omega, k):
    """
    주어진 start_idx부터 block_size개 노드를 '연속'으로 이상치로 만들어 준다.
    (업데이트가 누적되지 않도록, 업데이트 전에 원본을 따로 저장한다.)
    """
    # 1. 모든 노드를 일단 False로 초기화
    for n in node_list:
        H.nodes[n]["ANOMALY"] = False

    # 2. 원본 speed/course를 따로 저장해 둔다
    original_values = {}
    for n in node_list:
        original_values[n] = (
            H.nodes[n]["SPEED"],
            H.nodes[n]["COURSE"]
        )

    # 3. 블록 내 노드들을 업데이트
    end_idx = min(start_idx + block_size, len(node_list))

    for j in range(start_idx, end_idx):

        curr_n = node_list[j]
        prev_n = node_list[j - 1]

        # "이전 노드의 원본값"을 참조
        spd_prev_orig  = original_values[prev_n][0]
        crs_prev_orig  = original_values[prev_n][1]

        t_curr = H.nodes[curr_n].get("time")
        t_prev = H.nodes[prev_n].get("time")
        if not t_curr or not t_prev:
            continue

        dt_sec = (t_curr - t_prev).total_seconds()
        if dt_sec <= 0:
            continue

        # 여기서 a_star, omega_star는 이상치 크기
        a_star = mu_a + k * sigma_a
        omega_star = mu_omega + k * sigma_omega

        # "갱신된 speed/course"가 아니라, "원본 spd_prev_orig" 기준으로 계산
        new_speed = spd_prev_orig + a_star * dt_sec
        new_course = (crs_prev_orig + omega_star * dt_sec) % 360

        H.nodes[curr_n]["SPEED"]      = new_speed
        H.nodes[curr_n]["COURSE"]     = new_course
        H.nodes[curr_n]["COURSE_SIN"] = sin(radians(new_course))
        H.nodes[curr_n]["COURSE_COS"] = cos(radians(new_course))
        H.nodes[curr_n]["ANOMALY"]    = True

    return H

""" MARK - 전체 route 중 일부를 선택해 이상 주입 실행 """
def force_route_graph_injection(route_graph_list, route_graph_ratio=0.3, node_ratio=0.5, k=3.5):
    """
    route graph 중 일부에 anomaly를 주입
    - route_graph_ratio: 전체 중 anomaly로 만들 route 비율
    - node_ratio: 한 route에서 anomaly가 될 노드 비율
    """
    random.shuffle(route_graph_list)
    n_total = len(route_graph_list)
    n_anomaly = int(math.ceil(route_graph_ratio * n_total))
    if n_anomaly < 1:
        return route_graph_list

    for i in tqdm(range(n_anomaly), desc="Injecting anomalies into routes"):
        H = route_graph_list[i]
        node_list = sorted(H.nodes(), key=lambda n: H.nodes[n].get("time", 0))
        n_nodes = len(node_list)
        if n_nodes < 2:
            continue

        mu_a, sigma_a, mu_omega, sigma_omega = compute_motion_statistics(H, node_list)
        block_size = int(math.ceil(node_ratio * n_nodes))	
        if block_size < 1 or block_size > n_nodes:
            continue
        start_idx = random.randint(1, n_nodes - block_size)
        H = apply_anomaly_block(H, node_list, start_idx, block_size, mu_a, sigma_a, mu_omega, sigma_omega, k)
        route_graph_list[i] = H

    return route_graph_list

def group_routes_by_cluster_id(route_graphs):
    """
    route-level 그래프들을 cluster_id 기준으로 다시 묶어서 클러스터 단위로 복원
    + edge는 create_cluster_edges() 기준으로 재정의
    """
    clusters_dict = defaultdict(list)
    for G in route_graphs:
        cluster_id = G.graph.get("cluster_id", -1)
        clusters_dict[cluster_id].append(G)

    combined_clusters = []
    for cluster_id in sorted(clusters_dict.keys()):
        routes = clusters_dict[cluster_id]
        if len(routes) != 3:
            continue  # 정확히 3개의 route가 있어야 함

        G_combined = nx.DiGraph()
        for G in routes:
            G_combined.add_nodes_from(G.nodes(data=True))

        # ⚠️ 기존 엣지 추가 X (G.edges) → 대신 우리가 정의한 방식으로 edge 생성
        custom_edges = create_cluster_edges(G_combined)
        G_combined.add_edges_from(custom_edges)

        combined_clusters.append(G_combined)

    return combined_clusters
