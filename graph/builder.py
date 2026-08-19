import pandas as pd
import networkx as nx

def get_cluster_data(prev_df, cluster_number):
    return prev_df[prev_df["cluster"] == cluster_number]

def build_temporal_graph_unify_last_as_T_all_future(
    df,
    time_col="TIMESTAMP",
    track_col="Track_ID",
    feature_cols=None,
    use_vessel_mapping=True
):
    """
    클러스터 구간의 AIS 레코드를 시간 노드로 바꾼 NetworkX 그래프를 만든다.

    노드 id 는 "{track}_{t 또는 t-k}" 형태이고, 노드 속성으로 feature_cols 값을 싣는다.
    엣지는 여기서 만들지 않는다 — 이상치 주입 뒤 graph.convert.create_cluster_edges 가
    항적 내/항적 간 규칙에 맞춰 한 번에 구성한다.
    """
    if feature_cols is None:
        feature_cols = []
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df.dropna(subset=[time_col], inplace=True)
    
    unique_tracks = sorted(df[track_col].unique())
    if use_vessel_mapping:
        vessel_mapping = {}
        for i, tid in enumerate(unique_tracks):
            if i < 26:
                vessel_mapping[tid] = chr(65 + i)
            else:
                vessel_mapping[tid] = str(i)
    else:
        vessel_mapping = {t: t for t in unique_tracks}
    
    df["_mapped_id_"] = df[track_col].map(vessel_mapping)
    
    node_info = []
    for mid, sub_df in df.groupby("_mapped_id_"):
        sub_df = sub_df.sort_values(time_col)
        last_time = sub_df[time_col].max()
        for _, row in sub_df.iterrows():
            diff_hours = (last_time - row[time_col]).total_seconds() / 3600.0
            offset = int(round(diff_hours))
            label = "t" if offset == 0 else f"t-{offset}"
            node_id = f"{mid}_{label}"
            node_attrs = {"track_id": mid, "time": row[time_col], "offset": offset}
            for c in feature_cols:
                node_attrs[c] = row.get(c, None)
            node_info.append((node_id, offset, mid, node_attrs))
    
    G = nx.DiGraph()
    for node_id, offset, mid, attrs in node_info:
        G.add_node(node_id, **attrs)
    
    return G, vessel_mapping

def process_single_cluster(args):
    """
    injection X, Nx 그래프 생성
    """
    (prev_data, cluster_number) = args
    cluster_data = get_cluster_data(prev_data, cluster_number)
    if cluster_data.empty:
        return None
    G, _ = build_temporal_graph_unify_last_as_T_all_future(
        cluster_data,
        time_col="TIMESTAMP",
        track_col="Track_ID",
        feature_cols=["LON","LAT","COURSE","SPEED","COURSE_SIN","COURSE_COS","ANOMALY"],
        use_vessel_mapping=True
    )
    return G