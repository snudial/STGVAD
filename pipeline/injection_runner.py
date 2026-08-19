from graph.io import load_graphs_from_pickle, save_graphs_to_pickle
from graph.inject_anomaly import extract_route_graphs, force_route_graph_injection, group_routes_by_cluster_id
from graph.convert import cluster_to_pyg_data


def inject_graphs_and_save(original_path, route_ratio, node_ratio):
    """
    클러스터 그래프에서 route-level 추출 후 이상치 주입하고 저장
    """
    injected_path = f"storage/nx_graphs_injected/nx_graphs_injected_r={route_ratio}_n={node_ratio}.pkl"

    # 1) Load original graphs
    nx_graph_list = load_graphs_from_pickle(original_path)
    print(f"[INFO] Loaded {len(nx_graph_list)} original Nx graphs")

    # 2) Extract routes
    route_graphs = extract_route_graphs(nx_graph_list)
    print(f"[INFO] Extracted {len(route_graphs)} route graphs")

    # 3) Inject anomalies
    injected_routes = force_route_graph_injection(
        route_graph_list=route_graphs,
        route_graph_ratio=route_ratio,
        node_ratio=node_ratio,
        k=3.5
    )
    print(f"[INFO] Anomaly injection with route_ratio = {route_ratio}, node_ratio = {node_ratio} complete")

    # 4) Cluster routes 
    injected_graphs_cluster = group_routes_by_cluster_id(injected_routes)
    print("[INFO] Clustering injected route graphs complete")

    # 5) Save
    save_graphs_to_pickle(injected_graphs_cluster, injected_path)
    print(f"[INFO] Injected graphs cluster saved to {injected_path}")


def load_injected_graphs_and_convert(route_ratio, node_ratio):
    """
    주입된 그래프 로드 후 PyG 형식으로 변환
    """
    path = f"storage/nx_graphs_injected/nx_graphs_injected_r={route_ratio}_n={node_ratio}.pkl"
    nx_list = load_graphs_from_pickle(path)
    pyg_list = []
    for G in nx_list:
        data_obj = cluster_to_pyg_data(G, block_size=10*node_ratio)
        if data_obj is not None:
            pyg_list.append(data_obj)
    return pyg_list
