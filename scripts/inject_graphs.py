import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.injection_runner import inject_graphs_and_save

if __name__ == "__main__":

    route_ratio = float(sys.argv[1])
    node_ratio = float(sys.argv[2])

    original_path = "storage/nx_graphs_original.pkl"
    
    inject_graphs_and_save(
        original_path=original_path,
        route_ratio=route_ratio,
        node_ratio=node_ratio
    )