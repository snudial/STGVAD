import os
import pickle

def save_graphs_to_pickle(graphs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"[INFO] 저장 완료: {path}")

def load_graphs_from_pickle(path):
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    print(f"[INFO] 불러오기 완료: {path}")
    return graphs
