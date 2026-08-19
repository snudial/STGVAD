from data.loader import load_data, interpolate_data
from cluster.optics import optics_clustering, map_clusters_to_prev_data
from graph.builder import process_single_cluster

from graph.io import save_graphs_to_pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def prepare_data_pipeline():
    year_list = [2018, 2019, 2020]
    df = load_data(year_list)
    df_interp = interpolate_data(df)
    track_counts = df_interp["Track_ID"].value_counts()
    exclude_ids = track_counts[track_counts < 10].index
    df_filtered = df_interp[~df_interp["Track_ID"].isin(exclude_ids)].copy()
    return df_filtered

def create_cluster_graphs(data, h=10, window_step_hours=1, desc="Processing", max_workers=120, apply_h_k_constraints=True, k=3):
    data = data.copy()
    start_time = data["TIMESTAMP"].min()
    end_time = data["TIMESTAMP"].max()
    window_step = pd.Timedelta(hours=window_step_hours)
    total_steps = int((end_time - (start_time + pd.Timedelta(hours=h))) // window_step) + 1
    nx_graph_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(total_steps), desc=f"{desc} - Clustering"):
            current_target = start_time + pd.Timedelta(hours=h) + i * window_step
            try:
                target_data, cluster_map = optics_clustering(data, current_target)
                prev_data = map_clusters_to_prev_data(data, cluster_map, current_target, h)
                if apply_h_k_constraints:
                    prev_data = prev_data.groupby("Track_ID").filter(lambda x: len(x) >= h)
                cluster_args = []
                cnums = prev_data["cluster"].dropna().unique()
                if len(cnums) == 0:
                    continue
                for cnum in cnums:
                    cnum = int(cnum)
                    cdata = prev_data[prev_data["cluster"] == cnum]
                    unique_tracks = cdata["Track_ID"].unique()
                    if len(unique_tracks) < k:
                        continue
                    selected = np.random.choice(unique_tracks, k, replace=False)
                    cdata = cdata[cdata["Track_ID"].isin(selected)]
                    cluster_args.append((cdata, cnum))
                graphs = list(executor.map(process_single_cluster, cluster_args))
                for g in graphs:
                    if g is not None:
                        nx_graph_list.append(g)
            except ValueError:
                continue
    return nx_graph_list

def run_pipeline_and_save(path="storage/nx_graphs_original.pkl"):
    df_filtered = prepare_data_pipeline()
    nx_list = create_cluster_graphs(df_filtered, h=10, window_step_hours=1, k=3)
    save_graphs_to_pickle(nx_list, path)
