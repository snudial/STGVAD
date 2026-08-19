from sklearn.cluster import OPTICS
import pandas as pd

def optics_clustering(df, target_datetime, min_samples=3, xi=0.05, min_cluster_size=0.05):
    target_df = df[df["TIMESTAMP"] == target_datetime].copy()
    coords = target_df[["LAT", "LON"]].values
    opt = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    labels = opt.fit_predict(coords)
    target_df["cluster"] = labels
    cluster_map = target_df[["Track_ID", "cluster"]].drop_duplicates()
    return target_df, cluster_map

def map_clusters_to_prev_data(df, cluster_map, target_datetime, h=10):
    prev_df = df[
        (df["TIMESTAMP"] >= target_datetime - pd.Timedelta(hours=h)) &
        (df["TIMESTAMP"] < target_datetime)
    ].copy()
    prev_df = prev_df.merge(cluster_map, on="Track_ID", how="left")
    return prev_df