# =============================================================================
# Data Loading and Preprocessing Functions
# =============================================================================
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd

# OMTAD 데이터 루트. 다른 위치에 두었다면 환경변수로 덮어쓸 수 있다.
#   export OMTAD_ROOT="/path/to/OMTAD/West Grid"
OMTAD_ROOT = os.environ.get("OMTAD_ROOT", os.path.join("data", "OMTAD", "West Grid"))


def load_data(year_list):
    """
    주어진 연도들의 전체 월(cargo, passenger, tanker) AIS CSV를 읽어 하나의 DataFrame으로 합친다.
    """
    columns = ["CRAFT_ID", "LON", "LAT", "COURSE", "SPEED", "TIMESTAMP", "Track_ID"]
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_mapping = {
        "Jan":("Jan","Jan"), "Feb":("Feb","Feb"), "Mar":("Mar","Mar"),
        "Apr":("Apr","Apr"), "May":("May","May"), "Jun":("Jun","Jun"),
        "Jul":("Jul","Jul"), "Aug":("Aug","Aug"), "Sep":("Sept","Sep"),
        "Oct":("Oct","Oct"), "Nov":("Nov","Nov"), "Dec":("Dec","Dec")
    }
    dfs = []
    for year in year_list:
        base_path = os.path.join(OMTAD_ROOT, str(year))
        types = ["cargo", "passenger", "tanker"]
        
        for t in types:
            for m in months:
                f_m, f_n = month_mapping[m]
                file_path = os.path.join(
                    base_path, t, f_m, f"MPF_{f_n}_{year}_Grid_{t.capitalize()}.csv"
                )
                if os.path.exists(file_path):
                    tmp = pd.read_csv(file_path, names=columns, skiprows=1)
                    tmp["TYPE"] = t
                    tmp["MONTH"] = m
                    dfs.append(tmp)
                else:
                    print(f"[WARN] 파일 없음: {file_path}")
    if not dfs:
        raise ValueError(
            f"{year_list} 데이터를 하나도 읽지 못했습니다. OMTAD_ROOT={OMTAD_ROOT} 경로를 확인하세요."
        )
    df = pd.concat(dfs, ignore_index=True)
    df.dropna(inplace=True)
    
    # TIMESTAMP 변환
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df.dropna(subset=["TIMESTAMP"], inplace=True)
    df["TIMESTAMP"] = df["TIMESTAMP"].dt.round("1H")
    
    # COURSE -> sin, cos
    df["COURSE_SIN"] = np.sin(np.deg2rad(df["COURSE"]))
    df["COURSE_COS"] = np.cos(np.deg2rad(df["COURSE"]))
    
    df["Track_ID"] = df["Track_ID"].astype(str)
    return df

def interpolate_group(group):
    """Track_ID별 1시간 간격 보간"""
    track_id = group["Track_ID"].iloc[0]
    craft_id = group["CRAFT_ID"].iloc[0]
    group = group.set_index("TIMESTAMP")
    group = group[~group.index.duplicated(keep="first")]
    group = group.resample("1H").interpolate()
    group["Track_ID"] = track_id
    group["CRAFT_ID"] = group["CRAFT_ID"].fillna(craft_id).astype(float).astype(int)
    if "ANOMALY" not in group.columns:
        group["ANOMALY"] = False
    else:
        group["ANOMALY"] = group["ANOMALY"].fillna(False)
    return group

def interpolate_data(df):
    return df.groupby("Track_ID", group_keys=False).apply(interpolate_group).reset_index()