import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime 
from scipy.interpolate import interp1d


def interpolate_nans(padata, pkind='linear'):
    """
    see: https://stackoverflow.com/a/53050216/2167159
    """
    aindexes = np.arange(padata.shape[0])
    agood_indexes, = np.where(np.isfinite(padata))
    f = interp1d(agood_indexes
               , padata[agood_indexes]
               , bounds_error=False
               , copy=False
               , fill_value="extrapolate"
               , kind=pkind)
    return f(aindexes)

def date_encode(date):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    return (
        np.sin(2 * np.pi * date.timetuple().tm_yday / 366),
        np.cos(2 * np.pi * date.timetuple().tm_yday / 366),
    )

def loadXY(
    df,
    window_size=25 * 7 , 
    target_size=6,
    fuse_past=True, 
    return_fips=False, 
    encode_season=True,
    stride = 25 * 7, 
):
    soil_df = pd.read_csv("/kaggle/input/drought-dataset-bychristoph/soil_data.csv")
    time_data_cols = sorted(
        [c for c in df.columns if c not in ["fips", "date", "score"]]
    )
    static_data_cols = sorted(
        [c for c in soil_df.columns if c not in ["fips", "lat", "lon"]]
    )
    count = 0
    score_df = df.dropna(subset=["score"])
    X_static = np.empty(((len(df) - window_size) // stride, len(static_data_cols)))
    X_fips_date = []
    add_dim = 0
    if fuse_past:
        add_dim += 1
    if encode_season:
        add_dim += 2
    X_time = np.empty(
        ((len(df) - window_size) // stride, window_size, len(time_data_cols) + add_dim)
    )
    y_target = np.empty(((len(df) - window_size) // stride, target_size))
    for fips in tqdm(score_df.index.get_level_values(0).unique()):
        fips_df = df[(df.index.get_level_values(0) == fips)]
        X = fips_df[time_data_cols].values
        y = fips_df["score"].values
        X_s = soil_df[soil_df["fips"] == fips][static_data_cols].values[0]
        start_i = np.where(~np.isnan(y))[0][0]  # Thứ 3 đầu tiên 
        for i in range(start_i, len(y) - (window_size + target_size * 7), stride):
            X_fips_date.append((fips, fips_df.index[i : i + window_size][-1]))
            current_index = 0 
            X_time[count, :, current_index : len(time_data_cols)] = X[i : i + window_size]
            current_index +=  len(time_data_cols)
            if  fuse_past:
                X_time[count, :, current_index] = interpolate_nans(y[i : i + window_size])
                current_index += 1
            if encode_season:
                enc_dates = [date_encode(d) for f, d in fips_df.index[i : i + window_size].values]
                d_sin, d_cos = [s for s, c in enc_dates], [c for s, c in enc_dates]
                X_time[count, :, current_index ] = d_sin
                X_time[count, :, current_index + 1] = d_cos
            temp_y = y[i + window_size : i + window_size + target_size * 7]
            y_target[count] = np.array(temp_y[~np.isnan(temp_y)][:target_size])
            X_static[count] = X_s
            count += 1
    print(f"loaded {count} samples")
    results = [X_static[:count], X_time[:count], y_target[:count]]
    if return_fips:
        results.append(X_fips_date)
    return results