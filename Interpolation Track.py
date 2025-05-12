import numpy as np
import pandas as pd


def linear_interpolation_1s(df_latlon, res=1):
    # 保留原始顺序并保留所有列
    temp_latlon = df_latlon.copy()

    # 首尾行不做更改
    first_row = temp_latlon.iloc[0]
    last_row = temp_latlon.iloc[-1]

    # 提取需要对中间部分进行插值的列
    time = temp_latlon["event_timestamp"][1:-1]
    lat = temp_latlon["latitude"][1:-1]
    lon = temp_latlon["longitude"][1:-1]
    alt = temp_latlon["altitude"][1:-1]
    heading = temp_latlon["derived_heading"][1:-1]
    speed = temp_latlon["derived_speed"][1:-1]
    cas = temp_latlon["CAS"][1:-1]

    # 为中间部分生成新的时间序列，间隔为1秒
    if len(time) > 1:
        time_new = np.arange(time.min(), time.max() + res, res)

        # 对中间部分的每个变量进行线性插值
        lat_new = np.interp(time_new, time, lat)
        lon_new = np.interp(time_new, time, lon)
        alt_new = np.interp(time_new, time, alt)
        heading_new = np.interp(time_new, time, heading)
        speed_new = np.interp(time_new, time, speed)
        cas_new = np.interp(time_new, time, cas)

        # 创建中间部分插值后的DataFrame
        interpolated_df = pd.DataFrame({
            "event_timestamp": time_new,
            "latitude": lat_new,
            "longitude": lon_new,
            "altitude": alt_new,
            "derived_heading": heading_new,
            "derived_speed": speed_new,
            "CAS": cas_new
        })

        # 合并首行、插值后的中间部分和尾行
        result_df = pd.concat([first_row.to_frame().T, interpolated_df, last_row.to_frame().T], ignore_index=True)
    else:
        # 如果中间部分没有足够的数据进行插值，则保持原样
        result_df = temp_latlon

    # 添加未插值的其他列，保留原始值
    for col in temp_latlon.columns:
        if col not in ["event_timestamp", "latitude", "longitude", "altitude", "derived_heading", "derived_speed",
                       "CAS"]:
            result_df[col] = temp_latlon[col].iloc[0]

    return result_df


# 利用Track.csv
df = pd.read_csv('Track.csv')

# Group by 'callsign' and apply the interpolation function to each group
df_interpolated = df.groupby('callsign', group_keys=False, sort=False).apply(linear_interpolation_1s).reset_index(drop=True)

# 得到Track_interpolation.csv
output_file_path = 'Track_interpolation.csv'
df_interpolated.to_csv(output_file_path, index=False)