import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
track_checked = pd.read_csv("Track_starchecked_with_other_plane.csv")
track_interp = pd.read_csv("Track_interpolation.csv")

# 输出路径
os.makedirs("airspace_snapshot", exist_ok=True)

# 映射范围
lon_min, lon_max = 100, 108
lat_min, lat_max = 0, 3

# 速度线段缩放系数（可调）
speed_scale = 0.0005  # km/h to degrees (大概合适线长)

for idx, row in track_checked.iterrows():
    time = row['time']
    current_callsign = row['callsign']
    parameter = row['parameter']
    identifier = f"{int(time)}_{current_callsign}_{parameter}"

    # 提取当前时间所有飞机
    snapshot = track_interp[track_interp['event_timestamp'] == time]
    if snapshot.empty:
        continue

    # 初始化图像
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.axis('off')

    for _, ac in snapshot.iterrows():
        lon = ac['longitude']
        lat = ac['latitude']
        heading = ac['derived_heading']
        speed = ac['CAS']
        cs = ac['callsign']

        # 计算方向向量（以当前位置为线尾）
        angle_rad = np.radians(heading)
        dx = speed_scale * speed * np.sin(angle_rad)
        dy = speed_scale * speed * np.cos(angle_rad)

        x_end = lon + dx
        y_end = lat + dy

        color = 'red' if cs == current_callsign else 'blue'
        ax.plot([lon, x_end], [lat, y_end], color=color, linewidth=2, alpha=0.9)

        # color = 'red' if cs == current_callsign else 'blue'
        # ax.arrow(lon, lat, dx, dy, head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.8)

    # 保存图像
    save_path = f"airspace_snapshot/{identifier}.jpg"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
