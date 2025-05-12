import pandas as pd
import os
import matplotlib.pyplot as plt

# 加载数据
track_checked = pd.read_csv("Track_starchecked_with_other_plane.csv")
track_interp = pd.read_csv("Track_interpolation.csv")

# 创建输出目录
os.makedirs("flight_extracts", exist_ok=True)
os.makedirs("flight_plots", exist_ok=True)

for idx, row in track_checked.iterrows():
    time = row['time']
    callsign = row['callsign']
    parameter = row['parameter']
    identifier = f"{int(time)}_{callsign}_{parameter}"

    # 1️⃣ 提取所有早于 time 的轨迹（用于绘图）
    full_track = track_interp[
        (track_interp['callsign'] == callsign) &
        (track_interp['event_timestamp'] < time)
    ]

    if full_track.empty:
        continue

    # 2️⃣ 提取仅限过去 2 分钟的数据（用于 CSV）
    csv_track = full_track[
        full_track['event_timestamp'] >= (time - 120)
    ]

    # 保存 CSV
    csv_path = f"flight_extracts/{identifier}.csv"
    csv_track.to_csv(csv_path, index=False)

    # 3️⃣ 可视化完整轨迹（无坐标轴）
    plt.figure(figsize=(8, 6))
    plt.plot(full_track['longitude'], full_track['latitude'], color='blue', linewidth=2)

    plt.xlim(100, 108)
    plt.ylim(0, 3)
    plt.axis('off')  # 去掉坐标轴
    plt.tight_layout(pad=0)

    # 保存图片
    jpg_path = f"flight_plots/{identifier}.jpg"
    plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0)
    plt.close()
