import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV
df = pd.read_csv("Track_interpolation.csv")  # 替换为你的CSV文件名
df = df.sort_values(by=["callsign", "event_timestamp"])

# 设置机场位置（你可以根据实际机场经纬度修改）
airport_lat = 1.35019  # 新加坡樟宜机场纬度
airport_lon = 103.994  # 新加坡樟宜机场经度1.35019	103.994


# 创建图像
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制每一条轨迹
for callsign, group in df.groupby("callsign"):
    ax.plot(group["longitude"], group["latitude"], color='blue', linewidth=0.7)

# 画机场跑道点
ax.plot(airport_lon, airport_lat, marker='>', color='red', markersize=10)
ax.text(airport_lon - 0.115, airport_lat + 0.045, "airport", color='red', fontsize=10)

# 画同心圆（距离以NM为单位），文字在左侧
def draw_range_rings(ax, center_lon, center_lat, rings=[10, 20, 30, 40, 50, 60]):
    for r in rings:
        radius_deg = r * 0.015  # 粗略换算比例
        circle = plt.Circle((center_lon, center_lat), radius_deg,
                            color='cyan', fill=False, linestyle='-', alpha=0.5)
        ax.add_patch(circle)
        ax.text(center_lon - radius_deg, center_lat, str(r), fontsize=8, ha='right')

draw_range_rings(ax, airport_lon, airport_lat)

# 文字说明
ax.text(airport_lon - 0.8, airport_lat + 0.1, "dist to airport (NM)", fontsize=10)

# 设置图例与范围
ax.set_xlim(102.8, 105.2)
ax.set_ylim(0, 2.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(["trajectory"], loc="upper right")

plt.grid(False)
plt.tight_layout()
plt.show()