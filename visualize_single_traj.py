import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV
df = pd.read_csv("Track_interpolation.csv")
df = df.sort_values(by=["callsign", "event_timestamp"])

# 仅保留CPA759
df_cpa759 = df[df["callsign"] == "CPA759"]

# 创建图像
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制CPA759的轨迹
ax.plot(df_cpa759["longitude"], df_cpa759["latitude"], color='blue', linewidth=0.7)


# 设置图例与范围
ax.set_xlim(102.8, 105.2)
ax.set_ylim(0, 2.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(["trajectory"], loc="upper right")

# 去除图例、网格、标题等
plt.grid(False)
plt.tight_layout()
plt.show()
