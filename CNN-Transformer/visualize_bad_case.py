import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 中文支持和负号处理
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 自定义颜色
palette = {"underestimated": "tab:blue", "overestimated": "tab:orange"}

# 读取数据
file_path = "ensemble_predictions.csv"
df = pd.read_csv(file_path)
underestimated = df[df["sum"] > 10]
overestimated = df[df["sum"] < -10]

# 添加标签
underestimated["label"] = "underestimated"
overestimated["label"] = "overestimated"
df_filtered = pd.concat([underestimated, overestimated])

# ========== Boxplot：time offset / duration ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df_filtered, x="label", y="time_offset", ax=axes[0], palette=palette)
axes[0].set_title("Time Offset 分布")
sns.boxplot(data=df_filtered, x="label", y="duration", ax=axes[1], palette=palette)
axes[1].set_title("Duration 分布")
plt.tight_layout()
plt.show()

# ========== 空域位置分布 ==========
plt.figure(figsize=(6, 6))
sns.scatterplot(data=df_filtered, x="x", y="y", hue="label", alpha=0.6, palette=palette)
plt.title("空域位置分布对比")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# ========== 航司行为差异 ==========
df_filtered["airline"] = df_filtered["callsign"].astype(str).str.extract(r"([A-Z]{2,3})")
plt.figure(figsize=(12, 5))
sns.countplot(data=df_filtered, x="airline", hue="label",
              order=df_filtered["airline"].value_counts().index, palette=palette)
plt.title("航司分布对比")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== 指令类型占比（flight level / head / velocity） ==========
labels = ['flight level', 'head', 'velocity']
under_props = underestimated[labels].mean()
over_props = overestimated[labels].mean()

x = range(len(labels))
bar_width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x, under_props, width=bar_width, label='低估样本', alpha=0.8, color="tab:blue")
plt.bar([i + bar_width for i in x], over_props, width=bar_width, label='高估样本', alpha=0.8, color="tab:orange")
plt.xticks([i + bar_width / 2 for i in x], labels)
plt.ylabel("占比")
plt.ylim(0, 1)
plt.title("指令类型占比对比")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
