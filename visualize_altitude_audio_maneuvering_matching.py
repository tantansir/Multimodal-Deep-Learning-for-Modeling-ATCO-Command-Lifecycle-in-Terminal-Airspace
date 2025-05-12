import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# 可视化altitude匹配结果
# 读取数据
track_data = pd.read_csv('Track_interpolation.csv')
audio_maneuvering_data = pd.read_csv('副本-altitude_audio_maneuvering_match.csv', encoding='ISO-8859-1')
commands_data = pd.read_csv('all_callsigns_altitude_commands.csv', encoding='ISO-8859-1')

# 所有 callsign 的唯一列表
# selected_callsign = track_data['callsign'].unique()
selected_callsign = ["CEB805"] #"SIA937", "SIA256", "TGW979"

for callsign in selected_callsign:
    # 筛选出当前 callsign 的轨迹数据
    track_df = track_data[track_data['callsign'] == callsign]

    # 对 Altitude 数据进行平滑处理
    track_df["fl_smooth"] = signal.savgol_filter(track_df["altitude"], 11, 5)

    # 筛选高度低于 30000 的数据
    # track_df = track_df[track_df['altitude'] < 30000]

    # 筛选出匹配的和所有的 command 数据
    matched_df = audio_maneuvering_data[audio_maneuvering_data['callsign'] == callsign]
    command_points_df = commands_data[commands_data['callsign'] == callsign]

    # 找出未匹配的 command 点
    matched_times = matched_df['time'].unique()
    unmatched_commands_df = command_points_df[~command_points_df['time'].isin(matched_times)]

    # 创建图形和子图布局
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # 第一幅图：显示匹配点
    axs[0].plot(track_df['event_timestamp'], track_df['altitude'], label=f'{callsign} Original Altitude', alpha=0.5, color="blue")
    axs[0].plot(track_df['event_timestamp'], track_df['fl_smooth'], label=f'{callsign} Smoothed Altitude', color="orange")

    # 标记 audio 和 maneuvering 的匹配时间点并显示参数值
    for idx, row in matched_df.iterrows():
        command_altitude = track_df.loc[track_df['event_timestamp'] == row['time'], 'altitude']
        maneuvering_altitude = track_df.loc[track_df['event_timestamp'] == row['maneuvering_time'], 'altitude']

        if not command_altitude.empty:
            condition_text = row['condition'] if pd.notna(row['condition']) else ""
            axs[0].plot(row['time'], command_altitude.values[0], 'ro', label="Matched Command Time" if idx == 0 else "")
            axs[0].text(row['time'], command_altitude.values[0] + 200, f"{condition_text}\n{row['parameter']}\n{row['time']}", color="red", ha='center')  # 红色代表 command 点

        if not maneuvering_altitude.empty:
            axs[0].plot(row['maneuvering_time'], maneuvering_altitude.values[0], 'go', label="Maneuvering Time" if idx == 0 else "")
            axs[0].text(row['maneuvering_time'], maneuvering_altitude.values[0] - 300, f"{row['parameter']}\n{row['maneuvering_time']}", color="green", ha='center')  # 绿色代表 maneuvering 点

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Altitude")
    axs[0].set_title(f"Altitude and Matched Maneuvering Points for Callsign: {callsign}")
    axs[0].legend()
    axs[0].grid(True)

    # 第二幅图：显示未匹配的 command 点
    axs[1].plot(track_df['event_timestamp'], track_df['altitude'], label=f'{callsign} Original Altitude', alpha=0.5, color="blue")
    axs[1].plot(track_df['event_timestamp'], track_df['fl_smooth'], label=f'{callsign} Smoothed Altitude', color="orange")

    # 标记未匹配的 command 点
    for idx, row in unmatched_commands_df.iterrows():
        command_altitude = track_df.loc[track_df['event_timestamp'] == row['time'], 'altitude']
        if not command_altitude.empty:
            condition_text = row['condition'] if pd.notna(row['condition']) else ""
            axs[1].plot(row['time'], command_altitude.values[0], 'bo', label="Unmatched Command" if idx == 0 else "")
            axs[1].text(row['time'], command_altitude.values[0] + 200, f"{condition_text}\n{row['parameter']}\n{row['time']}", color="blue", ha='center')  # 蓝色代表未匹配的 command 点

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Altitude")
    axs[1].set_title(f"Altitude with Unmatched Command Points for Callsign: {callsign}")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
