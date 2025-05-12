import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# 可视化CAS匹配结果
# 读取数据
track_data = pd.read_csv('Track_interpolation.csv')
audio_maneuvering_data = pd.read_csv('副本-CAS_audio_maneuvering_match.csv', encoding='ISO-8859-1')
commands_data = pd.read_csv('all_callsigns_CAS_commands.csv', encoding='ISO-8859-1')

# 所有 callsign 的唯一列表
# selected_callsign = track_data['callsign'].unique()
selected_callsign = ["JSA780"]

for callsign in selected_callsign:
    # 筛选出当前 callsign 的轨迹数据
    track_df = track_data[track_data['callsign'] == callsign]

    # 对 CAS 数据进行平滑处理
    cas = track_df["CAS"].values
    time = track_df["event_timestamp"].values
    spline = interpolate.UnivariateSpline(time, cas, k=5, s=7000)  # k for cubic spline, s for smoothing factor
    track_df["cas_smooth"] = spline(time)

    # 筛选高度低于 30000 的数据
    track_df = track_df[track_df['altitude'] < 30000]

    # 筛选出匹配的和所有的 command 数据
    matched_df = audio_maneuvering_data[audio_maneuvering_data['callsign'] == callsign]
    command_points_df = commands_data[commands_data['callsign'] == callsign]

    # 找出未匹配的 command 点
    matched_times = matched_df['time'].unique()
    unmatched_commands_df = command_points_df[~command_points_df['time'].isin(matched_times)]

    # 创建图形和子图布局
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # 第一幅图：显示匹配点
    axs[0].plot(track_df['event_timestamp'], track_df['CAS'], label=f'{callsign} Original CAS', alpha=0.5, color="blue")
    axs[0].plot(track_df['event_timestamp'], track_df['cas_smooth'], label=f'{callsign} Smoothed CAS', color="orange")

    # 标记 audio 和 maneuvering 的匹配时间点并显示参数值
    for idx, row in matched_df.iterrows():
        command_cas = track_df.loc[track_df['event_timestamp'] == row['time'], 'CAS']
        maneuvering_cas = track_df.loc[track_df['event_timestamp'] == row['maneuvering_time'], 'CAS']

        if not command_cas.empty:
            condition_text = row['condition'] if pd.notna(row['condition']) else ""
            axs[0].plot(row['time'], command_cas.values[0], 'ro', label="Matched Command Time" if idx == 0 else "")
            axs[0].text(row['time'], command_cas.values[0] + 10, f"{condition_text}\n{row['parameter']}\n{row['time']}", color="red", ha='center') #红色代表command点

        if not maneuvering_cas.empty:
            axs[0].plot(row['maneuvering_time'], maneuvering_cas.values[0], 'go', label="Maneuvering Time" if idx == 0 else "")
            axs[0].text(row['maneuvering_time'], maneuvering_cas.values[0] - 20, f"{row['parameter']}\n{row['maneuvering_time']}", color="green", ha='center') #绿色代表maneuvering点

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("CAS")
    axs[0].set_title(f"CAS and Matched Maneuvering Points for Callsign: {callsign}")
    axs[0].legend()
    axs[0].grid(True)

    # 第二幅图：显示未匹配的 command 点
    axs[1].plot(track_df['event_timestamp'], track_df['CAS'], label=f'{callsign} Original CAS', alpha=0.5, color="blue")
    axs[1].plot(track_df['event_timestamp'], track_df['cas_smooth'], label=f'{callsign} Smoothed CAS', color="orange")

    # 标记未匹配的 command 点
    for idx, row in unmatched_commands_df.iterrows():
        command_cas = track_df.loc[track_df['event_timestamp'] == row['time'], 'CAS']
        if not command_cas.empty:
            condition_text = row['condition'] if pd.notna(row['condition']) else ""
            axs[1].plot(row['time'], command_cas.values[0], 'bo', label="Unmatched Command" if idx == 0 else "")
            axs[1].text(row['time'], command_cas.values[0] + 10, f"{condition_text}\n{row['parameter']}\n{row['time']}", color="blue", ha='center') #蓝色代表未匹配到的command点

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("CAS")
    axs[1].set_title(f"CAS with Unmatched Command Points for Callsign: {callsign}")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
