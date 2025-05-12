import pandas as pd
import matplotlib.pyplot as plt

# 可视化heading匹配结果
# 读取数据
track_data = pd.read_csv('Track_interpolation.csv')
audio_maneuvering_data = pd.read_csv('副本-heading_audio_maneuvering_match.csv', encoding='ISO-8859-1')
commands_data = pd.read_csv('all_callsigns_heading_commands.csv', encoding='ISO-8859-1')

# 所有 callsign 的唯一列表
# selected_callsign = track_data['callsign'].unique()
selected_callsign = ["HYT9033"]

for callsign in selected_callsign:

    # 筛选出当前 callsign 的轨迹数据
    track_df = track_data[track_data['callsign'] == callsign]
    track_df["heading_smooth"] = track_df["derived_heading"]

    # 筛选高度低于 30000 的数据
    # track_df = track_df[track_df['altitude'] < 30000]

    # 筛选出匹配的和所有的 command 数据
    matched_df = audio_maneuvering_data[audio_maneuvering_data['callsign'] == callsign]
    command_points_df = commands_data[commands_data['callsign'] == callsign]

    # 找出未匹配的 command 点
    matched_times = matched_df['time'].unique()
    unmatched_commands_df = command_points_df[~command_points_df['time'].isin(matched_times)]

    print(matched_df)
    # 如果没有匹配点或未匹配点，跳过当前 callsign
    if matched_df.empty or (matched_df['time_offset'].abs() > 2).all():
        print(f"No data to display for callsign {callsign}, skipping...")
        continue

    # 创建图形和子图布局
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # 第一幅图：显示匹配点
    axs[0].plot(track_df['event_timestamp'], track_df['derived_heading'], label=f'{callsign} Heading', alpha=0.5, color="blue")
    axs[0].plot(track_df['event_timestamp'], track_df['heading_smooth'], label=f'{callsign} Smoothed Altitude', color="orange")

    # 标记 audio 和 maneuvering 的匹配时间点并显示参数值
    for idx, row in matched_df.iterrows():
        command_heading = track_df.loc[track_df['event_timestamp'] == row['time'], 'derived_heading']
        maneuvering_heading = track_df.loc[track_df['event_timestamp'] == row['maneuvering_time'], 'derived_heading']

        if not command_heading.empty:
            condition_text = row['condition'] if pd.notna(row['condition']) else ""
            axs[0].plot(row['time'], command_heading.values[0], 'ro', label="Matched Command Time" if idx == 0 else "")
            axs[0].text(row['time'], command_heading.values[0] + 5,
                        f"{condition_text}\n{row['parameter']}\n{row['time']}", color="red",
                        ha='center')  # 红色代表 command 点

        if not maneuvering_heading.empty:
            axs[0].plot(row['maneuvering_time'], maneuvering_heading.values[0], 'go',
                        label="Maneuvering Time" if idx == 0 else "")
            axs[0].text(row['maneuvering_time'], maneuvering_heading.values[0] - 5,
                        f"{row['parameter']}\n{row['maneuvering_time']}", color="green",
                        ha='center')  # 绿色代表 maneuvering 点

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Heading")
    axs[0].set_title(f"Heading and Matched Maneuvering Points for Callsign: {callsign}")
    axs[0].legend()
    axs[0].grid(True)

    # 第二幅图：显示未匹配的 command 点
    axs[1].plot(track_df['event_timestamp'], track_df['derived_heading'], label=f'{callsign} Original Heading',
                alpha=0.5, color="blue")
    axs[1].plot(track_df['event_timestamp'], track_df['heading_smooth'], label=f'{callsign} Smoothed Heading',
                color="orange")

    # 标记未匹配的 command 点
    for idx, row in unmatched_commands_df.iterrows():
        command_heading = track_df.loc[track_df['event_timestamp'] == row['time'], 'derived_heading']
        if not command_heading.empty:
            condition_text = row['condition'] if pd.notna(row['condition']) else ""
            axs[1].plot(row['time'], command_heading.values[0], 'bo', label="Unmatched Command" if idx == 0 else "")
            axs[1].text(row['time'], command_heading.values[0] + 5,
                        f"{condition_text}\n{row['parameter']}\n{row['time']}", color="blue",
                        ha='center')  # 蓝色代表未匹配的 command 点

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Heading")
    axs[1].set_title(f"Heading with Unmatched Command Points for Callsign: {callsign}")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
