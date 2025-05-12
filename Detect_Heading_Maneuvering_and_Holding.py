import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
from math import radians, sin, cos, sqrt, atan2
import os


def heading_changes(df, win):
    """
    检测出了所有大于45秒的平台，又基于平台检测出了所有大于150度的转角
    """
    fl_df = df.copy()

    # 后来发现不需要平滑,也不用转化成弧度
    # Convert heading to sin and cos components
    # fl_df["sin_heading"] = np.sin(np.radians(fl_df["derived_heading"]))
    # fl_df["cos_heading"] = np.cos(np.radians(fl_df["derived_heading"]))
    #
    # # Apply Savitzky-Golay filter to smooth both sin and cos components
    # fl_df["sin_smooth"] = signal.savgol_filter(fl_df["sin_heading"], win, 5)
    # fl_df["cos_smooth"] = signal.savgol_filter(fl_df["cos_heading"], win, 5)
    #
    # # Convert smoothed sin and cos back to heading (in degrees)
    # fl_df["heading_smooth"] = np.degrees(np.arctan2(fl_df["sin_smooth"], fl_df["cos_smooth"]))
    # fl_df["heading_smooth"] = np.mod(fl_df["heading_smooth"], 360)

    fl_df["heading_smooth"] = fl_df["derived_heading"]

    # Sliding window parameters 45 5 2.5 2.5
    window_size = 45
    midpoints = np.arange(0, 365, 5)  # Heading has midpoints every 5 degrees
    lower_bounds = midpoints[:-1] + 2.5
    upper_bounds = midpoints[1:] + 2.5

    print(lower_bounds)
    print(upper_bounds)

    platforms = []
    for i in range(len(fl_df) - window_size):
        window_data = fl_df['heading_smooth'].iloc[i:i + window_size]
        window_data_corrected = np.where(window_data > 358, window_data - 360, window_data)

        # Histogram for detecting stable platforms in heading
        fl_hist = np.histogram(window_data_corrected, bins=np.concatenate([lower_bounds - 360, lower_bounds]))
        counts = fl_hist[0]

        max_count = np.max(counts)
        if max_count >= (window_size * 0.92):  # 判断往后一个时间窗口内的95％的点都在区间内
            platform_bin = np.argmax(counts)
            platform_heading = midpoints[platform_bin % len(midpoints)] + 10
            if platform_heading > 360:
                platform_heading = 360
            platforms.append((fl_df['event_timestamp'].iloc[i], platform_heading))

    print(platforms)
    platform_df = pd.DataFrame(platforms, columns=["event_timestamp", "heading"])

    # Detect changes in heading and keep the last point in each consecutive block
    last_stable_points = platform_df[
        (platform_df['heading'] != platform_df['heading'].shift(-1)) |
        (platform_df['event_timestamp'].diff().shift(-1).abs() > 100)
        ]
    last_stable_points = last_stable_points.copy()
    last_stable_points.loc[:, "heading_smooth"] = last_stable_points["event_timestamp"].map(
        fl_df.set_index("event_timestamp")["heading_smooth"]
    )

    #print(last_stable_points)

    first_stable_points = platform_df[platform_df['heading'] != platform_df['heading'].shift(1)]
    first_stable_points = first_stable_points.copy()
    first_stable_points.loc[:, "heading_smooth"] = first_stable_points["event_timestamp"].map(
        fl_df.set_index("event_timestamp")["heading_smooth"]
    )

    #print(first_stable_points)

    first_stable_points_filtered = first_stable_points[['event_timestamp', 'heading']]
    last_stable_points_filtered = last_stable_points[['event_timestamp', 'heading']]
    platform_start_end = pd.concat([first_stable_points_filtered, last_stable_points_filtered], axis=0)
    platform_start_end = platform_start_end.sort_values(by='event_timestamp').reset_index(drop=True)

    print(platform_start_end)

    def calculate_angle_difference(angle1, angle2):
        """
        计算两个角度之间的差值，并处理角度的循环性（即0和360度等价）。
        返回的是有符号的角度差，考虑顺时针和逆时针方向。
        """
        diff = (angle2 - angle1) % 360  # 顺时针方向的差
        if diff > 180:
            diff -= 360  # 选择最小的角度差，并确保方向性
        return diff

    # 利用platform_start_end
    platform_start_end['next_heading'] = platform_start_end['heading'].shift(-1)
    platform_start_end['angle_diff'] = platform_start_end.apply(
        lambda row: calculate_angle_difference(row['heading'], row['next_heading']), axis=1
    )

    # 检测转角大于160度的段
    large_turns = platform_start_end[np.abs(platform_start_end['angle_diff']) >= 150]
    print("large_turns:", large_turns)

    # 输出大幅度转角期的时间戳和起始、终点角度
    large_turns_filtered = large_turns[['event_timestamp', 'heading', 'next_heading', 'angle_diff']]

    # 检测出的last_stable_points，再往后0.75个时间窗口，得到change points，用以抵消误差
    change_points = []
    for idx, row in last_stable_points.iterrows():
        timestamp = row["event_timestamp"]
        fl_window = fl_df[fl_df['event_timestamp'] > timestamp].iloc[:int(window_size * 0.92)]
        if not fl_window.empty:
            change_point_row = fl_window.iloc[-1].copy()  # Make a copy to avoid the warning
            change_point_row['heading'] = row['heading']  # Now safely add heading value
            change_points.append(change_point_row)

    change_points_df = pd.DataFrame(change_points)
    change_points_df = change_points_df.copy()
    change_points_df.loc[:, "heading_smooth"] = change_points_df["event_timestamp"].map(
        fl_df.set_index("event_timestamp")["heading_smooth"]
    )

    print(change_points_df)

    # 对 large_turns_filtered 的时间戳进行0.75窗口的时间调整
    large_turn_change_points = []
    for idx, row in large_turns_filtered.iterrows():
        timestamp = row["event_timestamp"]
        fl_window = fl_df[fl_df['event_timestamp'] > timestamp].iloc[:int(window_size * 0.92)]
        if not fl_window.empty:
            change_point_row = fl_window.iloc[-1].copy()  # Make a copy to avoid the warning
            change_point_row['heading'] = row['heading']  # 将原来的heading值复制到新的点
            large_turn_change_points.append(change_point_row)

    # 转换为DataFrame
    if large_turn_change_points:
        large_turn_change_points_df = pd.DataFrame(large_turn_change_points)
        large_turn_change_points_df = large_turn_change_points_df.copy()

        large_turn_change_points_df.loc[:, "heading_smooth"] = large_turn_change_points_df["event_timestamp"].map(
            fl_df.set_index("event_timestamp")["heading_smooth"]
        )
    else:
        large_turn_change_points_df = pd.DataFrame(columns=["event_timestamp", "heading"])

    # 输出最终结果
    print(large_turn_change_points_df)

    return fl_df, change_points_df, large_turn_change_points_df, last_stable_points, platform_start_end


# 地球半径（千米）
EARTH_RADIUS_KM = 6371.0


def calculate_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    计算两个经纬度和高度之间的三维距离，alt1 和 alt2 为米。
    :param lat1: 第一个点的纬度
    :param lon1: 第一个点的经度
    :param alt1: 第一个点的高度
    :param lat2: 第二个点的纬度
    :param lon2: 第二个点的经度
    :param alt2: 第二个点的高度
    :return: 两点之间的三维距离（单位：米）
    """
    # 地球半径（千米）
    EARTH_RADIUS_KM = 6371.0

    # 将经纬度从度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # 经纬度之间的距离（地表距离）
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    surface_distance = EARTH_RADIUS_KM * c * 1000  # 地表距离，单位为米

    # 高度差
    altitude_diff = alt2 - alt1

    # 使用勾股定理计算三维距离
    distance_3d = sqrt(surface_distance ** 2 + altitude_diff ** 2)
    return distance_3d


def detect_holding(fl_df, large_turn_change_points_df, platform_start_end, window_size=45, time_threshold=240, distance_threshold=20000):

    """
    检测holding pattern，基于大转角和平台期，并根据event_timestamp与原始数据fl_df匹配经纬度和高度信息
    :param large_turn_change_points_df: 包含大转角数据的DataFrame
    :param platform_start_end: 包含平台期数据的DataFrame
    :param fl_df: 原始数据，包含经纬度、时间戳和高度信息
    :param window_size: 时间窗口大小（默认45秒）
    :param time_threshold: 时间差阈值（秒）
    :param distance_threshold: 距离差阈值（米）
    :return: holding patterns 的 DataFrame
    """

    # 存储检测到的holding patterns
    holding_patterns = []
    print("start detect_holding")

    # 对 platform_start_end 的奇数索引行进行 0.75 时间窗口的操作
    for idx in range(1, len(platform_start_end), 2):  # 仅操作奇数索引行
        platform_start_end.loc[idx, 'event_timestamp'] += int(window_size * 0.92)  # 往后推 0.75 个时间窗口

    # 遍历所有大转角
    for i in range(len(large_turn_change_points_df) - 1):
        current_turn = large_turn_change_points_df.iloc[i]
        next_turn = large_turn_change_points_df.iloc[i + 1]
        print(current_turn)
        print(next_turn)

        # # 在 platform_start_end 中找到当前转角和下一个转角的对应平台起点
        # current_platform_idx = platform_start_end[platform_start_end['event_timestamp'] == current_turn['event_timestamp']].index[0]
        # next_platform_idx = platform_start_end[platform_start_end['event_timestamp'] == next_turn['event_timestamp']].index[0]
        # print(current_platform_idx)
        # print(next_platform_idx)

        # 在 platform_start_end 中找到当前转角和下一个转角的对应平台起点
        matching_current = platform_start_end[platform_start_end['event_timestamp'] == current_turn['event_timestamp']]
        if matching_current.empty:
            print(f"No matching platform found for current turn at {current_turn['event_timestamp']}")
            continue
        current_platform_idx = matching_current.index[0]

        matching_next = platform_start_end[platform_start_end['event_timestamp'] == next_turn['event_timestamp']]
        if matching_next.empty:
            print(f"No matching platform found for next turn at {next_turn['event_timestamp']}")
            continue
        next_platform_idx = matching_next.index[0]

        # 检查两个大转角的起点索引是否相差为2 (两弧夹一平台)
        if next_platform_idx - current_platform_idx == 2:
            current_turn_begin = platform_start_end.iloc[current_platform_idx]
            current_turn_end = platform_start_end.iloc[current_platform_idx + 1]
            next_turn_end = platform_start_end.iloc[next_platform_idx + 2]  # 获取终点
            print(current_turn_begin)
            print(next_turn_end)
            # 根据 event_timestamp 从原始数据 fl_df 中提取对应的经纬度和高度
            current_turn_begin_data = fl_df[fl_df['event_timestamp'] == current_turn_begin['event_timestamp']].iloc[0]
            current_turn_end_data = fl_df[fl_df['event_timestamp'] == current_turn_begin['event_timestamp']].iloc[0]
            next_turn_end_data = fl_df[fl_df['event_timestamp'] == next_turn_end['event_timestamp']].iloc[0]

            print(current_turn_begin_data)
            print(next_turn_end_data)

            # 计算两个大转角之间的时间差
            time_diff = next_turn['event_timestamp'] - current_turn['event_timestamp']
            print("time_diff:", time_diff)

            # 如果时间差小于给定阈值，继续检查距离差
            if time_diff <= time_threshold:
                # 计算第一个大转角的起点和第二个大转角的终点之间的三维距离
                distance_diff = calculate_distance(
                    current_turn_begin_data['latitude'], current_turn_begin_data['longitude'], current_turn_begin_data['altitude'],
                    current_turn_end_data['latitude'], current_turn_end_data['longitude'], current_turn_end_data['altitude']
                )
                print("distance_diff:", distance_diff)
                # 如果距离差也小于指定的阈值，则认为这是一个holding pattern
                if distance_diff <= distance_threshold:
                    # 获取平台期（第一个大转角与第二个大转角之间的平台）
                    platforms_in_between = platform_start_end[
                        (platform_start_end['event_timestamp'] > current_turn['event_timestamp']) &
                        (platform_start_end['event_timestamp'] < next_turn_end['event_timestamp'])
                        ]

                    # 将两个大转角和中间的平台期合并为一个holding pattern
                    holding_pattern = {
                        "holding_start": current_turn_begin['event_timestamp'],
                        "holding_end": next_turn_end['event_timestamp'],
                        "start_heading": current_turn_begin['heading'],
                        "end_heading": next_turn_end['heading'],
                        "platforms": platforms_in_between,
                        "distance_diff": distance_diff  # 记录两点之间的距离
                    }

                    # 将该holding pattern存入列表
                    holding_patterns.append(holding_pattern)

        # 将holding patterns转换为DataFrame返回
    holding_patterns_df = pd.DataFrame(holding_patterns)
    return holding_patterns_df


def detect_holding2(df, platforms, threshold_time=240, max_angle_diff=90, merge_threshold=60):

    """
    该方法基于以下设想：holding只发生在两个稳定的平台之间，并没有考虑到holding pattern的leg

    如果在45秒的windows内，飞机的方向变化小于max_angle_diff（默认 90 度），并且在经纬度上的变化量较小（表示飞机正在盘旋或保持小范围飞行），将该windows记录。
    首先需要确保每个windows的holding_start和holding_end都在同一对相邻平台点之间。
    遍历windows列表，若相邻的两个windows，它们的holding_start之间的时间差小于 merge_threshold（默认 60 秒），并且它们发生在同一对相邻平台点之间，则将它们合并为一个盘旋事件。
    依次判断能否进行合并，将所有合并完之后，记录下来为待筛选盘旋事件列表。
    遍历待筛选盘旋事件列表，判断每个事件的持续时间是否超过 threshold_time（默认 240 秒），如果是，则将该事件记录为有效的盘旋事件。
    """

    holding_windows = []
    win_size = 45  # 45秒的窗口大小
    threshold = 0.05  # 经纬度变化量阈值

    # 计算纬度和经度的变化差值来估算飞机位移
    diff_lat = df['latitude'].diff()
    diff_lon = df['longitude'].diff()

    # 计算每个时刻的航向变化，并将其标准化为 0-180 度
    diff_heading = df['derived_heading'].diff().abs()
    diff_heading[diff_heading > 180] = 360 - diff_heading[diff_heading > 180]  # 处理圆周角度的边界情况

    # 1. 遍历数据，识别基于连续小角度变化的盘旋模式
    for i in range(len(df) - win_size):
        window_heading = diff_heading.iloc[i:i + win_size]
        window_lat = diff_lat.iloc[i:i + win_size]
        window_lon = diff_lon.iloc[i:i + win_size]

        # 通过判断航向变化和地理移动量来检测盘旋模式
        if (window_heading.mean() < max_angle_diff) and (window_lat.abs().sum() < threshold) and (
                window_lon.abs().sum() < threshold):
            start_time = df['event_timestamp'].iloc[i]
            end_time = df['event_timestamp'].iloc[i + win_size - 1]

            # 2. 确保盘旋事件发生在同一对相邻平台点之间
            for j in range(len(platforms) - 1):
                if platforms['event_timestamp'].iloc[j] <= start_time <= platforms['event_timestamp'].iloc[j + 1] and \
                        platforms['event_timestamp'].iloc[j] <= end_time <= platforms['event_timestamp'].iloc[j + 1]:
                    if j % 2 == 1:  # 只允许奇数的 platform_idx
                        holding_windows.append((start_time, end_time, j))  # 记录平台点的索引 j
                    break

    if not holding_windows:
        return pd.DataFrame(columns=["holding_start", "holding_end", "platform_idx"])

    holding_df = pd.DataFrame(holding_windows, columns=["holding_start", "holding_end", "platform_idx"])
    holding_df = holding_df.sort_values('holding_start').reset_index(drop=True)

    # 3. 合并相邻的盘旋事件
    merged_holding_events = []
    current_start = holding_df.loc[0, 'holding_start']
    current_end = holding_df.loc[0, 'holding_end']
    current_platform_idx = holding_df.loc[0, 'platform_idx']

    for i in range(1, len(holding_df)):
        time_diff = holding_df.loc[i, 'holding_start'] - current_end
        next_platform_idx = holding_df.loc[i, 'platform_idx']

        # 如果相邻事件满足合并条件：时间差小于 merge_threshold 且在同一平台点对之间
        if time_diff <= merge_threshold and next_platform_idx == current_platform_idx:
            current_end = holding_df.loc[i, 'holding_end']
        else:
            # 先将当前事件记录下来，然后开始新的事件
            merged_holding_events.append((current_start, current_end, current_platform_idx))
            current_start = holding_df.loc[i, 'holding_start']
            current_end = holding_df.loc[i, 'holding_end']
            current_platform_idx = next_platform_idx

    # 添加最后一个合并后的事件
    merged_holding_events.append((current_start, current_end, current_platform_idx))
    merged_holding_df = pd.DataFrame(merged_holding_events, columns=["holding_start", "holding_end", "platform_idx"])

    # 4. 筛选有效的盘旋事件（持续时间大于 threshold_time）
    valid_holding_events = []
    for index, row in merged_holding_df.iterrows():
        if row['holding_end'] - row['holding_start'] >= threshold_time:
            valid_holding_events.append((row['holding_start'], row['holding_end'], row['platform_idx']))

    # 返回有效的盘旋事件
    valid_holding_df = pd.DataFrame(valid_holding_events, columns=["holding_start", "holding_end", "platform_idx"])

    return valid_holding_df


track_data = pd.read_csv('Track_interpolation.csv')

# "CPA759", "AAR753", "TGW843", "CTV510", "PAL507" 这些使用新方法
# "CAL753", "SIA935" cancel hold
# "MMA231"、"TGW265"leg段只有10秒; "FIN131"leg段有30秒，但波动剧烈+-7; 这三个使用老方法, 优先信任新方法

# selected_callsign = track_data['callsign'].unique()
selected_callsign = ["CPA759", "TGW843","AAR753"]
all_events = pd.DataFrame(columns=["time", "callsign", "event", "parameter", "x", "y"])

for callsign in selected_callsign:
    df_selected = track_data[track_data['callsign'] == callsign]

    # # 适应waypoint
    # df_selected = df_selected[df_selected['latitude'] <= 4.5]
    #
    # # Process data for the selected callsign
    #     # fl_df, change_points_df, large_turn_change_points_df, lower_bounds, upper_bounds, platform_start_end = heading_changes(df_selected, win=11)
    #
    # # Plot heading vs time for the callsign
    # plt.plot(df_selected['event_timestamp'], df_selected['derived_heading'], label=f'{callsign} Original', alpha=0.5)
    # plt.plot(fl_df['event_timestamp'], fl_df['heading_smooth'], label=f'{callsign} Smoothed')
    #
    # # Mark change points on the plot using real heading values
    # if not change_points_df.empty:
    #     plt.scatter(change_points_df['event_timestamp'], change_points_df['heading_smooth'],
    #                 color='red', label=f'{callsign} Change Points')

    # holding_patterns_df1 = detect_holding(fl_df, large_turn_change_points_df, platform_start_end)
    # print(holding_patterns_df1)
    #
    # # Highlight holding patterns in yellow
    # if not holding_patterns_df1.empty:
    #     for idx, row in holding_patterns_df1.iterrows():
    #         plt.axvspan(row['holding_start'], row['holding_end'], color='yellow', alpha=0.3, label='Holding Pattern')
    # else:
    #     holding_patterns_df2 = detect_holding2(df_selected, platform_start_end)
    #     print(holding_patterns_df2)
    #
    #     # Highlight holding patterns in yellow
    #     if not holding_patterns_df2.empty:
    #         for idx, row in holding_patterns_df2.iterrows():
    #             plt.axvspan(row['holding_start'], row['holding_end'], color='yellow', alpha=0.3, label='Holding Pattern')

    heading_df, heading_changes_df, large_turns_df, last_stable_points, platform_start_end = heading_changes(df_selected, win=11)

    # Detect holding patterns：
    # 1、detect_holding方法：开始遍历，对于每一个大转角，判断是否为大转角+平台+大转角的连续形式
    # （两个大转角之间的时间差小于200，且第一个大转角的起点和第二个大转角的终点之间的距离小于XXX），如果是，则合并为同一个holding pattern
    # 2、detect_holding2方法：如果有两个大转角中间无平台怎么办？使用detect_holding2方法（基于平台之间才有holding的思想）
    # 3、两个方法同时进行，提高稳健性（holding leg较小的情况）

    holding_patterns_df = detect_holding(heading_df, large_turns_df, platform_start_end)
    if holding_patterns_df.empty:
        holding_patterns_df = detect_holding2(heading_df, platform_start_end)

    print(holding_patterns_df)

    # 创建图形
    plt.figure(figsize=(15, 8))

    # 绘制原始 Heading 和 Smoothed Heading
    plt.plot(heading_df['event_timestamp'], heading_df['heading_smooth'], label=f'{callsign} Heading',
             color='orange')

    # 标记 Change Points
    if not heading_changes_df.empty:

        window_size = 60
        parameters = []

        for idx, row in heading_changes_df.iterrows():
            # 获取当前变化点的时间
            current_time = row['event_timestamp']
            next_platform = last_stable_points[last_stable_points['event_timestamp'] > current_time - 0.92 * window_size].head(1)

            # 如果找到匹配的平台，使用其 altitude 值作为 parameter；否则为 NaN
            if not next_platform.empty:
                parameter_value = next_platform.iloc[0]['heading']  # 取第一个满足条件的平台
            else:
                parameter_value = None

            parameters.append(parameter_value)

        # 将 Change Points 存储到事件表
        heading_changes_df["event"] = "heading change"
        heading_changes_df["parameter"] = parameters
        heading_changes_df["parameter"] = heading_changes_df["parameter"].fillna(0)
        heading_changes_df["x"] = heading_changes_df["event_timestamp"].map(
            heading_df.set_index("event_timestamp")["longitude"])
        heading_changes_df["y"] = heading_changes_df["event_timestamp"].map(
            heading_df.set_index("event_timestamp")["latitude"])
        heading_changes_df["callsign"] = callsign
        heading_changes_df = heading_changes_df.rename(columns={"event_timestamp": "time"})

        heading_changes_df = heading_changes_df[["time", "callsign", "event", "parameter", "x", "y"]]
        all_events = pd.concat([all_events, heading_changes_df], ignore_index=True)

        plt.scatter(heading_changes_df['time'], heading_changes_df['parameter'],
                    color='red', label="Change Points", zorder=3)
        for _, row in heading_changes_df.iterrows():
            plt.text(row['time'], row['parameter'] + 2,
                     f"{row['parameter']}", color='red', ha='center', fontsize=9)

    # 标记 Holding Patterns
    if not holding_patterns_df.empty:
        for _, row in holding_patterns_df.iterrows():
            plt.axvspan(row['holding_start'], row['holding_end'], color='yellow', alpha=0.3,
                        label="Holding Pattern")
            plt.text((row['holding_start'] + row['holding_end']) / 2,
                     heading_df['heading_smooth'].mean(),
                     "Holding", color='black', ha='center', fontsize=10, rotation=90)

        # 将 Holding Patterns 存储到事件表
        holding_patterns_events = pd.DataFrame([{
            "time": row['holding_start'],
            "callsign": callsign,
            "event": "holding start",
            "parameter": "",
            "x": None,
            "y": None
        } for _, row in holding_patterns_df.iterrows()])

        holding_patterns_events_end = pd.DataFrame([{
            "time": row['holding_end'],
            "callsign": callsign,
            "event": "holding end",
            "parameter": "",
            "x": None,
            "y": None
        } for _, row in holding_patterns_df.iterrows()])

        all_events = pd.concat([all_events, holding_patterns_events, holding_patterns_events_end],
                               ignore_index=True)

    plt.xlabel('Time')
    plt.ylabel('Heading')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'Heading with Detected Maneuvering Points and Holding Patterns for Callsign: {callsign}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# # 保存所有事件
# all_events.to_csv("999all_callsigns_heading_maneuvering_points2.csv", index=False)
#
# # 读取 CSV 文件
# input_file = "999all_callsigns_heading_maneuvering_points2.csv"  # 替换为您的文件名
# output_file = "999all_callsigns_heading_maneuvering_points.csv"  # 输出文件名（这才是需要的）
# # 加载数据
# data = pd.read_csv(input_file)
#
# # 创建存储更新后的数据列表
# updated_data = []
#
# # 遍历每个 callsign
# for callsign in data['callsign'].unique():
#     # 筛选出当前 callsign 的数据
#     callsign_data = data[data['callsign'] == callsign]
#
#     # 将 holding 事件单独保留
#     holding_events = callsign_data[callsign_data['event'].str.contains('holding', case=False, na=False)]
#
#     # 处理 heading change 事件
#     heading_changes = callsign_data[callsign_data['event'] == 'heading change']
#
#     # 按时间排序
#     heading_changes = heading_changes.sort_values(by="time").reset_index(drop=True)
#
#     # 更新 heading change 的 parameter
#     for i in range(len(heading_changes) - 1):
#         heading_changes.loc[i, 'parameter'] = heading_changes.loc[i + 1, 'parameter']
#
#     # 最后一个 heading change 的 parameter 设置为 0
#     if not heading_changes.empty:
#         heading_changes.loc[len(heading_changes) - 1, 'parameter'] = 0
#
#     # 合并 heading change 和 holding 事件
#     updated_callsign_data = pd.concat([heading_changes, holding_events], ignore_index=True)
#     updated_callsign_data = updated_callsign_data.sort_values(by="time").reset_index(drop=True)
#
#     # 添加到总数据列表
#     updated_data.append(updated_callsign_data)
#
# # 合并所有更新后的数据
# updated_data = pd.concat(updated_data, ignore_index=True)
#
# # 保存到新的 CSV 文件
# updated_data.to_csv(output_file, index=False)























# # 可视化route, 并且判断在哪一条plan route上, 没有整好
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from geopy.distance import geodesic
#
#
# # 加载 sid_star.csv 的航路点数据
# waypoints_path = 'sid_star.csv'
# waypoints_df = pd.read_csv(waypoints_path)
#
# # 加载 STAR 路线文件夹中所有 STAR 路线文件
# star_folder_path = "Track_processing_plotting/tools/STAR"
# star_files = [os.path.join(star_folder_path, f) for f in os.listdir(star_folder_path) if f.startswith("STAR") and f.endswith(".csv")]
# sid_folder_path = "Track_processing_plotting/tools/SID"
# sid_files = [os.path.join(sid_folder_path, f) for f in os.listdir(sid_folder_path) if f.startswith("SID") and f.endswith(".csv")]
# files = star_files + sid_files
#
# # 将 sid_star.csv 中的 DMS 格式经纬度转换为小数格式
# def dms_to_decimal(dms_str):
#     if 'N' in dms_str or 'S' in dms_str:
#         d, m, s = int(dms_str[:2]), int(dms_str[2:4]), int(dms_str[4:6])
#         hemisphere = dms_str[6]
#     else:
#         d, m, s = int(dms_str[:3]), int(dms_str[3:5]), int(dms_str[5:7])
#         hemisphere = dms_str[7]
#
#     decimal = d + m / 60 + s / 3600
#     if hemisphere in 'SW':
#         decimal *= -1
#     return decimal
#
#
# # 转换经纬度并去除航路点名称的空格
# waypoints_df['Name '] = waypoints_df['Name '].str.strip()
# waypoints_df['latitude'] = waypoints_df['Latitude '].apply(dms_to_decimal)
# waypoints_df['longitude'] = waypoints_df['Longitude '].apply(dms_to_decimal)
#
#
# # 加载每条 STAR 路线数据并存储在字典中
# def load_star_route(file_path):
#     # 加载数据并移除多余空格
#     df = pd.read_csv(file_path, header=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)
#     return df
#
#
# star_routes = {}
# for file in files:
#     route_name = os.path.basename(file).replace(".csv", "")
#     star_routes[route_name] = load_star_route(file)
#
#
# # 定义判断点是否在 STAR 路线附近的函数
# def is_point_near_star_route(point, star_waypoints, threshold=1.0):
#     for _, waypoint in star_waypoints.iterrows():
#         distance = geodesic(point, (waypoint['latitude'], waypoint['longitude'])).nautical
#         if distance <= threshold:
#             return True
#     return False
#
#
# # 初始化匹配路线路径
# matching_route = None
# matching_score = 0
#
# # 检查每个 STAR 路线的匹配程度
# for route_name, route_df in star_routes.items():
#     route_df = route_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
#     waypoint_names = route_df.iloc[:, 1].tolist()
#     route_waypoints = waypoints_df.set_index('Name ').loc[waypoint_names].reset_index()
#
#     print(route_waypoints)
#
#     # 计算角度变化点中在 STAR 路线附近的点数
#     matching_points = change_points_df.apply(lambda row: is_point_near_star_route((row['latitude'], row['longitude']), route_waypoints), axis=1)
#     score = matching_points.sum()
#     print(score)
#     print(666)
#     # 更新匹配程度最高的 STAR 路线
#     if score >= matching_score:
#         matching_score = score
#         matching_route = route_name
#
# # 可视化所有航路点、飞行轨迹和角度变化点
# fig, ax = plt.subplots(figsize=(10, 10))
#
# # 绘制 sid_star.csv 中的所有航路点
# ax.scatter(waypoints_df['longitude'], waypoints_df['latitude'], color='gray', label='All Waypoints (sid_star.csv)', s=10, alpha=0.5)
#
# # 绘制匹配的 STAR 路线
# if matching_route:
#     route_df = star_routes[matching_route]
#     waypoint_names = route_df.iloc[:, 1].tolist()
#     route_waypoints = waypoints_df.set_index('Name ').loc[waypoint_names].reset_index()
#     ax.plot(route_waypoints['longitude'], route_waypoints['latitude'], label=f'{matching_route} Route', linestyle='--')
#
# # 绘制飞行轨迹和角度变化点
# ax.plot(df_selected['longitude'], df_selected['latitude'], color='blue', label='Flight Track', alpha=0.5)
# in_star_points = change_points_df[change_points_df.apply(lambda row: is_point_near_star_route((row['latitude'], row['longitude']), route_waypoints), axis=1)]
# out_of_star_points = change_points_df[~change_points_df.apply(lambda row: is_point_near_star_route((row['latitude'], row['longitude']), route_waypoints), axis=1)]
#
# # 绘制变化点
# ax.scatter(in_star_points['longitude'], in_star_points['latitude'], color='red', label='In STAR Route', marker='o')
# ax.scatter(out_of_star_points['longitude'], out_of_star_points['latitude'], color='orange', label='Out of STAR Route', marker='x')
#
# # Plotting the STAR route and labeling each waypoint
# for _, waypoint in route_waypoints.iterrows():
#     ax.text(waypoint['longitude'], waypoint['latitude'], waypoint['Name '], fontsize=8, ha='right', color='green')
#
# # 设置图例和标题
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.legend(loc='upper right')
# ax.set_title(f'Flight Track and Best Matching STAR Route\nBest Match: {matching_route}')
# ax.set_aspect('equal', 'box')
#
# plt.show()
#
# # 输出最佳匹配的 STAR 路线结果
# if matching_route:
#     print(f"The flight track best matches the STAR route: {matching_route}")
# else:
#     print("No matching STAR route was found for the flight track.")
