import pandas as pd
import matplotlib.pyplot as plt

# 提取三个参数基础的语音指令
# 加载轨迹语音数据
df1 = pd.read_csv('matched_voice_to_track.csv')

# name = 'SIA851'
# df = df1[df1['callsign'] == name]

all_callsigns = df1['callsign'].unique()

# 创建用于存储最终结果的DataFrame
events = pd.DataFrame(columns=["time", "callsign", "event", "parameter", "x", "y", "condition"])

for name in all_callsigns:
    # 筛选出当前 callsign 的数据
    df = df1[df1['callsign'] == name]

    # 过滤出包含 CAS Command 的行，且 maintain 列不等于 1
    cas_commands = df[df['next_speed'].notna() & (df['maintain'] != 1)]

    # 添加 CAS Command 事件
    cas_events = pd.DataFrame([{
        "time": row['event_timestamp'],
        "callsign": row['callsign'],
        "event": "velocity change",
        "parameter": row['next_speed'],
        "x": row['longitude'],
        "y": row['latitude'],
        "condition": row['condition'] if pd.notna(row['condition']) and 'after' in str(row['condition']).lower() else ""
    } for idx, row in cas_commands.iterrows()])

    filtered_cas_events_df = cas_events

    if 'time' in cas_events.columns:
        # 按时间排序，方便后续比较
        cas_events = cas_events.sort_values(by="time").reset_index(drop=True)
        # 过滤掉内容相同且时间间隔小于 60 秒的前一个 command
        filtered_cas_events = []
        for i in range(len(cas_events)):
            # 检查当前行和前一行
            if i > 0:
                current_row = cas_events.iloc[i]
                previous_row = cas_events.iloc[i - 1]

                # 检查 command 内容和时间差
                if (
                        current_row["event"] == previous_row["event"] and
                        #(current_row["parameter"] - previous_row["parameter"]) <= 10 and
                        (current_row["time"] - previous_row["time"]) < 60
                ):
                    # 跳过前一个 command
                    continue

            # 保留当前行
            filtered_cas_events.append(cas_events.iloc[i])

        # 将过滤后的数据转为 DataFrame
        filtered_cas_events_df = pd.DataFrame(filtered_cas_events)

    # 将当前 callsign 的 CAS Command 事件添加到最终的 events 表中
    events = pd.concat([events, filtered_cas_events_df], ignore_index=True)

# 保存为 CSV 文件
events.to_csv('all_callsigns_CAS_commands.csv', index=False)


# 创建用于存储最终结果的 DataFrame
events = pd.DataFrame(columns=["time", "callsign", "event", "parameter", "x", "y", "condition"])

for name in all_callsigns:
    # 筛选出当前 callsign 的数据
    df = df1[df1['callsign'] == name]

    # 过滤出包含 Altitude Command 的行，且 maintain 列不等于 1
    altitude_commands = df[df['next_alt'].notna() & (df['maintain'] != 1)]

    # 添加 Altitude Command 事件
    altitude_events = pd.DataFrame([{
        "time": row['event_timestamp'],
        "callsign": row['callsign'],
        "event": "flight level change",
        "parameter": row['next_alt'],
        "x": row['longitude'],
        "y": row['latitude'],
        "condition": row['condition'] if pd.notna(row['condition']) and 'after' in str(row['condition']).lower() else ""
    } for idx, row in altitude_commands.iterrows()])

    filtered_altitude_events_df = altitude_events

    # 如果 'time' 列存在，按时间排序，并移除相似的连续命令
    if 'time' in altitude_events.columns:
        # 按时间排序，方便后续比较
        altitude_events = altitude_events.sort_values(by="time").reset_index(drop=True)

        # 过滤掉内容相同且时间间隔小于 60 秒的前一个 command
        filtered_altitude_events = []
        for i in range(len(altitude_events)):
            # 检查当前行和前一行
            if i > 0:
                current_row = altitude_events.iloc[i]
                previous_row = altitude_events.iloc[i - 1]

                # 检查 command 内容和时间差
                if (
                    current_row["event"] == previous_row["event"] and
                    (current_row["time"] - previous_row["time"]) < 60
                ):
                    # 跳过前一个 command
                    continue

            # 保留当前行
            filtered_altitude_events.append(altitude_events.iloc[i])

        # 将过滤后的数据转为 DataFrame
        filtered_altitude_events_df = pd.DataFrame(filtered_altitude_events)

    # 将当前 callsign 的 Altitude Command 事件添加到最终的 events 表中
    events = pd.concat([events, filtered_altitude_events_df], ignore_index=True)

# 保存最终的 Altitude Command 事件为 CSV 文件
events.to_csv('all_callsigns_altitude_commands.csv', index=False)

# 创建用于存储最终结果的 DataFrame
events = pd.DataFrame(columns=["time", "callsign", "event", "parameter", "x", "y", "condition"])

for name in all_callsigns:
    # 筛选出当前 callsign 的数据
    df = df1[df1['callsign'] == name]

    # 过滤出包含 Heading Command 的行
    heading_commands = df[df['next_heading'].notna()]

    # 添加 Heading Command 事件
    heading_events = pd.DataFrame([{
        "time": row['event_timestamp'],
        "callsign": row['callsign'],
        "event": "head change",
        "parameter": row['next_heading'],
        "x": row['longitude'],
        "y": row['latitude'],
        "condition": row['condition'] if pd.notna(row['condition']) and 'after' in str(row['condition']).lower() else ""
    } for idx, row in heading_commands.iterrows()])

    # # 如果 'time' 列存在，按时间排序，并移除相似的连续命令
    # if 'time' in heading_events.columns:
    #     # 按时间排序，方便后续比较
    #     heading_events = heading_events.sort_values(by="time").reset_index(drop=True)
    #
    #     # 过滤掉内容相同且时间间隔小于 60 秒的前一个 command
    #     filtered_heading_events = []
    #     for i in range(len(heading_events)):
    #         # 检查当前行和前一行
    #         if i > 0:
    #             current_row = heading_events.iloc[i]
    #             previous_row = heading_events.iloc[i - 1]
    #
    #             # 检查 command 内容和时间差
    #             if (
    #                 current_row["event"] == previous_row["event"] and
    #                 current_row["parameter"] == previous_row["parameter"] and
    #                 (current_row["time"] - previous_row["time"]) < 1
    #             ):
    #                 # 跳过前一个 command
    #                 continue
    #
    #         # 保留当前行
    #         filtered_heading_events.append(heading_events.iloc[i])
    #
    #     # 将过滤后的数据转为 DataFrame
    #     filtered_heading_events_df = pd.DataFrame(filtered_heading_events)
    #
    # # 将当前 callsign 的 Heading Command 事件添加到最终的 events 表中
    # events = pd.concat([events, filtered_heading_events_df], ignore_index=True)

# 匹配 "hold" 事件
    hold_conditions = df[df['condition'].notna() & df['condition'].str.contains('hold', case=False, na=False)]
    holding_events = pd.DataFrame([{
        "time": row['event_timestamp'],
        "callsign": row['callsign'],
        "event": "holding",
        "parameter": "",
        "x": row['longitude'],
        "y": row['latitude'],
        "condition": row['condition']
    } for idx, row in hold_conditions.iterrows()])

    # 匹配 "hold cancel" 事件
    hold_cancel_conditions = df[df['cancel'].notna() & df['cancel'].str.contains('hold', case=False, na=False)]
    holding_cancel_events = pd.DataFrame([{
        "time": row['event_timestamp'],
        "callsign": row['callsign'],
        "event": "holding cancel",
        "parameter": "",
        "x": row['longitude'],
        "y": row['latitude'],
        "condition": ""
    } for idx, row in hold_cancel_conditions.iterrows()])

    # 合并所有事件
    events = pd.concat([events, heading_events, holding_events, holding_cancel_events], ignore_index=True)

# 保存最终的 Heading Command 事件为 CSV 文件
events.to_csv('all_callsigns_heading_commands.csv', index=False)
