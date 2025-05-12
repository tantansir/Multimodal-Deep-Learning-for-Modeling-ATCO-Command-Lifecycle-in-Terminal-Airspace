import pandas as pd

# 利用3个参数的真实的语音command，将其匹配到提取得到的maneuvering point上

# 读取数据
commands_df = pd.read_csv('all_callsigns_CAS_commands.csv')
maneuvering_df = pd.read_csv('all_callsigns_CAS_maneuvering_points.csv')

# 创建用于存储最终匹配结果的 DataFrame
matches = pd.DataFrame(
    columns=["time", "callsign", "event", "parameter", "x", "y", "maneuvering_time", "maneuvering_parameter", "time_offset"])

# 创建一个副本，用于标记已经匹配的 maneuvering 点，确保一对一匹配
available_maneuvering_df = maneuvering_df.copy()

# 分离包含 "after" 的 commands 和其他 commands
after_commands = commands_df[commands_df['condition'].str.contains('after', case=False, na=False)]
normal_commands = commands_df[~commands_df.index.isin(after_commands.index)]

# 处理包含 "after" 的 commands
for idx, command_row in after_commands.iterrows():
    command_time = command_row['time']
    command_callsign = command_row['callsign']
    command_parameter = command_row['parameter']

    # 找到时间在 command 之后的所有 maneuvering 点
    possible_matches = available_maneuvering_df[
        (available_maneuvering_df['callsign'] == command_callsign) &
        (available_maneuvering_df['time'] > command_time) &
        (available_maneuvering_df['time'] <= command_time + 600) &
        (abs(available_maneuvering_df['parameter'] - command_parameter) <= 10)
    ]

    # 如果有符合条件的 maneuvering，找到时间最接近的那个
    if not possible_matches.empty:
        best_match_index = (possible_matches['time'] - command_time).idxmin()
        best_match = possible_matches.loc[best_match_index]
        time_offset = best_match['time'] - command_time

        # 将匹配结果添加到 matches 中
        match_entry = pd.DataFrame([{
            "time": command_row['time'],
            "callsign": command_row['callsign'],
            "event": command_row['event'],
            "parameter": command_row['parameter'],
            "x": command_row['x'],
            "y": command_row['y'],
            "condition": command_row.get('condition', ''),
            "maneuvering_time": best_match['time'],
            "maneuvering_parameter": best_match['parameter'],
            "time_offset": time_offset
        }])
        matches = pd.concat([matches, match_entry], ignore_index=True)
        # 从 available_maneuvering_df 中移除已匹配的 maneuvering 点
        available_maneuvering_df = available_maneuvering_df.drop(best_match_index)

# 使用优先队列机制处理其他 commands
unmatched_commands = normal_commands.copy()

while not unmatched_commands.empty:
    # 构建匹配候选列表
    match_candidates = []

    for idx, command_row in unmatched_commands.iterrows():
        command_time = command_row['time']
        command_parameter = command_row['parameter']
        command_callsign = command_row['callsign']

        # 找到在时间窗口内的 maneuvering 点
        possible_matches = available_maneuvering_df[
            (available_maneuvering_df['callsign'] == command_callsign) &
            (available_maneuvering_df['time'] >= command_time - 60) &
            (available_maneuvering_df['time'] <= command_time + 180)
            # (abs(available_maneuvering_df['parameter'] - command_parameter) <= 20)
        ]

        # 如果存在匹配的 maneuvering 点
        if not possible_matches.empty:
            # 创建 possible_matches 副本，避免赋值警告
            possible_matches = possible_matches.copy()
            # 计算时间差
            possible_matches['time_diff'] = abs(possible_matches['time'] - command_time)
            # 找到时间差最小的 maneuvering
            best_match = possible_matches.loc[possible_matches['time_diff'].idxmin()]
            match_candidates.append((idx, best_match['time_diff'], command_row, best_match))

    # 如果没有候选匹配，跳出循环
    if not match_candidates:
        break

    # 按时间差排序，优先匹配最近的点
    match_candidates.sort(key=lambda x: x[1])

    # 处理当前匹配的第一个候选
    best_idx, _, best_command, best_maneuvering = match_candidates[0]

    # 记录匹配结果
    time_offset = best_maneuvering['time'] - best_command['time']
    match_entry = pd.DataFrame([{
        "time": best_command['time'],
        "callsign": best_command['callsign'],
        "event": best_command['event'],
        "parameter": best_command['parameter'],
        "x": best_command['x'],
        "y": best_command['y'],
        "condition": best_command.get('condition', ''),
        "maneuvering_time": best_maneuvering['time'],
        "maneuvering_parameter": best_maneuvering['parameter'],
        "time_offset": time_offset
    }])
    matches = pd.concat([matches, match_entry], ignore_index=True)

    # 移除已匹配的 command 和 maneuvering 点
    unmatched_commands = unmatched_commands.drop(best_idx)
    available_maneuvering_df = available_maneuvering_df.drop(best_maneuvering.name)

# 保存匹配结果
matches.to_csv('CAS_audio_maneuvering_match.csv', index=False)


# 读取数据
commands_df = pd.read_csv('all_callsigns_altitude_commands.csv')
maneuvering_df = pd.read_csv('all_callsigns_altitude_maneuvering_points.csv')

# 创建用于存储最终匹配结果的 DataFrame
matches = pd.DataFrame(
    columns=["time", "callsign", "event", "parameter", "x", "y", "maneuvering_time", "maneuvering_parameter", "time_offset"])

# 创建一个副本，用于标记已经匹配的 maneuvering 点，确保一对一匹配
available_maneuvering_df = maneuvering_df.copy()

# 分离包含 "after" 的 commands 和其他 commands
after_commands = commands_df[commands_df['condition'].str.contains('after', case=False, na=False)]
normal_commands = commands_df[~commands_df.index.isin(after_commands.index)]

# 处理包含 "after" 的 commands
for idx, command_row in after_commands.iterrows():
    command_time = command_row['time']
    command_callsign = command_row['callsign']

    # 找到时间在 command 之后的所有 maneuvering 点
    possible_matches = available_maneuvering_df[
        (available_maneuvering_df['callsign'] == command_callsign) &
        (available_maneuvering_df['time'] > command_time) &
        (available_maneuvering_df['time'] <= command_time + 600) &
        (abs(available_maneuvering_df['parameter'] - command_parameter) <= 1000)
    ]

    # 如果有符合条件的 maneuvering，找到时间最接近的那个
    if not possible_matches.empty:
        best_match_index = (possible_matches['time'] - command_time).idxmin()
        best_match = possible_matches.loc[best_match_index]
        time_offset = best_match['time'] - command_time

        # 将匹配结果添加到 matches 中
        match_entry = pd.DataFrame([{
            "time": command_row['time'],
            "callsign": command_row['callsign'],
            "event": command_row['event'],
            "parameter": command_row['parameter'],
            "x": command_row['x'],
            "y": command_row['y'],
            "condition": command_row.get('condition', ''),
            "maneuvering_time": best_match['time'],
            "maneuvering_parameter": best_match['parameter'],
            "time_offset": time_offset
        }])
        matches = pd.concat([matches, match_entry], ignore_index=True)
        # 从 available_maneuvering_df 中移除已匹配的 maneuvering 点
        available_maneuvering_df = available_maneuvering_df.drop(best_match_index)

# 使用优先队列机制处理其他 commands
unmatched_commands = normal_commands.copy()

while not unmatched_commands.empty:
    # 构建匹配候选列表
    match_candidates = []

    for idx, command_row in unmatched_commands.iterrows():
        command_time = command_row['time']
        command_parameter = command_row['parameter']
        command_callsign = command_row['callsign']

        # 找到在时间窗口内的 maneuvering 点
        possible_matches = available_maneuvering_df[
            (available_maneuvering_df['callsign'] == command_callsign) &
            (available_maneuvering_df['time'] >= command_time - 60) &
            (available_maneuvering_df['time'] <= command_time + 180)
            # (abs(available_maneuvering_df['parameter'] - command_parameter) <= 2000)
        ]

        # 如果存在匹配的 maneuvering 点
        if not possible_matches.empty:
            # 创建 possible_matches 副本，避免赋值警告
            possible_matches = possible_matches.copy()
            # 计算时间差
            possible_matches['time_diff'] = abs(possible_matches['time'] - command_time)
            # 找到时间差最小的 maneuvering
            best_match = possible_matches.loc[possible_matches['time_diff'].idxmin()]
            match_candidates.append((idx, best_match['time_diff'], command_row, best_match))

    # 如果没有候选匹配，跳出循环
    if not match_candidates:
        break

    # 按时间差排序，优先匹配最近的点
    match_candidates.sort(key=lambda x: x[1])

    # 处理当前匹配的第一个候选
    best_idx, _, best_command, best_maneuvering = match_candidates[0]

    # 记录匹配结果
    time_offset = best_maneuvering['time'] - best_command['time']
    match_entry = pd.DataFrame([{
        "time": best_command['time'],
        "callsign": best_command['callsign'],
        "event": best_command['event'],
        "parameter": best_command['parameter'],
        "x": best_command['x'],
        "y": best_command['y'],
        "condition": best_command.get('condition', ''),
        "maneuvering_time": best_maneuvering['time'],
        "maneuvering_parameter": best_maneuvering['parameter'],
        "time_offset": time_offset
    }])
    matches = pd.concat([matches, match_entry], ignore_index=True)

    # 移除已匹配的 command 和 maneuvering 点
    unmatched_commands = unmatched_commands.drop(best_idx)
    available_maneuvering_df = available_maneuvering_df.drop(best_maneuvering.name)

# 保存匹配结果
matches.to_csv('altitude_audio_maneuvering_match.csv', index=False)


commands_df = pd.read_csv('all_callsigns_heading_commands.csv')
maneuvering_df = pd.read_csv('all_callsigns_heading_maneuvering_points.csv')

# 初始化最终结果 DataFrame
matches = pd.DataFrame(
    columns=["time", "callsign", "event", "parameter", "x", "y", "condition", "maneuvering_time", "maneuvering_parameter", "time_offset"]
)

# 创建副本以标记未匹配的 maneuvering 点
available_maneuvering_df = maneuvering_df.copy()

# 分离包含 "after" 的 commands 和其他 commands
after_commands = commands_df[commands_df['condition'].str.contains('after', case=False, na=False)]
normal_commands = commands_df[~commands_df.index.isin(after_commands.index)]

# 处理包含 "after" 的 commands
for idx, command_row in after_commands.iterrows():
    command_time = command_row['time']
    command_callsign = command_row['callsign']
    command_event = command_row['event']

    # 找到时间在 command 之后的匹配 maneuvering
    if command_event == "head change":
        possible_matches = available_maneuvering_df[
            (available_maneuvering_df['callsign'] == command_callsign) &
            (available_maneuvering_df['event'] == "heading change") &
            (available_maneuvering_df['time'] > command_time)
        ]
    elif command_event == "holding":
        possible_matches = available_maneuvering_df[
            (available_maneuvering_df['callsign'] == command_callsign) &
            (available_maneuvering_df['event'] == "holding start") &
            (available_maneuvering_df['time'] > command_time)
        ]
    else:
        continue

    # 找到时间最接近的匹配点
    if not possible_matches.empty:
        best_match_index = (possible_matches['time'] - command_time).idxmin()
        best_match = possible_matches.loc[best_match_index]
        time_offset = best_match['time'] - command_time

        # 添加匹配结果
        match_entry = pd.DataFrame([{
            "time": command_row['time'],
            "callsign": command_row['callsign'],
            "event": command_row['event'],
            "parameter": command_row['parameter'],
            "x": command_row['x'],
            "y": command_row['y'],
            "condition": command_row.get('condition', ''),
            "maneuvering_time": best_match['time'],
            "maneuvering_parameter": best_match['parameter'],
            "time_offset": time_offset
        }])
        matches = pd.concat([matches, match_entry], ignore_index=True)

        # 移除已匹配的 maneuvering 点
        available_maneuvering_df = available_maneuvering_df.drop(best_match_index)

# 使用优先队列处理其他 commands
unmatched_commands = normal_commands.copy()

while not unmatched_commands.empty:
    match_candidates = []

    for idx, command_row in unmatched_commands.iterrows():
        command_time = command_row['time']
        command_callsign = command_row['callsign']
        command_event = command_row['event']

        # 根据 event 类型筛选可匹配的 maneuvering 点
        if command_event == "head change":
            possible_matches = available_maneuvering_df[
                (available_maneuvering_df['callsign'] == command_callsign) &
                (available_maneuvering_df['event'] == "heading change") &
                (available_maneuvering_df['time'] >= command_time - 60) &
                (available_maneuvering_df['time'] <= command_time + 180)
            ]
        elif command_event == "holding":
            possible_matches = available_maneuvering_df[
                (available_maneuvering_df['callsign'] == command_callsign) &
                (available_maneuvering_df['event'] == "holding start") &
                (available_maneuvering_df['time'] >= command_time - 240)
                # (available_maneuvering_df['time'] <= command_time + 720)
            ]
        else:
            continue

        # 如果存在匹配的 maneuvering 点
        if not possible_matches.empty:
            possible_matches = possible_matches.copy()  # 避免 SettingWithCopyWarning
            possible_matches['time_diff'] = abs(possible_matches['time'] - command_time)
            best_match = possible_matches.loc[possible_matches['time_diff'].idxmin()]
            match_candidates.append((idx, best_match['time_diff'], command_row, best_match))

    # 如果没有候选匹配，跳出循环
    if not match_candidates:
        break

    # 按时间差排序，优先处理最近的点
    match_candidates.sort(key=lambda x: x[1])
    best_idx, _, best_command, best_maneuvering = match_candidates[0]

    # 添加匹配结果
    time_offset = best_maneuvering['time'] - best_command['time']
    match_entry = pd.DataFrame([{
        "time": best_command['time'],
        "callsign": best_command['callsign'],
        "event": best_command['event'],
        "parameter": best_command['parameter'],
        "x": best_command['x'],
        "y": best_command['y'],
        "condition": best_command.get('condition', ''),
        "maneuvering_time": best_maneuvering['time'],
        "maneuvering_parameter": best_maneuvering['parameter'],
        "time_offset": time_offset
    }])
    matches = pd.concat([matches, match_entry], ignore_index=True)

    # 移除已匹配的 command 和 maneuvering 点
    unmatched_commands = unmatched_commands.drop(best_idx)
    available_maneuvering_df = available_maneuvering_df.drop(best_maneuvering.name)

# 保存匹配结果
matches.to_csv('heading_audio_maneuvering_match.csv', index=False)