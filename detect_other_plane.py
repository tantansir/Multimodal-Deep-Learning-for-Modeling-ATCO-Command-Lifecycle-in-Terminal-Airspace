import pandas as pd

# 读取两个文件
track_checked = pd.read_csv("Track_starchecked.csv")
track_interp = pd.read_csv("Track_interpolation.csv")

# 确保时间格式一致（如果是字符串可以加下面这行）
# track_checked['time'] = pd.to_datetime(track_checked['time'])
# track_interp['event_timestamp'] = pd.to_datetime(track_interp['event_timestamp'])

# 建一个基于 event_timestamp 的分组映射：每个时间点有哪些 callsign
timestamp_to_callsigns = track_interp.groupby('event_timestamp')['callsign'].apply(set).to_dict()

# 存储结果
num_other_plane = []

for idx, row in track_checked.iterrows():
    time = row['time']
    own_callsign = row['callsign']

    # 获取同一时间的所有 callsign
    callsigns_at_time = timestamp_to_callsigns.get(time, set())

    # 移除自己
    other_callsigns = callsigns_at_time - {own_callsign}

    # 统计数量
    num_other_plane.append(len(other_callsigns))

# 添加列
track_checked['num_other_plane'] = num_other_plane

# 保存结果（可选）
track_checked.to_csv("Track_starchecked_with_other_plane.csv", index=False)