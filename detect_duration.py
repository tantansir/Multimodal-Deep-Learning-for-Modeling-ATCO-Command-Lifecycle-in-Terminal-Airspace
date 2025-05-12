import pandas as pd
import numpy as np

# 读取数据
track_df = pd.read_csv("Track_starchecked_with_other_plane.csv")
voice_df = pd.read_csv("Voice.csv")

# 向下取整 end_time 用于匹配
voice_df['end_time_floor'] = np.floor(voice_df['end_time']).astype(int)

# 创建一个映射表 {(callsign, end_time_floor): duration}
voice_df['duration'] = voice_df['end_time'] - voice_df['start_time']
voice_map = {
    (row['suggested_callsign'], row['end_time_floor']): row['duration']
    for _, row in voice_df.iterrows()
}

# 查找匹配并添加 duration
durations = []
for _, row in track_df.iterrows():
    key = (row['callsign'], int(row['time']))
    duration = voice_map.get(key, None)
    durations.append(duration)

# 添加新列
track_df['duration'] = durations

# 保存结果（如果需要）
track_df.to_csv("Track_used_for_train.csv", index=False)
