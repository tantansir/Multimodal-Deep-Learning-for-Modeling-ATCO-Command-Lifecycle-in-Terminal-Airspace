import pandas as pd

track_df = pd.read_csv('Track_interpolation.csv')
voice_df = pd.read_csv('Voice.csv')

# 预处理时间戳，将其转换为数值格式
track_df['event_timestamp'] = pd.to_numeric(track_df['event_timestamp'])
voice_df['end_time'] = pd.to_numeric(voice_df['end_time'])
track_df['callsign'] = track_df['callsign'].str.strip().str.upper()
voice_df['suggested_callsign'] = voice_df['suggested_callsign'].str.strip().str.upper()

# 设置匹配时间差阈值为1秒
TIME_THRESHOLD = 1

# 为track_df添加空的列，用于存储voice数据匹配的结果
track_df['next_speed'] = None
track_df['next_heading'] = None
track_df['next_alt'] = None
track_df['complete'] = None
track_df['condition'] = None
track_df['cancel'] = None
track_df['maintain'] = None

# 遍历voice数据集中的每条指令记录
for i, voice_row in voice_df.iterrows():
    # 空管指令的匹配条件
    if 'C' in voice_row['Source']:
        # 根据callsign和时间在track数据集中找到唯一匹配记录
        matched_track = track_df[
            (track_df['callsign'] == voice_row['suggested_callsign']) &
            (abs(track_df['event_timestamp'] - voice_row['end_time']) <= TIME_THRESHOLD)
            ]

        if not matched_track.empty:
            # 只匹配到第一个符合条件的轨迹点
            track_row_index = matched_track.index[0]

            # 为匹配到的轨迹数据行，填充voice数据中的对应值
            track_df.loc[track_row_index, 'next_speed'] = voice_row['next_speed']
            track_df.loc[track_row_index, 'next_heading'] = voice_row['next_heading']
            track_df.loc[track_row_index, 'next_alt'] = voice_row['next_alt']
            track_df.loc[track_row_index, 'complete'] = voice_row['complete']
            track_df.loc[track_row_index, 'condition'] = voice_row['condition']
            track_df.loc[track_row_index, 'cancel'] = voice_row['cancel']
            track_df.loc[track_row_index, 'maintain'] = voice_row['maintain']

# 删除不需要的列
final_df = track_df.drop(
    columns=['derived_speed'])

# 重新排列列的顺序
final_df = final_df[
    ['tracknumber', 'callsign', 'event_timestamp', 'latitude', 'longitude', 'CAS', 'derived_heading', 'altitude',
     'next_speed', 'next_heading', 'next_alt', 'complete', 'condition', 'cancel', 'maintain']]

# 得到matched_voice_to_track.csv
final_df.to_csv('matched_voice_to_track.csv', index=False)
