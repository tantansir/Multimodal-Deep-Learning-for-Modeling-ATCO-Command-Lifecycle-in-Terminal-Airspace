import pandas as pd

df = pd.read_csv('data/sample/Track_25-11-2022_.csv')

# 删除无用的列
df = df.drop(columns=['flight_id', 'catgory', 'callsign_eventdatecnt', 'heading'])

# 去除包含空值的行(derek的代码对速度derived_speeding进行了滑动窗口处理，每个callsign开头都有十几行缺失)
df = df.dropna()

# 按指定顺序排列列名
df = df[['tracknumber', 'callsign', 'event_timestamp', 'latitude', 'longitude', 'derived_heading', 'altitude', 'derived_speed', 'CAS']]

# 得到Track.csv
df.to_csv('Track.csv', index=False)
df.head()
