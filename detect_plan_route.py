import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# ---------- 1. 读取数据 ----------
track_data = pd.read_csv('Track_starcheck.csv')
waypoints_df = pd.read_csv('sid_star.csv')

# 读取所有STAR路线数据
star_folder_path = "Track_processing_plotting/tools/STAR"
star_files = [os.path.join(star_folder_path, f) for f in os.listdir(star_folder_path)
              if f.startswith("STAR") and f.endswith(".csv")]

# ---------- 2. 经纬度解析 ----------
def dms_to_decimal(dms_str):
    if pd.isna(dms_str): return None
    dms_str = str(dms_str).strip()

    if dms_str[-1] in ['N', 'S', 'E', 'W']:
        direction = dms_str[-1]
        dms_core = dms_str[:-1]

        if len(dms_core) == 6:  # lat: DDMMSS
            deg = int(dms_core[:2])
            min = int(dms_core[2:4])
            sec = int(dms_core[4:])
        elif len(dms_core) == 7:  # lon: DDDMMSS
            deg = int(dms_core[:3])
            min = int(dms_core[3:5])
            sec = int(dms_core[5:])
        else:
            raise ValueError(f"Invalid DMS format: {dms_str}")
    else:
        raise ValueError(f"No direction found in DMS: {dms_str}")

    decimal = deg + min / 60 + sec / 3600
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal


def convert_lat_lon(row):
    lat = dms_to_decimal(row['Latitude'])
    lon = dms_to_decimal(row['Longitude'])
    return pd.Series({'lat': lat, 'lon': lon})

waypoints_df[['lat', 'lon']] = waypoints_df.apply(convert_lat_lon, axis=1)
waypoint_dict = waypoints_df.set_index('Name')[['lat', 'lon']].dropna().to_dict('index')

# ---------- 3. 构建STAR路径 ----------
def clean_wp_name(name):
    return str(name).strip().upper()

waypoints_df['Name_clean'] = waypoints_df['Name'].apply(clean_wp_name)
waypoint_dict = waypoints_df.set_index('Name_clean')[['lat', 'lon']].dropna().to_dict('index')

star_routes = []

for idx, file in enumerate(star_files):
    df = pd.read_csv(file, header=None)
    points = []

    for i, row in df.iterrows():
        wp_name = clean_wp_name(row[1])
        if wp_name in waypoint_dict:
            latlon = waypoint_dict[wp_name]
            if pd.notna(latlon['lat']) and pd.notna(latlon['lon']):
                points.append(latlon)

    print(f"Loaded {len(points)} waypoints from {file}")
    if len(points) > 1:
        star_routes.append(points)

# ---------- 4. 判断点是否在航路上 ----------
def point_to_segment_distance(p, a, b):
    """计算点p到线段ab的最短距离（单位：km）"""
    # 将lat/lon转换为笛卡尔坐标（简化处理，误差可接受）
    from numpy import array, dot
    from numpy.linalg import norm

    def to_vec(pt): return array([pt['lat'], pt['lon']])
    p, a, b = to_vec(p), to_vec(a), to_vec(b)

    ab = b - a
    ap = p - a
    ab_len2 = dot(ab, ab)

    # 投影比例
    t = dot(ap, ab) / ab_len2 if ab_len2 != 0 else -1

    if t < 0.0:
        closest = a
    elif t > 1.0:
        closest = b
    else:
        closest = a + t * ab

    # 把最近点从笛卡尔转回lat/lon（只是近似，不影响km距离）
    closest_latlon = {'lat': closest[0], 'lon': closest[1]}
    return geodesic((p[0], p[1]), (closest[0], closest[1])).km

def point_to_route_distance(point, route):
    """返回点到整条路径中最近线段的距离"""
    min_dist = float('inf')
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i+1]
        dist = point_to_segment_distance(point, a, b)
        min_dist = min(min_dist, dist)
    return min_dist


# ---------- 5. 添加新列 ----------
threshold_km = 2.0  # 距离小于30km认为在航路上

is_planroute = []
num_planroute = []
nearest_wp_names = []
min_wp_dists = []

for idx, row in track_data.iterrows():
    point = {'lat': row['y'], 'lon': row['x']}
    found = False
    matched_route = -1

    for i, route in enumerate(star_routes):
        dist = point_to_route_distance(point, route)
        if dist < threshold_km:
            found = True
            matched_route = i
            break

    is_planroute.append(found)
    num_planroute.append(matched_route if found else None)

    # 添加最近航路点及距离
    min_wp_dist = float('inf')
    nearest_wp_name = None
    current_point = (row['y'], row['x'])  # 注意 (lat, lon)

    for wp_name, wp_coord in waypoint_dict.items():
        if pd.notna(wp_coord['lat']) and pd.notna(wp_coord['lon']):
            wp_point = (wp_coord['lat'], wp_coord['lon'])
            dist = geodesic(current_point, wp_point).km
            if dist < min_wp_dist:
                min_wp_dist = dist
                nearest_wp_name = wp_name

    nearest_wp_names.append(nearest_wp_name)
    min_wp_dists.append(min_wp_dist)

track_data['is_planroute'] = is_planroute
track_data['num_planroute'] = num_planroute

track_data['nearest_wp'] = nearest_wp_names
track_data['nearest_wp_dist_km'] = min_wp_dists

# ---------- 6. 可视化 ----------
plt.figure(figsize=(12, 10))

# 绘制所有STAR路径，并标记waypoints
for route in star_routes:
    lats = [p['lat'] for p in route]
    lons = [p['lon'] for p in route]
    plt.plot(lons, lats, color='lightgray', linestyle='--', alpha=0.8, linewidth=1)

# 标注所有waypoints
for name, coord in waypoint_dict.items():
    if pd.notna(coord['lat']) and pd.notna(coord['lon']):
        plt.scatter(coord['lon'], coord['lat'], color='blue', s=10)
        plt.text(coord['lon'], coord['lat'], name, fontsize=8, ha='left', va='bottom', color='blue')

# 绘制轨迹点
on_route = track_data[track_data['is_planroute']]
off_route = track_data[~track_data['is_planroute']]

plt.scatter(off_route['x'], off_route['y'], color='red', label='Off Plan Route', s=8, alpha=0.7)
plt.scatter(on_route['x'], on_route['y'], color='green', label='On Plan Route', s=8, alpha=0.7)

# 图形配置
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.title("Aircraft Trajectory with STAR Routes and Waypoints", fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()


track_data.to_csv('Track_starchecked.csv', index=False)