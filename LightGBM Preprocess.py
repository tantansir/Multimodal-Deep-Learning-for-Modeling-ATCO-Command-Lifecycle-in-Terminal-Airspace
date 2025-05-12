import pandas as pd
from math import radians, sin, cos, sqrt, atan2, degrees


def load_and_clean_data(csv_files, callsign_wtc_path):
    # Load and concatenate all datasets
    all_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

    # Filter out unwanted rows
    all_data = all_data[~all_data['condition'].str.contains('after', na=False)]
    all_data = all_data[~all_data['event'].str.contains('hold', na=False)]

    # Load and preprocess callsign-WTC data
    callsign_type_wtc = pd.read_excel(callsign_wtc_path)
    callsign_type_wtc.columns = ['callsign', 'unused', 'WTC']
    callsign_type_wtc = callsign_type_wtc[['callsign', 'WTC']].dropna()

    callsign_type_wtc['WTC'] = callsign_type_wtc['WTC'].map({'M': 1, 'H': 2, 'J': 3})

    # Merge with main dataset
    all_data = all_data.merge(callsign_type_wtc, on='callsign', how='left')

    # One-hot encode event type
    all_data = pd.get_dummies(all_data, columns=['event'], prefix='event', drop_first=False)

    # Ensure all dummy columns are numeric (convert True/False to 1/0)
    for col in all_data.columns:
        if col.startswith('event_'):
            all_data[col] = all_data[col].astype(int)

    return all_data


def calculate_distance_and_bearing(data):
    CHANGI_LAT, CHANGI_LON = 1.357, 103.988  # Changi Airport coordinates

    # Calculate distance and bearing
    def haversine_and_bearing(row):
        lat1, lon1, lat2, lon2 = map(radians, [row['y'], row['x'], CHANGI_LAT, CHANGI_LON])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))
        bearing = (degrees(atan2(sin(dlon) * cos(lat2), cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon))) + 360) % 360
        return pd.Series([distance, bearing])

    data[['distance_to_changi', 'bearing_to_changi']] = data.apply(haversine_and_bearing, axis=1)
    return data


def preprocess_data(data, track_data, weather_data):
    # Merge track data
    track_data = track_data.rename(columns={'event_timestamp': 'time', 'derived_heading': 'heading', 'CAS': 'cas'})
    data = data.merge(track_data[['callsign', 'time', 'cas', 'heading', 'altitude']], on=['callsign', 'time'], how='left')

    # Process weather data
    weather_data['time'] = pd.to_datetime(weather_data['time']).dt.hour * 3600 + \
                           pd.to_datetime(weather_data['time']).dt.minute * 60 + \
                           pd.to_datetime(weather_data['time']).dt.second

    # Ensure 'time' columns in both datasets have the same data type
    data['time'] = data['time'].astype('int64')
    weather_data['time'] = weather_data['time'].astype('int64')

    data = pd.merge_asof(data.sort_values('time'), weather_data[['time', 'drct', 'sknt', 'skyl1']], on='time', direction='nearest')

    # Add distance and bearing features
    data = calculate_distance_and_bearing(data)

    # Add peak hour feature
    data['is_peakhour'] = data['time'].apply(
        lambda t: 1 if (32400 <= t <= 39599 or 57600 <= t <= 64799) else 0
    )

    return data


if __name__ == "__main__":
    # File paths
    csv_files = [
        "副本-altitude_audio_maneuvering_match.csv",
        "999CAS_audio_maneuvering_match.csv",
        "副本-heading_audio_maneuvering_match.csv"
    ]

    callsign_wtc_path = "data/sample/CallsignTYPEAndWTC.xlsx"
    weather_file_path = "data/sample/Weather.csv"
    track_file_path = "Track_interpolation.csv"

    # Load and preprocess data
    data = load_and_clean_data(csv_files, callsign_wtc_path)
    weather_data = pd.read_csv(weather_file_path)
    track_data = pd.read_csv(track_file_path)

    data = preprocess_data(data, track_data, weather_data)
    data = data.drop_duplicates()
    data.to_csv("data3.csv", index=False)