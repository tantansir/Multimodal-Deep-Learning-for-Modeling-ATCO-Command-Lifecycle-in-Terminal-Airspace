from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Fixed Histogram+Sliding Window (目前最好的检测Altitude方法，其余三个表现不如它)

def flight_lvs_changes(df, win):
    """
    Refactor the logic to detect flight level changes using sliding window detection
    based on the altitude histogram and sequential platform identification.
    """
    fl_df = df.copy()
    fl_df["fl_smooth"] = signal.savgol_filter(fl_df["altitude"], win, 5)

    # Calculate ROCD using a longer sliding window to reduce noise
    rocd_window = 30  # Window size in seconds for ROCD calculation, can adjust based on data granularity altitude相差25的话 rocd是-1500
    fl_df["ROCD"] = fl_df["altitude"].rolling(window=rocd_window).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / rocd_window * 60)

    # Parameters for CDO detection
    cdo_threshold = -300  # ROCD threshold for descent, adjust as needed
    min_cdo_duration = 30  # minimum duration in seconds for CDO, adjust as needed

    # Detect CDO segments
    fl_df["CDO"] = (fl_df["ROCD"] < cdo_threshold).astype(int)  # mark where ROCD < threshold
    fl_df["CDO_segment"] = (fl_df["CDO"].diff() != 0).cumsum()  # segment CDO regions


    # Identify CDO segments that meet the minimum duration
    cdo_segments = []
    for segment_id in fl_df["CDO_segment"].unique():
        segment = fl_df[fl_df["CDO_segment"] == segment_id]
        if segment["CDO"].iloc[0] == 1 and len(segment) >= min_cdo_duration:
            cdo_segments.append(segment)

    # Store CDO segment data for plotting
    cdo_data = pd.concat(cdo_segments) if cdo_segments else pd.DataFrame(columns=fl_df.columns)

    # Create sliding window to detect platforms
    window_size = 60  # Adjustable window size for platform detection
    midpoints = np.arange(0, 45000, 500)
    lower_bounds = midpoints - 100  # Midpoints with tolerance bounds
    upper_bounds = midpoints + 100

    print(lower_bounds)
    print(upper_bounds)

    # Create an empty list to store detected platforms with their timestamps
    platforms = []

    for i in range(len(fl_df) - window_size):
        window_data = fl_df['fl_smooth'].iloc[i:i + window_size]

        # Calculate the histogram within the window to detect stable platforms
        fl_hist = np.histogram(window_data, bins=lower_bounds)
        counts = fl_hist[0]

        # Find if the majority of points fall into one bin (a platform)
        max_count = np.max(counts)
        if max_count >= (window_size * 0.95):  # Threshold for platform detection
            platform_bin = np.argmax(counts)
            platform_altitude = midpoints[platform_bin]
            platforms.append((fl_df['event_timestamp'].iloc[i], platform_altitude))

    print(platforms)

    # Convert detected platforms into a DataFrame
    platform_df = pd.DataFrame(platforms, columns=["event_timestamp", "altitude"])

    # Identify only the last stable point in each platform
    last_stable_points = platform_df[(platform_df['altitude'] != platform_df['altitude'].shift(-1))]

    print(last_stable_points)

    # Retrieve change points
    change_points = []
    for idx, row in last_stable_points.iterrows():
        timestamp = row["event_timestamp"]
        fl_window = fl_df[fl_df['event_timestamp'] > timestamp].iloc[:int(window_size * 0.85)]
        if not fl_window.empty:
            change_points.append(fl_window.iloc[-1])

    change_points_df = pd.DataFrame(change_points)

    return fl_df, change_points_df, cdo_data, last_stable_points


track_data = pd.read_csv('Track_interpolation.csv')

# selected_callsign = ["JSA762", "AXM1775"]
selected_callsign = track_data['callsign'].unique()
#selected_callsign = ["SIA185"]
all_change_points = pd.DataFrame(columns=["time", "callsign", "event", "parameter", "x", "y"]) #去除cdo的所有altitude change点

plt.figure(figsize=(15, 10))

for callsign in selected_callsign:
    df_selected = track_data[track_data['callsign'] == callsign]

    # Process data for the selected callsign
    fl_df, change_points_df, cdo_data, last_stable_points = flight_lvs_changes(df_selected, win=11)

    # Plot altitude vs time for the callsign
    plt.plot(df_selected['event_timestamp'], df_selected['altitude'], label=f'{callsign} Original', alpha=0.5)
    plt.plot(fl_df['event_timestamp'], fl_df['fl_smooth'], label=f'{callsign} Smoothed')

    # Mark change points on the plot
    if not change_points_df.empty:
        plt.scatter(change_points_df['event_timestamp'], change_points_df['altitude'], color='red', label=f'{callsign} Change Points')

    # 更新变化点表格
    if not change_points_df.empty:
        change_points_df["event"] = "altitude change"
        window_size = 60
        parameters = []

        for idx, row in change_points_df.iterrows():
            # 获取当前变化点的时间
            current_time = row['event_timestamp']
            next_platform = last_stable_points[last_stable_points['event_timestamp'] > current_time - 0.85 * window_size].head(1)

            # 如果找到匹配的平台，使用其 altitude 值作为 parameter；否则为 NaN
            if not next_platform.empty:
                parameter_value = next_platform.iloc[0]['altitude']  # 取第一个满足条件的平台
            else:
                parameter_value = None

            parameters.append(parameter_value)

        # 将下一个平台的高度值作为 parameter
        change_points_df["parameter"] = parameters
        change_points_df["parameter"] = change_points_df["parameter"].fillna(0)

        change_points_df["x"] = change_points_df["event_timestamp"].map(
            fl_df.set_index("event_timestamp")["longitude"])
        change_points_df["y"] = change_points_df["event_timestamp"].map(
            fl_df.set_index("event_timestamp")["latitude"])
        change_points_df["callsign"] = callsign
        change_points_df = change_points_df.rename(columns={"event_timestamp": "time"})

        change_points_df = change_points_df[["time", "callsign", "event", "parameter", "x", "y"]]

        print(change_points_df)

    # Remove change points that are within CDO segments
    if not cdo_data.empty:
        # Find timestamps in change_points_df that are also in CDO segments
        change_points_df = change_points_df[~change_points_df['time'].isin(cdo_data['event_timestamp'])]
        plt.scatter(cdo_data['event_timestamp'], cdo_data['altitude'], color='blue', label=f'{callsign} CDO Segments', marker='x')
        all_change_points = pd.concat([all_change_points, change_points_df], ignore_index=True)

    # # Mark last stable points on the plot
    # if not last_stable_points_df.empty:
    #     plt.scatter(last_stable_points_df['event_timestamp'], last_stable_points_df['altitude'], color='green',
    #                 label=f'{callsign} Last Stable Points')

    # plt.xlabel('Time')
    # plt.ylabel('Altitude')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.title('Altitude with Detected Maneuvering Points and Platforms')
    # plt.tight_layout()
    # plt.show()

# 保存所有变化点
all_change_points.to_csv("999all_callsigns_altitude_maneuvering_points.csv", index=False)