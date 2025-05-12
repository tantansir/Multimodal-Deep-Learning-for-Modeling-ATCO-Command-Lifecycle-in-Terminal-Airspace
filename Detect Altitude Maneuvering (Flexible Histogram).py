import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def flight_lvs_changes(df, win):

    fl_df = df.copy()
    fl_df["fl_smooth"] = signal.savgol_filter(df["altitude"], win, 5)

    # Generate histogram to identify potential flight level platforms
    fl_hist = np.histogram(fl_df['fl_smooth'].values, bins=185)  #数据范围0-43000, 极端情况下43000/345=124.6 < 125, 实际情况飞机每秒下降25英尺左右，检测大概有5秒左右延迟
    fl_hist_df = pd.DataFrame({"count": fl_hist[0], "fl": fl_hist[1][:-1], "ce": fl_hist[1][1:]})
    print(fl_hist_df)
    print()

    fl_midpoints = (fl_hist[1][1:] + fl_hist[1][:-1]) / 2  # Midpoints of each bin
    fl_upper_bounds = fl_hist[1][1:]  # Upper bounds of each bin
    fl_lower_bounds = fl_hist[1][:-1]  # Lower bounds of each bin

    # Filter to find the flight levels where aircraft stays for platforms
    staying_fls = fl_lower_bounds[fl_hist_df["count"] > 40]
    print(staying_fls)
    print()

    # Detect changes between platforms
    fl_df["platform"] = np.digitize(fl_df["fl_smooth"], staying_fls)
    print(fl_df["platform"])
    print()

    # Identify the points where platform changes (maneuvering start points)
    fl_df["platform_change"] = fl_df["platform"].diff().fillna(0) != 0
    print(fl_df["platform_change"])
    print()

    # Filter out low altitude points (voice.complete=1)
    valid_points = fl_df[fl_df['altitude'] >= 0].copy()

    # Detect first change point within a trend (trend detection improvement)
    change_points = []
    in_transition = False
    trend_detected = False

    for idx, row in valid_points.iterrows():
        if row['platform_change'] and not in_transition:
            if not trend_detected:
                change_points.append(row)
                trend_detected = True
            in_transition = True
        if not row['platform_change']:
            in_transition = False
            trend_detected = False

    change_points_df = pd.DataFrame(change_points)
    print(change_points_df['event_timestamp'], change_points_df['altitude'])
    print()

    # Retrieve previous point for each change point
    previous_points = []
    for idx, row in change_points_df.iterrows():
        if idx > 0:
            previous_points.append(valid_points.loc[idx - 1])

    previous_points_df = pd.DataFrame(previous_points)
    print(previous_points_df['event_timestamp'], previous_points_df['altitude'])
    print()

    return fl_df, change_points_df, fl_lower_bounds, fl_upper_bounds, previous_points_df


track_data = pd.read_csv('Track_interpolation.csv')

selected_callsign = track_data['callsign'].unique()[:1]
# selected_callsign = ["JSA762"]

plt.figure(figsize=(15, 10))


for callsign in selected_callsign:
    df_selected = track_data[track_data['callsign'] == callsign]

    # Process data for each selected callsign
    fl_df, change_points_df, fl_lower_bounds, fl_upper_bounds, previous_points_df = flight_lvs_changes(df_selected, win=11)

    # Plot altitude vs time for each callsign
    plt.plot(df_selected['event_timestamp'], df_selected['altitude'], label=f'{callsign} Original', alpha=0.5)
    plt.plot(fl_df['event_timestamp'], fl_df['fl_smooth'], label=f'{callsign} Smoothed')

    # Mark change points on the plot using real altitude values
    if not change_points_df.empty:
        plt.scatter(change_points_df['event_timestamp'], change_points_df['altitude'], color='red',
                    label=f'{callsign} Change Points')
        # Annotate each change point
        for idx, row in change_points_df.iterrows():
            plt.text(row['event_timestamp'], row['altitude'], f"({row['event_timestamp']}, {row['altitude']})", fontsize=8, color='red')

    # Mark previous points on the plot
    if not previous_points_df.empty:
        plt.scatter(previous_points_df['event_timestamp'], previous_points_df['altitude'], color='blue', label=f'{callsign} Previous Points')
        # Annotate each previous point
        for idx, row in previous_points_df.iterrows():
            plt.text(row['event_timestamp'], row['altitude'], f"({row['event_timestamp']}, {row['altitude']})", fontsize=8, color='blue')


plt.xlabel('Time')
plt.ylabel('Altitude')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Altitude with Detected Maneuvering Points and Platforms')
plt.tight_layout()
plt.show()