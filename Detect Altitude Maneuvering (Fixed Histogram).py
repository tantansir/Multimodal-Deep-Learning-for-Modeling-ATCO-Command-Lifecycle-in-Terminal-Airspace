import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def flight_lvs_changes(df, win):

    #fl_df = df[df['altitude'] <= 25000].copy()
    fl_df = df.copy()
    fl_df["fl_smooth"] = signal.savgol_filter(fl_df["altitude"], win, 5)

    # Use altitude levels at every 500 ft as midpoints, with bounds of +/- 250 ft
    midpoints = np.arange(0, 45000, 500)  # Midpoints at every 500 ft
    lower_bounds = midpoints - 100  # Lower bounds for each midpoint  对于JSA762，需要100
    upper_bounds = midpoints + 100  # Upper bounds for each midpoint

    # Generate histogram to identify potential flight level platforms
    fl_hist = np.histogram(fl_df['fl_smooth'].values, bins=lower_bounds)
    fl_hist_df = pd.DataFrame({"count": fl_hist[0], "fl": fl_hist[1][:-1], "ce": fl_hist[1][1:]})
    print(fl_hist_df)
    print()

    # Filter to find the flight levels where aircraft stays for a significant period
    staying_fls = fl_hist_df["fl"][fl_hist_df["count"] > 70]
    staying_lower_bounds = staying_fls.values

    # Detect changes between platforms
    fl_df["platform"] = np.digitize(fl_df["fl_smooth"], staying_lower_bounds, right=False)
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

    return fl_df, change_points_df, lower_bounds, upper_bounds, previous_points_df


track_data = pd.read_csv('Track_interpolation.csv')

selected_callsign = track_data['callsign'].unique()[:1]
# selected_callsign = ["JSA762", "AXM1775"]

plt.figure(figsize=(15, 10))

for callsign in selected_callsign:
    df_selected = track_data[track_data['callsign'] == callsign]

    # Process data for each selected callsign
    fl_df, change_points_df, lower_bounds, upper_bounds, previous_points_df = flight_lvs_changes(df_selected, win=11)

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