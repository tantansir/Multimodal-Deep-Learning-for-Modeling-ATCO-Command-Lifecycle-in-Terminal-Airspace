import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal


def determine_smoothing_factor(cas_values, window_size=50):
    """
    根据滑动窗口标准差动态确定平滑因子 s.
    - 如果窗口内波动标准差较大，则返回较高的平滑因子。
    - 如果窗口内波动标准差较小，则返回较低的平滑因子。
    """
    # 过滤出大于 280 节的 CAS 值
    cas_high_speed = cas_values[cas_values > 280]

    # 如果大于 280 节的数据不足以进行窗口计算，则直接返回默认平滑因子
    if len(cas_high_speed) < window_size:
        return 7000

    # 滑动窗口计算标准差
    rolling_std = pd.Series(cas_values).rolling(window=window_size, min_periods=1).std()
    print(rolling_std.mean())
    # 判断波动情况
    if rolling_std.mean() > 5.5:
        return 84000
    elif rolling_std.mean() > 5.3:
        return 56000
    elif rolling_std.mean() > 5.1:
        return 28000
    elif rolling_std.mean() > 4.7:
        return 21000
    elif rolling_std.mean() > 4.3:
        return 14000 #7000 10000 14000
    elif rolling_std.mean() > 0:
        return 7000 #7000 10000 14000


# selected_callsign = ["SIA7855"]


def cas_changes(df):
    """
    Detect CAS (Calibrated Air Speed) changes using sliding window detection based on the CAS histogram.
    """
    fl_df = df.copy()
    # Use spline interpolation for smoothing CAS data
    cas = fl_df["CAS"].values
    time = fl_df["event_timestamp"].values

    # 动态设置平滑因子 s
    smoothing_factor = determine_smoothing_factor(cas)

    # Perform spline interpolation
    spline = interpolate.UnivariateSpline(time, cas, k=5, s=smoothing_factor)  # k for cubic spline, s for smoothing factor
    fl_df["cas_smooth"] = spline(time)

    # fl_df["cas_smooth"] = signal.savgol_filter(fl_df["CAS"], 200, 5)

    # Sliding window parameters 10 7 3
    window_size = 30
    midpoints = np.arange(0, 370, 5)
    lower_bounds = midpoints[:-1] + 2.5
    upper_bounds = midpoints[1:] + 2.5

    print(lower_bounds)
    print(upper_bounds)

    platforms = []
    for i in range(len(fl_df) - window_size):
        window_data = fl_df['cas_smooth'].iloc[i:i + window_size]

        # Histogram for detecting stable platforms in CAS
        fl_hist = np.histogram(window_data, bins=np.concatenate((lower_bounds, [upper_bounds[-1]])))
        counts = fl_hist[0]

        max_count = np.max(counts)
        if max_count >= (window_size * 0.95):  # High threshold for CAS stability
            platform_bin = np.argmax(counts)
            platform_cas = midpoints[platform_bin]
            platforms.append((fl_df['event_timestamp'].iloc[i], platform_cas))

    print(platforms)
    if not platforms:
        print("No platforms detected. Returning empty results.")
        return fl_df, pd.DataFrame(), pd.DataFrame(), lower_bounds, upper_bounds, pd.DataFrame(), pd.DataFrame()
    platform_df = pd.DataFrame(platforms, columns=["event_timestamp", "CAS"])

    # Detect changes in CAS platforms by keeping the last stable point in each block
    last_stable_points = platform_df[(platform_df['CAS'] != platform_df['CAS'].shift(-1))]
    last_stable_points = last_stable_points.copy()
    last_stable_points.loc[:, "cas_smooth"] = last_stable_points["event_timestamp"].map(
        fl_df.set_index("event_timestamp")["cas_smooth"]
    )
    #print(last_stable_points)

    # Step 1: Remove points with timestamp difference less than 60 seconds
    filtered_points = []
    for i in range(len(last_stable_points)):
        if i == 0 or (last_stable_points.iloc[i]["event_timestamp"] - last_stable_points.iloc[i - 1]["event_timestamp"]) >= 60:
            filtered_points.append(last_stable_points.iloc[i])

    last_stable_points = pd.DataFrame(filtered_points)
    #print("After removing points with <60s intervals:\n", last_stable_points)

    # Step 2: Retain points based on CAS +-5 range
    final_points = []
    i = 0
    while i < len(last_stable_points):
        current_point = last_stable_points.iloc[i]
        j = i

        # Find the last point within +-5 CAS range of the current point
        while j < len(last_stable_points) and abs(last_stable_points.iloc[j]["CAS"] - current_point["CAS"]) <= 5:
            j += 1

        # Append the last point within the range
        final_points.append(last_stable_points.iloc[j - 1])
        i = j  # Move to the next range after the last matched point

    last_stable_points = pd.DataFrame(final_points)
    #print("After applying CAS +-5 range filter (keeping the last point):\n", last_stable_points)

    # Retrieve change points by matching timestamps in fl_df and adding a time window
    change_points = []
    for idx, row in last_stable_points.iterrows():
        timestamp = row["event_timestamp"]
        fl_window = fl_df[fl_df['event_timestamp'] > timestamp].iloc[:int(window_size * 0.95)]
        if not fl_window.empty:
            change_point_row = fl_window.iloc[-1].copy()
            change_point_row['CAS'] = row['CAS']  # Add CAS to the change point row
            change_points.append(change_point_row)

    change_points_df = pd.DataFrame(change_points)
    change_points_df = change_points_df.copy()
    change_points_df.loc[:, "cas_smooth"] = change_points_df["event_timestamp"].map(
        fl_df.set_index("event_timestamp")["cas_smooth"]
    )
    #print(change_points_df)

    # 用平滑后的数据，提取出来的last point的+-3融合(抛弃该想法)

    # 作平行线，若与之相交的cas_smooth点，且时间差小于60，则换成该点

    refined_points = []
    for i in range(len(change_points_df)):
        current_point = change_points_df.iloc[i]
        current_time = current_point["event_timestamp"]
        current_cas_smooth = current_point["cas_smooth"] #cas_smooth

        # 在fl_df中查找符合条件的后续点
        later_points = fl_df[(fl_df["event_timestamp"] > current_time) &
                             (fl_df["event_timestamp"] <= current_time + 60)]
        #print(later_points)
        same_cas_points = later_points[abs(later_points["cas_smooth"] - current_cas_smooth) < 1]

        if not same_cas_points.empty:
            # 找到符合条件的最后一个点，替换为该点
            refined_point = same_cas_points.iloc[-1]
            refined_points.append(refined_point)
        else:
            # 如果没有符合条件的点，保留当前点
            refined_points.append(current_point)

    change_points_df = pd.DataFrame(refined_points).drop_duplicates().reset_index(drop=True)
    #print("After refining with time difference <45s and same CAS smooth on the cas_smooth curve:\n", change_points_df)

    # 作平行线，若与之相交的CAS点，且时间差小于60，则换成该点

    refined_points = []
    for i in range(len(change_points_df)):
        current_point = change_points_df.iloc[i]
        current_time = current_point["event_timestamp"]
        current_cas_smooth = current_point["CAS"] #CAS

        # 在fl_df中查找符合条件的后续点
        later_points = fl_df[(fl_df["event_timestamp"] > current_time) &
                             (fl_df["event_timestamp"] <= current_time + 60)]
        #print(later_points)
        same_cas_points = later_points[abs(later_points["CAS"] - current_cas_smooth) < 1]

        if not same_cas_points.empty:
            # 找到符合条件的最后一个点，替换为该点
            refined_point = same_cas_points.iloc[-1]
            refined_points.append(refined_point)
        else:
            # 如果没有符合条件的点，保留当前点
            refined_points.append(current_point)

    change_points_df = pd.DataFrame(refined_points).drop_duplicates().reset_index(drop=True)


    # Detect changes in CAS platforms by keeping the first stable point in each block
    first_stable_points = platform_df[(platform_df['CAS'] != platform_df['CAS'].shift(1))]
    first_stable_points = first_stable_points.copy()
    first_stable_points.loc[:, "cas_smooth"] = first_stable_points["event_timestamp"].map(
        fl_df.set_index("event_timestamp")["cas_smooth"]
    )
    #print("First stable points:\n", first_stable_points)

    # Step 1: Remove points with timestamp difference less than 60 seconds in first_stable_points
    filtered_first_points = []
    for i in range(len(first_stable_points)):
        if i == 0 or (first_stable_points.iloc[i]["event_timestamp"] - first_stable_points.iloc[i - 1]["event_timestamp"]) >= 60:
            filtered_first_points.append(first_stable_points.iloc[i])

    first_stable_points = pd.DataFrame(filtered_first_points)
    #print("After removing points with <60s intervals in first_stable_points:\n", first_stable_points)

    # Step 2: Retain points based on CAS +-5 range in first_stable_points
    final_first_points = []
    i = 0
    while i < len(first_stable_points):
        current_point = first_stable_points.iloc[i]
        j = i

        # Find the first point within +-5 CAS range of the current point
        while j < len(first_stable_points) and abs(first_stable_points.iloc[j]["CAS"] - current_point["CAS"]) <= 5:
            j += 1

        # Append the first point within the range
        final_first_points.append(first_stable_points.iloc[i])
        i = j  # Move to the next range after the last matched point

    first_stable_points = pd.DataFrame(final_first_points)
    #print("After applying CAS +-5 range filter (keeping the first point):\n", first_stable_points)

    change_points2 = []
    for idx, row in first_stable_points.iterrows():
        timestamp = row["event_timestamp"]
        fl_window = fl_df[fl_df['event_timestamp'] > timestamp].iloc[:int(window_size * 0.05)]
        if not fl_window.empty:
            change_point_row = fl_window.iloc[-1].copy()
            change_point_row['CAS'] = row['CAS']  # Add CAS to the change point row
            change_points2.append(change_point_row)

    change_points_df2 = pd.DataFrame(change_points2)
    change_points_df2 = change_points_df2.copy()
    change_points_df2.loc[:, "cas_smooth"] = change_points_df2["event_timestamp"].map(
        fl_df.set_index("event_timestamp")["cas_smooth"]
    )
    #print(change_points_df2)

    return fl_df, change_points_df, change_points_df2, lower_bounds, upper_bounds, last_stable_points, first_stable_points


track_data = pd.read_csv('Track_interpolation.csv')

# selected_callsign = track_data['callsign'].unique()
selected_callsign = track_data['callsign'].unique()

all_change_points = pd.DataFrame(columns=["time", "callsign", "event", "parameter", "x", "y"])

plt.figure(figsize=(15, 10))

for callsign in selected_callsign:
    df_selected = track_data[track_data['callsign'] == callsign]
    df_selected = df_selected[df_selected['altitude'] < 30000] #根据这个调动态平滑参

    # Process data for the selected callsign
    fl_df, change_points_df, change_points_df2, lower_bounds, upper_bounds, last_stable_points_df, first_stable_points = cas_changes(df_selected)

    # Plot CAS vs time for the callsign
    plt.plot(df_selected['event_timestamp'], df_selected['CAS'], label=f'{callsign} Original CAS', alpha=0.5)
    plt.plot(fl_df['event_timestamp'], fl_df['cas_smooth'], label=f'{callsign} Smoothed CAS')

    # Mark change points on the plot using real CAS values
    if not change_points_df.empty:
        plt.scatter(change_points_df['event_timestamp'], change_points_df['cas_smooth'], color='red',
                    label=f'{callsign} Change Points')

    if not change_points_df.empty:
        # 为 change_points_df 添加 "parameter" 列
        change_points_df["parameter"] = np.nan

        for idx in range(len(change_points_df) - 1):
            current_point = change_points_df.iloc[idx]
            next_point_cas = change_points_df.iloc[idx + 1]["CAS"]

            # 找到下一个点 CAS 值所在的 histogram 区间
            for j in range(len(lower_bounds)):
                if lower_bounds[j] <= next_point_cas < upper_bounds[j]:
                    # 计算该区间的中位数
                    median = (lower_bounds[j] + upper_bounds[j]) / 2

                    # 以5结尾则进位
                    if median % 10 == 5:
                        median = (int(median) // 10 + 1) * 10

                    # 将进位后的中位数赋值给 parameter 列
                    change_points_df.at[idx, "parameter"] = median
                    break

        print("Updated change_points_df with parameter column:\n", change_points_df)

        change_points_df["event"] = "velocity change"
        change_points_df["x"] = change_points_df["event_timestamp"].map(fl_df.set_index("event_timestamp")["longitude"])
        change_points_df["y"] = change_points_df["event_timestamp"].map(fl_df.set_index("event_timestamp")["latitude"])
        change_points_df = change_points_df.rename(columns={"event_timestamp": "time"})
        change_points_df["callsign"] = callsign
        # 只保留 parameter 不为空的行
        change_points_df = change_points_df[change_points_df["parameter"].notna()]

        change_points_df = change_points_df[["time", "callsign", "event", "parameter", "x", "y"]]
        # 将当前 callsign 的数据附加到汇总表中
        all_change_points = pd.concat([all_change_points, change_points_df], ignore_index=True)

    # if not change_points_df2.empty:
    #     plt.scatter(change_points_df2['event_timestamp'], change_points_df2['cas_smooth'], color='green',
    #                 label=f'{callsign} Stable Points')

    plt.xlabel('Time')
    plt.ylabel('CAS')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('CAS with Detected Maneuvering Points and Platforms')
    plt.tight_layout()
    plt.show()

# 保存所有变化点
#all_change_points.to_csv("999all_callsigns_CAS_maneuvering_points.csv", index=False)