import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def altitude_changes(traj_zs):
    """
    Detect altitude changes using a combination of differences, convolution, and peak finding.
    """
    # Define the parameters for difference and filtering
    diff_gp = 5  # Difference gap for altitude
    win = 60  # Window size for filtering
    distance_threshold = 100  # Threshold for minimum distance between peaks

    # Step 1: Calculate altitude difference over gap size
    diffz = np.divide(traj_zs[diff_gp:] - traj_zs[0:-diff_gp], diff_gp)

    # Step 2: Convolve with a low-pass filter (using FIR window)
    diffz = np.convolve(diffz, signal.firwin(win, 0.01), mode='valid')

    # Step 3: Find the peaks where altitude changes indicate stable platforms
    z_dfpk = signal.find_peaks(-np.abs(diffz), height=-5, distance=distance_threshold)

    # Initialize the altitude levels array
    if z_dfpk[0].shape[0] > 0:
        z_levels = np.array([z_dfpk[0][0], traj_zs[z_dfpk[0][0]]])

        # Iterate through detected peaks to construct altitude levels
        for i in range(len(z_dfpk[0]) - 1):
            if abs(z_dfpk[0][i + 1] - z_dfpk[0][i]) > distance_threshold:
                z_levels = np.append(z_levels, [z_dfpk[0][i + 1], traj_zs[z_dfpk[0][i + 1]]], axis=0)

        # Reshape z_levels and round altitude to nearest 1000 feet
        z_levels = z_levels.reshape((-1, 2))
        z_levels[:, 1] = np.round(z_levels[:, 1] / 1000) * 1000

        # Initialize the final altitude levels
        z_lv = np.array([z_levels[0, 0], z_levels[0, 1]])

        for i in range(len(z_levels) - 1):
            if abs(z_levels[i + 1, 1] - z_levels[i, 1]) < 2000:
                continue
            else:
                z_lv = np.append(z_lv, [z_levels[i + 1, 0], z_levels[i + 1, 1]], axis=0)

        z_lv = z_lv.reshape((-1, 2))

        # Add first and last points
        z_lv = np.insert(z_lv, 0, np.array([0, traj_zs[0]])).reshape((-1, 2))
        z_lv = np.insert(z_lv, z_lv.shape[0] * 2, np.array([traj_zs.shape[0] - 1, traj_zs[-1]])).reshape((-1, 2))
    else:
        # In case no peaks are detected, use the start and end points
        z_lv = np.array([0, traj_zs[0]]).reshape((-1, 2))
        z_lv = np.insert(z_lv, z_lv.shape[0] * 2, np.array([traj_zs.shape[0] - 1, traj_zs[-1]])).reshape((-1, 2))

    return z_lv


track_data = pd.read_csv('Track_interpolation.csv')

selected_callsign = track_data['callsign'].unique()[:1]
# selected_callsign = ["AXM1775"]

plt.figure(figsize=(15, 10))

for callsign in selected_callsign:
    df_selected = track_data[track_data['callsign'] == callsign]
    traj_zs = df_selected['altitude'].values
    time = df_selected['event_timestamp'].values

    # Detect altitude changes
    altitude_levels = altitude_changes(traj_zs)

    # Plot altitude vs time for the selected callsign
    plt.plot(time, traj_zs, label=f'{callsign} Original Altitude', alpha=0.5)

    # Mark detected altitude levels on the plot
    for i in range(altitude_levels.shape[0]):
        plt.axhline(y=altitude_levels[i, 1], color='r', linestyle='--', alpha=0.6)
        plt.text(time[int(altitude_levels[i, 0])], altitude_levels[i, 1], f'{altitude_levels[i, 1]} ft', color='red',
                 fontsize=10)

plt.xlabel('Time')
plt.ylabel('Altitude (ft)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Altitude with Detected Platforms (AXM1775)')
plt.tight_layout()
plt.show()
