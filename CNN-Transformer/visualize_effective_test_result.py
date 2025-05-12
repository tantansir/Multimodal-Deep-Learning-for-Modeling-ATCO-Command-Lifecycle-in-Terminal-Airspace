import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

# Load data and choose first 5
df = pd.read_csv("effective_prediction_results_single.csv")
demo = df.head(5).copy()

# Relative time calculation
t0 = min(demo["voice_time"].min(), demo["maneuvering_time"].min())
demo["maneuver_start"] = demo["maneuvering_time"] - t0
demo["maneuver_end"] = demo["maneuver_start"] + 5

# Real and predicted voice timing
demo["voice_end"] = demo["maneuver_start"] - demo["time_offset"]
demo["voice_start"] = demo["voice_end"] - demo["duration"]
demo["pred_voice_end"] = demo["maneuver_start"] - demo["predicted_offset"]
demo["pred_voice_start"] = demo["pred_voice_end"] - demo["predicted_duration"]

# Colors
col_real_voice = "#1f77b4"
col_pred_voice = "#ff7f0e"
col_maneuver = "#2ca02c"

fig, ax = plt.subplots(figsize=(13, 4.8))

for idx, r in demo.iterrows():
    # Bar plots: note Y positions swapped (predicted now on top)
    ax.broken_barh([(r.maneuver_start, 5)], (0.2, 0.4), color=col_maneuver, alpha=0.8)
    ax.broken_barh([(r.voice_start, r.duration)], (2.6, 0.4), color=col_real_voice, alpha=0.9)
    ax.broken_barh([(r.pred_voice_start, r.predicted_duration)], (1.4, 0.4), color=col_pred_voice, alpha=0.9)

    # Offset brackets
    real_top = 3.25
    pred_top = 2.45

    # Real offset (black)
    ax.plot([r.voice_end, r.voice_end], [3.0, real_top], ls="--", color="black", lw=1)
    ax.plot([r.maneuver_start, r.maneuver_start], [0.6, real_top], ls="--", color="black", lw=1)
    ax.plot([r.voice_end, r.maneuver_start], [real_top, real_top], ls="--", color="black", lw=1)
    ax.text((r.voice_end + r.maneuver_start)/2, real_top + 0.03, f"{r.time_offset:.1f}s",
            ha='center', va='bottom', fontsize=12, color='black')

    # Predicted offset (grey)
    ax.plot([r.pred_voice_end, r.pred_voice_end], [1.8, pred_top], ls="--", color="black", lw=1)
    ax.plot([r.maneuver_start, r.maneuver_start], [0.6, pred_top], ls="--", color="black", lw=1)
    ax.plot([r.pred_voice_end, r.maneuver_start], [pred_top, pred_top], ls="--", color="black", lw=1)
    ax.text((r.pred_voice_end + r.maneuver_start)/2, pred_top + 0.03, f"{r.predicted_offset:.1f}s",
            ha='center', va='bottom', fontsize=12, color='black')

    # Duration labels
    ax.text(r.voice_start + r.duration / 2, 2.77, f"{r.duration:.1f}s", ha='center', va='center',
            fontsize=12, color='black')
    ax.text(r.pred_voice_start + r.predicted_duration / 2, 1.57, f"{r.predicted_duration:.1f}s", ha='center',
            va='center', fontsize=12, color='black')

    # Maneuver label + callsign next to triangle
    ax.text(r.maneuver_end - 3.8, 0.4, f"{r.callsign}\n{r.maneuvering_type} {r.maneuvering_parameter}",
            ha='left', va='center', fontsize=10)

    # Arrow triangle (pointing right, at end of bar)
    arrow = Polygon([[r.maneuver_end, 0.2],
                     [r.maneuver_end, 0.6],
                     [r.maneuver_end + 1, 0.4]],
                     closed=True, color=col_maneuver, alpha=0.8)
    ax.add_patch(arrow)

# Y-axis labels (order changed)
ax.set_yticks([1.6, 2.8, 0.4])
ax.set_yticklabels(["Est. ATCO", "Real ATCO", "Maneuver"], fontsize=10)
ax.set_xlabel("Relative time (s)")
ax.set_ylim(0, 3.7)
ax.grid(axis='x', ls=':', alpha=0.4)

# Legend
handles = [Line2D([0], [0], lw=6, color=c) for c in [col_real_voice, col_pred_voice, col_maneuver]]
ax.legend(handles, ["Real voice", "Predicted voice", "Maneuver"], loc='upper right')

plt.tight_layout()
out = "atco_lifecycle_single.png"
plt.savefig(out, dpi=300)
out
plt.show()