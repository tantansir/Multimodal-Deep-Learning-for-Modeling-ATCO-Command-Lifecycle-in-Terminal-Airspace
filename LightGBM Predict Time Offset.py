from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load the data to inspect its structure
# data1和data2为lightgbm的训练数据
# data1保留了时间较大的time offset ；data2去除了50秒以上的time offset（CAS 60 10）；data3用了新的CAS检测方法(999：（CAS 60 60）),去除了50秒以上的time offset
data = pd.read_csv('data2.csv')
results_folder = "LightGBM_results3"
os.makedirs(results_folder, exist_ok=True)

# Preprocessing
# Extract input features and target variable
input_features = [
    'x', 'y', 'WTC', 'cas', 'heading', 'altitude', 'drct', 'sknt', 'skyl1',
    'distance_to_changi', 'bearing_to_changi', 'is_peakhour',
    'flight level', 'head', 'velocity', 'parameter'
]
target = 'time_offset'

# Fill missing values with median for simplicity
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

X = data[input_features]
y = data[target]

# Initialize the LightGBM model
model = LGBMRegressor(random_state=42, verbosity=-1) #参数调优在这里没什么提升,曾经设置过max_depth=7

# Set up KFold for 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Variables to store results
fold = 1
rmse_scores = []
mae_scores = []
mape_scores = []
feature_importances = np.zeros(len(input_features))
training_metrics = []
train_rmse_scores = []


# Cross-validation loop
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Collect evaluation history
    evals_result = {}

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),  # Early stopping
            log_evaluation(period=10)         # Log evaluation every 10 iterations
        ],
        eval_metric='rmse' #mae
    )

    # Collect training and validation RMSE
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_rmse_scores.append(train_rmse)

    # Predict and evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)

    # Accumulate feature importances
    feature_importances += model.feature_importances_

    # Save training metrics for visualization
    training_metrics.append({
        "fold": fold,
        "true_values": y_val,
        "predicted_values": y_pred
    })

    # print(f"Fold {fold}, RMSE: {rmse:.4f}")
    fold += 1

# Average feature importances
feature_importances /= kf.get_n_splits()

# Summary of results
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
mean_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)
mean_mape = np.mean(mape_scores)
std_mape = np.std(mape_scores)

print("\nCross-Validation Results:")
print(f"Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}")
print(f"Mean MAE: {mean_mae:.4f}, Std MAE: {std_mae:.4f}")
print(f"Mean MAPE: {mean_mape:.4f}, Std MAPE: {std_mape:.4f}")

# Visualization: training vs validation errors
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_rmse_scores) + 1), train_rmse_scores, label="Training RMSE", marker="o")
plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, label="Validation RMSE", marker="x")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.title("Training vs Validation RMSE Across Folds")
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_folder, "overall_train_vs_val_rmse.png"))
plt.close()

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(input_features, feature_importances, align='center')
plt.xlabel('Average Feature Importance')
plt.title('Feature Importance Across Folds')
feature_importance_path = os.path.join(results_folder, "feature_importance.png")
plt.savefig(feature_importance_path)
plt.close()

# Visualization: Prediction Effect as trends
for metric in training_metrics[:10]:
    plt.figure(figsize=(10, 6))
    plt.plot(metric["true_values"].index, metric["true_values"], label='True Values', marker='o', alpha=0.6)
    plt.plot(metric["true_values"].index, metric["predicted_values"], label='Predicted Values', linestyle='dashed', marker='x', alpha=0.6)
    plt.xlabel('Index')
    plt.ylabel('Time Offset')
    plt.title(f'Fold {metric["fold"]} - True vs Predicted (Trend)')
    plt.legend()
    prediction_effect_path = os.path.join(results_folder, f"trend_fold_{metric['fold']}.png")
    plt.savefig(prediction_effect_path)
    plt.close()


# Visualization: Cross-Validation Results
cv_metrics = {
    "Metric": ["Mean RMSE", "Std RMSE", "Mean MAE", "Std MAE"],
    "Value": [mean_rmse, std_rmse, mean_mae, std_mae]
}
plt.figure(figsize=(8, 6))
bars = plt.bar(cv_metrics["Metric"], cv_metrics["Value"], color='skyblue', alpha=0.8)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha='center', va='bottom')
plt.title("Cross-Validation Results")
plt.ylabel("Value")
plt.xlabel("Metric")
cv_results_path = os.path.join(results_folder, "overall_cv_results.png")
plt.savefig(cv_results_path)
plt.close()


# Evaluate on different event types
event_types = ['flight level', 'head', 'velocity']
event_results = {}


for event in event_types:
    print(f"\nEvaluating for: {event}")
    event_data = data[data[event] == 1]  # Filter data for current event type

    X_event = event_data[input_features]
    y_event = event_data[target]

    # Perform a train-test split for evaluation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    event_rmse_scores = []
    event_mae_scores = []
    train_rmse_scores = []
    train_mae_scores = []

    for train_idx, val_idx in kf.split(X_event):
        X_train, X_val = X_event.iloc[train_idx], X_event.iloc[val_idx]
        y_train, y_val = y_event.iloc[train_idx], y_event.iloc[val_idx]

        # Train the best model on the current event type data
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=10)
            ],
            eval_metric='rmse' #mae
        )
        # Calculate training error
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse_scores.append(train_rmse)
        train_mae_scores.append(train_mae)

        # Predict
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        event_rmse_scores.append(rmse)
        event_mae_scores.append(mae)

    # Store results
    event_results[event] = {
        'mean_rmse': np.mean(event_rmse_scores),
        'std_rmse': np.std(event_rmse_scores),
        'mean_mae': np.mean(event_mae_scores),
        'std_mae': np.std(event_mae_scores)
    }
    print(f"{event} - Mean RMSE: {np.mean(event_rmse_scores):.4f}, Std RMSE: {np.std(event_rmse_scores):.4f}")
    print(f"{event} - Mean MAE: {np.mean(event_mae_scores):.4f}, Std MAE: {np.std(event_mae_scores):.4f}")

    # Visualization: training vs validation errors for each event type
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse_scores) + 1), train_rmse_scores, label="Training RMSE", marker="o")
    plt.plot(range(1, len(event_rmse_scores) + 1), event_rmse_scores, label="Validation RMSE", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title(f"Train vs Validation RMSE - {event}")
    plt.legend()
    plt.grid()
    train_vs_val_path = os.path.join(results_folder, f"{event.replace(' ', '_')}_train_vs_val_rmse.png")
    plt.savefig(train_vs_val_path)
    plt.close()


# Visualization: Event Results
event_metrics = {
    "Event": [],
    "Metric": [],
    "Value": []
}
# Append metrics for each event
for event, metrics in event_results.items():
    for metric_name, value in metrics.items():
        event_metrics["Event"].append(event)
        event_metrics["Metric"].append(metric_name)
        event_metrics["Value"].append(value)

# Ensure the correct metric order
metric_order = ['mean_rmse', 'std_rmse', 'mean_mae', 'std_mae']
unique_metrics = metric_order  # Force correct order

unique_events = sorted(set(event_metrics["Event"]))
x = range(len(unique_metrics))

plt.figure(figsize=(16, 12))
for i, event in enumerate(unique_events):
    # Collect metrics in the correct order
    values = [
        event_metrics["Value"][j]
        for j in range(len(event_metrics["Metric"]))
        if event_metrics["Event"][j] == event and event_metrics["Metric"][j] in unique_metrics
    ]
    bars = plt.bar(
        [xi + i * 0.2 for xi in x], values, width=0.2, label=event, alpha=0.8
    )
    # Add text annotations
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha='center', va='bottom')

plt.title("Event Cross-Validation Results")
plt.xticks([xi + 0.3 for xi in x], unique_metrics, rotation=45)
plt.ylabel("Value")
plt.xlabel("Metric")
plt.legend()
event_results_path = os.path.join(results_folder, "event_cv_results.png")
plt.savefig(event_results_path)
plt.close()


# 使用整体均值作为预测值，输出rmse和mae
overall_mean = y.mean()
overall_predictions = np.full_like(y, overall_mean)
overall_rmse = np.sqrt(mean_squared_error(y, overall_predictions))
overall_mae = mean_absolute_error(y, overall_predictions)

overall_results = {
    'mean_rmse': overall_rmse,
    'mean_mae': overall_mae,
}

# 使用整体均值作为预测值，输出每个event的rmse和mae
event_results_baseline = {}
for event in event_types:
    event_data = data[data[event] == 1]
    if event_data.empty:
        continue

    event_mean = event_data[target].mean()
    event_predictions = np.full_like(event_data[target], event_mean)

    # Calculate RMSE and MAE for event-specific mean prediction
    event_rmse = np.sqrt(mean_squared_error(event_data[target], event_predictions))
    event_mae = mean_absolute_error(event_data[target], event_predictions)

    event_results_baseline[event] = {
        'mean_rmse': event_rmse,
        'mean_mae': event_mae,
    }

# Plot 1: Overall Baseline RMSE and MAE
plt.figure(figsize=(8, 6))
metrics = ['mean_rmse', 'mean_mae']
overall_values = [overall_results[metric] for metric in metrics]
bars = plt.bar(metrics, overall_values, color='skyblue', alpha=0.8)

# Add text annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha='center', va='bottom')

plt.title('Overall Baseline RMSE and MAE')
plt.ylabel('Value')
plt.grid(axis='y', linestyle='--', alpha=0.7)
overall_baseline_path = os.path.join(results_folder, "overall_baseline_metrics.png")
plt.savefig(overall_baseline_path)
plt.close()

# Plot 2: Overall and Event-specific Baseline Prediction Comparison
comparison_metrics = []
comparison_values = []
comparison_colors = []
lightgbm_overall_rmse = cv_metrics["Value"][0]  # Mean RMSE
lightgbm_overall_mae = cv_metrics["Value"][2]  # Mean MAE

# Add Overall metrics
comparison_metrics.extend(['Overall RMSE (Baseline)', 'Overall RMSE (LightGBM)',
                           'Overall MAE (Baseline)', 'Overall MAE (LightGBM)'])
comparison_values.extend([
    overall_results['mean_rmse'],  # Baseline RMSE
    lightgbm_overall_rmse,  # LightGBM RMSE
    overall_results['mean_mae'],  # Baseline MAE
    lightgbm_overall_mae   # LightGBM MAE
])
comparison_colors.extend(['orange', 'blue', 'orange', 'blue'])

# Add Event-specific metrics
for event, baseline_results in event_results_baseline.items():
    # Baseline metrics
    comparison_metrics.append(f"{event} RMSE (Baseline)")
    comparison_values.append(baseline_results['mean_rmse'])
    comparison_colors.append('orange')

    # LightGBM metrics (replace with actual LightGBM metrics if available)
    lightgbm_rmse = event_results[event]['mean_rmse']
    comparison_metrics.append(f"{event} RMSE (LightGBM)")
    comparison_values.append(lightgbm_rmse)
    comparison_colors.append('blue')

    comparison_metrics.append(f"{event} MAE (Baseline)")
    comparison_values.append(baseline_results['mean_mae'])
    comparison_colors.append('orange')

    lightgbm_mae = event_results[event]['mean_mae']
    comparison_metrics.append(f"{event} MAE (LightGBM)")
    comparison_values.append(lightgbm_mae)
    comparison_colors.append('blue')

# Correct the x-axis labels to retain all bars and remove "Baseline" and "LightGBM" text
comparison_metrics_clean = [
    metric.replace(" (Baseline)", "").replace(" (LightGBM)", "") + (" (B)" if "Baseline" in metric else " (L)")
    for metric in comparison_metrics
]

plt.figure(figsize=(24, 16))
bars = plt.bar(comparison_metrics_clean, comparison_values, color=comparison_colors, alpha=0.8)

# Add text annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha='center', va='bottom')

# Add legend
plt.legend(
    handles=[
        plt.Line2D([0], [0], color='orange', lw=8, label='Baseline'),
        plt.Line2D([0], [0], color='blue', lw=8, label='LightGBM')
    ],
    loc='upper right', title="Prediction Type", fontsize=12, title_fontsize=14
)

# Chart formatting
plt.title('Overall and Event-specific Baseline vs LightGBM Prediction Comparison', fontsize=16)
plt.ylabel('Value')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the updated plot with corrected x-axis labels
detailed_comparison_path = os.path.join(results_folder, "Overall and Event-specific Baseline vs LightGBM Prediction Comparison.png")
plt.savefig(detailed_comparison_path)
plt.close()
