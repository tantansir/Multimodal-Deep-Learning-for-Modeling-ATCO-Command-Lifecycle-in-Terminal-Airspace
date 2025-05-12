import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import torch
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import KBinsDiscretizer

# ========= 1. 读取和预处理数据 =========
df = pd.read_csv("Track_used_for_train.csv")

features = [
    'x', 'y', 'parameter', 'maneuvering_parameter',
    'WTC', 'cas', 'heading', 'altitude', 'drct', 'sknt', 'skyl1',
    'flight level', 'head', 'velocity',
    'distance_to_changi', 'bearing_to_changi',
    'is_peakhour', 'num_other_plane',
    'is_planroute', 'nearest_wp_dist_km'
]

targets = ['time_offset', 'duration']

df = df.dropna(subset=features + targets)
df["is_planroute"] = df["is_planroute"].astype(float)

X = df[features]
y = df[targets].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========= 2. 模型训练函数 =========
def train_and_predict_single_target(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)

# ========= 3. LightGBM 和 XGBoost =========
lgb_model_offset = lgb.LGBMRegressor(random_state=42)
lgb_model_duration = lgb.LGBMRegressor(random_state=42)
xgb_model_offset = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
xgb_model_duration = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

lgb_pred_offset = train_and_predict_single_target(lgb_model_offset, X_train, y_train[:, 0], X_test)
lgb_pred_duration = train_and_predict_single_target(lgb_model_duration, X_train, y_train[:, 1], X_test)
xgb_pred_offset = train_and_predict_single_target(xgb_model_offset, X_train, y_train[:, 0], X_test)
xgb_pred_duration = train_and_predict_single_target(xgb_model_duration, X_train, y_train[:, 1], X_test)

lgb_pred = np.vstack([lgb_pred_offset, lgb_pred_duration]).T
xgb_pred = np.vstack([xgb_pred_offset, xgb_pred_duration]).T

# ========= 4. TabPFN =========
def run_tabpfn_regression(X_train, y_train_target, X_test):
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    y_train_class = discretizer.fit_transform(y_train_target.reshape(-1, 1)).ravel().astype(int)

    model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
    model.fit(X_train, y_train_class)

    y_pred_class = model.predict(X_test).astype(int)
    bin_edges = discretizer.bin_edges_[0]

    # 防止越界
    y_pred_class = np.clip(y_pred_class, 0, len(bin_edges) - 2)

    y_pred_reg = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in y_pred_class])
    return y_pred_reg



tabpfn_pred_offset = run_tabpfn_regression(X_train, y_train[:, 0], X_test)
tabpfn_pred_duration = run_tabpfn_regression(X_train, y_train[:, 1], X_test)
tabpfn_pred = np.vstack([tabpfn_pred_offset, tabpfn_pred_duration]).T

# ========= 5. 指标评估 =========
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAE_offset": mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
        "MAE_duration": mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
        "RMSE_offset": np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])),
        "RMSE_duration": np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])),
        "R2_offset": r2_score(y_true[:, 0], y_pred[:, 0]),
        "R2_duration": r2_score(y_true[:, 1], y_pred[:, 1]),
    }

results = [
    evaluate_model("LightGBM", y_test, lgb_pred),
    evaluate_model("XGBoost", y_test, xgb_pred),
    evaluate_model("TabPFN", y_test, tabpfn_pred),
]

# ========= 6. 输出结果 =========
results_df = pd.DataFrame(results)
print("\n=========== 模型评估结果对比 ===========")
print(results_df.to_string(index=False))
