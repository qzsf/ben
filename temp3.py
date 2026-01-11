# TRUE GLOBAL MULTI-OUTPUT MODEL
# ---------------------------------------------
# - Inputs: last k weeks of ALL 19 metrics + date features
# - Outputs: next week for ALL 19 metrics
# - k = 6
# - per-metric standardization using StandardScaler
# - time-respecting validation split
# - 1-step forecast

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# -----------------------
# 0) Load CSV
# -----------------------
csv_path = "metrics_19_metrics_60_weeks.csv"  # <-- your file
df = pd.read_csv(csv_path)
df = df.set_index("metric")

# Parse dates from columns
dates = pd.to_datetime(df.columns, format="%m/%d/%y")

num_metrics = df.shape[0]   # 19
num_weeks = df.shape[1]     # ~60+

print("Data shape:", df.shape)
print("Date range:", dates.min().date(), "→", dates.max().date())

# -----------------------
# 1) DATE FEATURES (cyclical)
# -----------------------
week_of_year = dates.isocalendar().week.astype(int).to_numpy()
month = dates.month.to_numpy()

date_features = np.column_stack([
    np.sin(2 * np.pi * week_of_year / 52),
    np.cos(2 * np.pi * week_of_year / 52),
    np.sin(2 * np.pi * month / 12),
    np.cos(2 * np.pi * month / 12),
]).astype(np.float32)

num_date_features = date_features.shape[1]

# -----------------------
# 2) PER-METRIC STANDARDIZATION (StandardScaler)
# -----------------------
# Scale each metric independently (row-wise)
scaler = StandardScaler()
series_scaled = scaler.fit_transform(df.to_numpy().T).T  # transpose trick

metric_means = scaler.mean_
metric_stds = np.sqrt(scaler.var_) + 1e-6

# -----------------------
# 3) Build supervised samples (MULTI-OUTPUT)
# -----------------------
k = 6

X_list, y_list, end_list = [], [], []

for end in range(k, num_weeks):
    # last k weeks of ALL metrics
    lag_block = series_scaled[:, end - k:end]   # (metrics, k)
    lag_block = lag_block.flatten()              # (metrics * k,)

    date_part = date_features[end]               # date of target week

    X_list.append(np.concatenate([lag_block, date_part]))
    y_list.append(series_scaled[:, end])         # next-week ALL metrics
    end_list.append(end)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
end_idx = np.array(end_list)

print("Supervised X shape:", X.shape)
print("Supervised y shape:", y.shape)

# -----------------------
# 4) Time-respecting split
# -----------------------
val_weeks = 4  # last 4 weeks for validation

max_end = num_weeks - 1
val_start = max_end - (val_weeks - 1)

train_mask = end_idx < val_start
val_mask = end_idx >= val_start

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

# -----------------------
# 5) Build MULTI-OUTPUT MLP
# -----------------------
tf.random.set_seed(42)

input_dim = num_metrics * k + num_date_features

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_metrics)   # <-- 19 outputs
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.Huber(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0
)

print("Training complete.")

# -----------------------
# 6) Predict NEXT WEEK
# -----------------------
last_k = series_scaled[:, -k:]           # (metrics, k)
last_k_flat = last_k.flatten()

next_date = dates.max() + pd.Timedelta(days=7)

# Build date features for next week
woy = int(next_date.isocalendar().week)
m = next_date.month

next_date_feat = np.array([
    np.sin(2 * np.pi * woy / 52),
    np.cos(2 * np.pi * woy / 52),
    np.sin(2 * np.pi * m / 12),
    np.cos(2 * np.pi * m / 12),
], dtype=np.float32)

X_next = np.concatenate([last_k_flat, next_date_feat]).reshape(1, -1)

pred_std = model.predict(X_next, verbose=0).flatten()

# Invert standardization
pred = pred_std * metric_stds + metric_means

# -----------------------
# 7) Output results
# -----------------------
pred_col = next_date.strftime("%m/%d/%y")

df_out = df.copy()
df_out[pred_col] = pred

print("Prediction added for:", pred_col)
print(df_out.head())

df_out.to_csv("metrics_with_next_week_prediction.csv")
print("Saved: metrics_with_next_week_prediction.csv")
