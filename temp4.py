# SHARED TRUNK + 19 HEADS (END-TO-END PIPELINE)
# ---------------------------------------------
# - Inputs: last k weeks of ALL 19 metrics + date features
# - Outputs: 19 separate heads (one per metric) predicting next week
# - k = 6
# - per-metric standardization using StandardScaler (row-wise via transpose)
# - time-respecting (blocked) validation split
# - 1-step forecast
#
# CSV format expected:
# metric,10/04/24,10/11/24,...  (weekly Fridays as column headers)
# metric01,0.78,0.81,...
# ...
# metric19,234000,231000,...
#
# Output:
# - metrics_with_next_week_prediction_shared_heads.csv

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# -----------------------
# 0) Load CSV
# -----------------------
csv_path = "metrics_19_metrics_60_weeks.csv"  # <-- update to your filename
df = pd.read_csv(csv_path)
df["metric"] = df["metric"].astype(str)
df = df.set_index("metric")

# Parse dates from columns (headers)
dates = pd.to_datetime(df.columns, format="%m/%d/%y", errors="raise")

num_metrics = df.shape[0]   # should be 19
num_weeks = df.shape[1]     # should be 60+

print("Data shape:", df.shape)
print("Date range:", dates.min().date(), "→", dates.max().date())

# -----------------------
# 1) DATE FEATURES (cyclical)
# -----------------------
week_of_year = dates.isocalendar().week.astype(int).to_numpy()
month = dates.month.to_numpy()

date_features = np.column_stack([
    np.sin(2 * np.pi * week_of_year / 52.0),
    np.cos(2 * np.pi * week_of_year / 52.0),
    np.sin(2 * np.pi * month / 12.0),
    np.cos(2 * np.pi * month / 12.0),
]).astype(np.float32)

num_date_features = date_features.shape[1]  # 4

# -----------------------
# 2) PER-METRIC STANDARDIZATION (StandardScaler, row-wise)
# -----------------------
scaler = StandardScaler()
series_scaled = scaler.fit_transform(df.to_numpy(dtype=np.float32).T).T  # (metrics, weeks)

metric_means = scaler.mean_                      # (metrics,)
metric_stds = np.sqrt(scaler.var_) + 1e-6        # (metrics,)

# -----------------------
# 3) Build supervised samples (shared input, 19 targets)
# -----------------------
k = 6
if num_weeks <= k:
    raise ValueError(f"Need more weeks than k. Got num_weeks={num_weeks}, k={k}")

X_list, y_list, end_list = [], [], []

for end in range(k, num_weeks):
    # Input = last k weeks for ALL metrics, flattened
    lag_block = series_scaled[:, end - k:end]  # (metrics, k)
    lag_flat = lag_block.reshape(-1)           # (metrics*k,)

    # Date features for the TARGET week (the week we're predicting)
    date_part = date_features[end]             # (4,)

    X_list.append(np.concatenate([lag_flat, date_part], axis=0))
    y_list.append(series_scaled[:, end])       # (metrics,)
    end_list.append(end)

X = np.array(X_list, dtype=np.float32)         # (samples, metrics*k + 4)
y = np.array(y_list, dtype=np.float32)         # (samples, metrics)
end_idx = np.array(end_list, dtype=np.int32)   # (samples,)

print("Supervised X shape:", X.shape)
print("Supervised y shape:", y.shape)

# -----------------------
# 4) Time-respecting split
# -----------------------
val_weeks = 4  # last 4 target weeks for validation
max_end = num_weeks - 1
val_start = max_end - (val_weeks - 1)

train_mask = end_idx < val_start
val_mask = end_idx >= val_start

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
print(f"Validation target dates: {dates[val_start].date()} .. {dates[max_end].date()}")

# Keras multi-output expects a list of targets, one per head
y_train_list = [y_train[:, i] for i in range(num_metrics)]
y_val_list = [y_val[:, i] for i in range(num_metrics)]

# -----------------------
# 5) Build SHARED TRUNK + 19 HEADS model
# -----------------------
tf.random.set_seed(42)

input_dim = num_metrics * k + num_date_features

inputs = tf.keras.Input(shape=(input_dim,), name="X")

x = tf.keras.layers.Dense(128, activation="relu")(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(x)

outputs = [
    tf.keras.layers.Dense(1, name=f"metric_{i+1:02d}")(x)
    for i in range(num_metrics)
]

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Same loss for each head; Keras will average them unless you set loss_weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.Huber(),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train_list,
    validation_data=(X_val, y_val_list),
    epochs=300,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0
)

print("Training complete.")

# -----------------------
# 6) Predict NEXT WEEK
# -----------------------
last_k = series_scaled[:, -k:]      # (metrics, k)
last_k_flat = last_k.reshape(-1)    # (metrics*k,)

next_date = dates.max() + pd.Timedelta(days=7)

# Build date features for next week
woy = int(next_date.isocalendar().week)
m = int(next_date.month)

next_date_feat = np.array([
    np.sin(2 * np.pi * woy / 52.0),
    np.cos(2 * np.pi * woy / 52.0),
    np.sin(2 * np.pi * m / 12.0),
    np.cos(2 * np.pi * m / 12.0),
], dtype=np.float32)

X_next = np.concatenate([last_k_flat, next_date_feat], axis=0).reshape(1, -1).astype(np.float32)

pred_list = model.predict(X_next, verbose=0)  # list of 19 arrays, each shape (1,1)

# Convert list-of-heads -> (19,)
pred_std = np.array([p[0, 0] for p in pred_list], dtype=np.float32)

# Invert standardization: back to original scale
pred = pred_std * metric_stds + metric_means

# -----------------------
# 7) Output results
# -----------------------
pred_col = next_date.strftime("%m/%d/%y")

df_out = df.copy()
df_out[pred_col] = pred

print("Prediction added for:", pred_col)
print(df_out.head())

out_path = "metrics_with_next_week_prediction_shared_heads.csv"
df_out.to_csv(out_path)
print("Saved:", out_path)
