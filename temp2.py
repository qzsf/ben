# End-to-end pipeline: MLP + metric embeddings + per-metric standardization
# + DATE (calendar) features from weekly Friday dates
#
# CSV format expected:
# metric,10/04/24,10/11/24,10/18/24,...,01/03/25
# metric01,0.78,0.81,...
# ...
#
# What this script does:
# 1) loads CSV (date columns)
# 2) parses dates + builds cyclical calendar features (week-of-year, month)
# 3) per-metric standardizes values (row-wise)
# 4) builds supervised windows: [k lags + date_features(target_week)] -> target
# 5) time-respecting (blocked) validation split
# 6) trains MLP with metric embeddings
# 7) predicts NEXT week (week15) using next Friday date features
# 8) inverts standardization back to original scale and saves output CSV

import numpy as np
import pandas as pd
import tensorflow as tf

# -----------------------
# Config
# -----------------------
csv_path = "metrics_34_metrics_14_weeks.csv"   # <-- your modified file with date columns
k = 8                                         # lookback window length
val_weeks = 2                                 # last N target-weeks used for validation
embedding_dim = 4
seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)

# -----------------------
# 0) Load CSV
# -----------------------
df = pd.read_csv(csv_path)
df["metric"] = df["metric"].astype(str)
df = df.set_index("metric")

# Parse the date columns (your columns are like 10/4/24, 10/11/24, ...)
# If your CSV uses 10/4/2024 (4-digit year), change format accordingly.
date_cols = list(df.columns)
dates = pd.to_datetime(date_cols, format="%m/%d/%y", errors="raise")

# Ensure columns are sorted by date (important)
sorted_idx = np.argsort(dates.values)
dates = dates[sorted_idx]
df = df.iloc[:, sorted_idx]

print("Loaded table shape (metrics x weeks):", df.shape)
print("First 3 dates:", dates[:3].strftime("%m/%d/%y").tolist())
print("Last 3 dates:", dates[-3:].strftime("%m/%d/%y").tolist(), "\n")
print(df.head(), "\n")

num_metrics = df.shape[0]
weeks_train = df.shape[1]
if weeks_train <= k:
    raise ValueError(f"Need more weeks than k. Got weeks_train={weeks_train}, k={k}")

# -----------------------
# 1) Build date features (cyclical encodings)
# -----------------------
# For weekly Friday data, the useful “calendar signal” usually comes from
# week-of-year and month. We encode them cyclically using sin/cos.
iso_week = dates.isocalendar().week.astype(int).to_numpy()  # 1..53
month = dates.month.to_numpy()                              # 1..12

date_feat = np.column_stack([
    np.sin(2 * np.pi * iso_week / 52.0),
    np.cos(2 * np.pi * iso_week / 52.0),
    np.sin(2 * np.pi * month / 12.0),
    np.cos(2 * np.pi * month / 12.0),
]).astype(np.float32)

num_date_features = date_feat.shape[1]
print("Date feature matrix shape:", date_feat.shape, "(weeks x features)\n")

# Helper to create date features for an arbitrary future date (e.g., next Friday)
def make_date_features(dt: pd.Timestamp) -> np.ndarray:
    iw = int(dt.isocalendar().week)
    m = int(dt.month)
    return np.array([
        np.sin(2 * np.pi * iw / 52.0),
        np.cos(2 * np.pi * iw / 52.0),
        np.sin(2 * np.pi * m / 12.0),
        np.cos(2 * np.pi * m / 12.0),
    ], dtype=np.float32)

# -----------------------
# 2) Per-metric standardization (row-wise)
# -----------------------
metric_means = df.mean(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1)
metric_stds = (df.std(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1) + 1e-6)

series_std = (df.to_numpy(dtype=np.float32) - metric_means) / metric_stds
# shape: (num_metrics, weeks_train)

# -----------------------
# 3) Build supervised windows: [lags + date_features(target_week)] -> target
# -----------------------
X_list, y_list, id_list, end_list = [], [], [], []

# end is the target week index (0-based), where target is series_std[:, end]
# lags are series_std[:, end-k : end]
for metric_id in range(num_metrics):
    s = series_std[metric_id]
    for end in range(k, weeks_train):
        lag_part = s[end - k:end]        # (k,)
        date_part = date_feat[end]       # date features of the TARGET week (end)
        X_list.append(np.concatenate([lag_part, date_part], axis=0))  # (k + F,)
        y_list.append(s[end])            # standardized target scalar
        id_list.append(metric_id)        # metric identity
        end_list.append(end)             # time index of target (for time split)

X = np.array(X_list, dtype=np.float32)                        # (samples, k+F)
y = np.array(y_list, dtype=np.float32).reshape(-1, 1)         # (samples, 1)
metric_ids = np.array(id_list, dtype=np.int32).reshape(-1, 1) # (samples, 1)
end_idx = np.array(end_list, dtype=np.int32)                  # (samples,)

# -----------------------
# 4) Time-respecting split (blocked): last val_weeks target weeks as validation
# -----------------------
max_end = weeks_train - 1
val_start_end = max_end - (val_weeks - 1)

train_mask = end_idx < val_start_end
val_mask = end_idx >= val_start_end

X_train, y_train, id_train = X[train_mask], y[train_mask], metric_ids[train_mask]
X_val, y_val, id_val = X[val_mask], y[val_mask], metric_ids[val_mask]

print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
print(
    f"Validation target weeks: index {val_start_end}..{max_end} "
    f"(dates {dates[val_start_end].strftime('%m/%d/%y')}..{dates[max_end].strftime('%m/%d/%y')})\n"
)

# -----------------------
# 5) Build MLP + metric embedding model
# -----------------------
input_dim = k + num_date_features

lag_plus_date_input = tf.keras.Input(shape=(input_dim,), name="lags_plus_date")
id_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name="metric_id")

emb = tf.keras.layers.Embedding(
    input_dim=num_metrics,
    output_dim=embedding_dim,
    name="metric_embedding"
)(id_input)
emb = tf.keras.layers.Flatten()(emb)

x = tf.keras.layers.Concatenate()([lag_plus_date_input, emb])
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
out = tf.keras.layers.Dense(1, name="next_week_std")(x)

model = tf.keras.Model(inputs=[lag_plus_date_input, id_input], outputs=out)
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

history = model.fit(
    {"lags_plus_date": X_train, "metric_id": id_train},
    y_train,
    validation_data=({"lags_plus_date": X_val, "metric_id": id_val}, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

best_val_mae = float(np.min(history.history["val_mae"]))
print(f"Training complete. Best val MAE (standardized space): {best_val_mae:.4f}\n")

# -----------------------
# 6) Predict NEXT week (week15) for each metric
# -----------------------
# Next week date = last date + 7 days (since they are consecutive Fridays)
next_date = dates[-1] + pd.Timedelta(days=7)
next_date_feat = make_date_features(next_date)                # (F,)
next_date_feat_batch = np.repeat(next_date_feat.reshape(1, -1), num_metrics, axis=0)  # (M,F)

# last k standardized values per metric
last_k_std = series_std[:, -k:]                               # (M,k)

# Build prediction inputs: [last_k + next_date_features]
X_pred = np.concatenate([last_k_std, next_date_feat_batch], axis=1).astype(np.float32)  # (M,k+F)
pred_ids = np.arange(num_metrics, dtype=np.int32).reshape(-1, 1)

pred_next_std = model.predict(
    {"lags_plus_date": X_pred, "metric_id": pred_ids},
    verbose=0
).reshape(-1)  # standardized predictions

# Invert per-metric standardization back to original scale
pred_next = pred_next_std * metric_stds.flatten() + metric_means.flatten()

# Name the prediction column by its date (clearer than "week15_pred")
pred_col_name = next_date.strftime("%m/%d/%y") + "_pred"

df_with_pred = df.copy()
df_with_pred[pred_col_name] = pred_next

print("Next prediction date:", next_date.strftime("%m/%d/%y"))
print("Table with next-week prediction appended:")
print(df_with_pred.head(), "\n")

out_path = "metrics_with_next_week_pred.csv"
df_with_pred.to_csv(out_path)
print("Saved:", out_path)
