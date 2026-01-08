# TensorFlow MLP + metric embeddings + time-respecting split
# WITH PER-METRIC STANDARDIZATION
#
# Uses:
# - metrics_34_metrics_14_weeks.csv
# - predicts week15
# - per-metric (row-wise) standardization BEFORE windowing

import numpy as np
import pandas as pd
import tensorflow as tf

# -----------------------
# 0) Load CSV
# -----------------------
csv_path = "metrics_34_metrics_14_weeks.csv"
df = pd.read_csv(csv_path)

df["metric"] = df["metric"].astype(str)
df = df.set_index("metric")

week_cols = sorted([c for c in df.columns if c.startswith("week")])
df = df[week_cols]

print("Loaded table:", df.shape)
print(df.head(), "\n")

num_metrics = df.shape[0]
weeks_train = df.shape[1]

# -----------------------
# 1) PER-METRIC STANDARDIZATION (KEY CHANGE)
# -----------------------
# Compute mean/std per metric (row-wise)
metric_means = df.mean(axis=1).to_numpy().reshape(-1, 1)
metric_stds = df.std(axis=1).to_numpy().reshape(-1, 1) + 1e-6

series = (df.to_numpy(dtype=np.float32) - metric_means) / metric_stds
# shape: (num_metrics, weeks_train)

# -----------------------
# 2) Build supervised windows + metric_id + time index
# -----------------------
k = 8  # lookback window

X_list, y_list, id_list, end_list = [], [], [], []

for metric_id in range(num_metrics):
    s = series[metric_id]
    for end in range(k, weeks_train):
        X_list.append(s[end - k:end])   # standardized lags
        y_list.append(s[end])           # standardized target
        id_list.append(metric_id)
        end_list.append(end)

X = np.array(X_list, dtype=np.float32)                 # (samples, k)
y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
metric_ids = np.array(id_list, dtype=np.int32).reshape(-1, 1)
end_idx = np.array(end_list, dtype=np.int32)

# -----------------------
# 3) Time-respecting (blocked) split
# -----------------------
val_weeks = 2
max_end = weeks_train - 1
val_start_end = max_end - (val_weeks - 1)

train_mask = end_idx < val_start_end
val_mask = end_idx >= val_start_end

X_train, y_train, id_train = X[train_mask], y[train_mask], metric_ids[train_mask]
X_val, y_val, id_val = X[val_mask], y[val_mask], metric_ids[val_mask]

print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
print(
    f"Validation target weeks: week{val_start_end+1:02d}..week{max_end+1:02d}\n"
)

# -----------------------
# 4) Build MLP + metric embedding
# -----------------------
tf.random.set_seed(42)

embedding_dim = 4

lag_input = tf.keras.Input(shape=(k,), name="lags")
id_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name="metric_id")

emb = tf.keras.layers.Embedding(
    input_dim=num_metrics,
    output_dim=embedding_dim,
    name="metric_embedding"
)(id_input)
emb = tf.keras.layers.Flatten()(emb)

x = tf.keras.layers.Concatenate()([lag_input, emb])
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
out = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=[lag_input, id_input], outputs=out)
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
    {"lags": X_train, "metric_id": id_train},
    y_train,
    validation_data=({"lags": X_val, "metric_id": id_val}, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

print("Training complete.\n")

# -----------------------
# 5) Predict week15 (standardized → invert scale)
# -----------------------
last_k = series[:, -k:]  # standardized last k weeks
pred_ids = np.arange(num_metrics, dtype=np.int32).reshape(-1, 1)

pred_week15_std = model.predict(
    {"lags": last_k, "metric_id": pred_ids},
    verbose=0
).reshape(-1)

# Invert per-metric standardization
pred_week15 = pred_week15_std * metric_stds.flatten() + metric_means.flatten()

df_with_pred = df.copy()
df_with_pred["week15_pred"] = pred_week15

print("Table with predicted week15:")
print(df_with_pred.head())

df_with_pred.to_csv("metrics_with_week15_pred.csv")
print("\nSaved: metrics_with_week15_pred.csv")
