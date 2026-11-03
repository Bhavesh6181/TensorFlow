# Practical 3D – LSTM Autoencoder for NASA Turbofan

```python

# Practical 3D: NASA Turbofan Engine Fault Detection using LSTM Autoencoder

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# Step 2: Load dataset
df = pd.read_csv("nasa_turbofan.csv")

print(df.head())
print(df.shape)

# Step 3: Preprocess data
df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
df = df.ffill().fillna(0)

features = [f"sensor{i}" for i in range(1, 22)]

X = df[features].values
y = df["Fault_Label"].values

# Step 4: Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Step 5: Create time windows for each engine
window = 10

X_windows = []
y_windows = []

for unit in df["unit"].unique():
    idx = df["unit"] == unit

    data = X[idx]
    labels = y[idx]

    for i in range(len(data) - window + 1):
        X_windows.append(data[i:i + window])
        y_windows.append(labels[i + window - 1])

X = np.array(X_windows)
y = np.array(y_windows)

# Step 6: Select healthy windows for training
normal = y == 0
X_train = X[normal]

# Step 7: Build LSTM Autoencoder
input_layer = Input(shape=(window, len(features)))

encoded = LSTM(32, activation="relu")(input_layer)

decoded = RepeatVector(window)(encoded)
decoded = LSTM(32, activation="relu", return_sequences=True)(decoded)

output_layer = TimeDistributed(Dense(len(features)))(decoded)

model = Model(input_layer, output_layer)

# Step 8: Compile the model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Step 9: Train the model
history = model.fit(
    X_train, X_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Step 10: Calculate reconstruction error and threshold
predictions = model.predict(X, verbose=0)

train_errors = np.mean(
    (X[normal] - predictions[normal]) ** 2,
    axis=(1, 2)
)

threshold = np.mean(train_errors) + np.std(train_errors)

print("Fault Threshold:", threshold)

# Step 11: Detect faults
errors = np.mean((X - predictions) ** 2, axis=(1, 2))

faults = (errors > threshold).astype(int)

y_actual = (y == 1).astype(int)

# Step 12: Evaluate the model
print("Number of detected faults:", np.sum(faults))
print("Accuracy:", accuracy_score(y_actual, faults))

print("Confusion Matrix:")
print(confusion_matrix(y_actual, faults))

print("Classification Report:")
print(classification_report(y_actual, faults))

# Step 13: Plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
```
