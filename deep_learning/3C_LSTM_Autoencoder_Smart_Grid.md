# Practical 3C – LSTM Autoencoder for Smart Grid

```python

# Practical 3C: Energy Consumption Anomaly Detection using LSTM Autoencoder

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# Step 2: Load dataset
df = pd.read_csv("energy_consumption.csv")

print(df.head())
print(df.shape)

# Step 3: Preprocess data
df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
df = df.sort_values("Timestamp")
df = df.ffill().fillna(0)

features = [
    "Electricity_Consumed",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Avg_Past_Consumption"
]

X = df[features].values
y = df["Anomaly_Label"].values

# Step 4: Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Step 5: Create time windows
window = 10

X = np.array([
    X[i:i + window]
    for i in range(len(X) - window + 1)
])

# Step 6: Select normal windows for training
normal = y[window - 1:] == 0
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

train_errors = np.mean((X[normal] - predictions[normal]) ** 2, axis=(1, 2))

threshold = np.mean(train_errors) + np.std(train_errors)

print("Anomaly Threshold:", threshold)

# Step 11: Detect anomalies
errors = np.mean((X - predictions) ** 2, axis=(1, 2))

anomalies = (errors > threshold).astype(int)

y_actual = (y[window - 1:] == 1).astype(int)

# Step 12: Evaluate the model
print("Number of detected anomalies:", np.sum(anomalies))
print("Accuracy:", accuracy_score(y_actual, anomalies))

print("Confusion Matrix:")
print(confusion_matrix(y_actual, anomalies))

print("Classification Report:")
print(classification_report(y_actual, anomalies))

# Step 13: Plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
```
