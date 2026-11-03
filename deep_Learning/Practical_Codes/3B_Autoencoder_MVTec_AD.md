# Practical 3B – Autoencoder for MVTec AD

```python

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras import layers, models

# Step 2: Load dataset
# Each row contains ECG values followed by its class label
df = pd.read_csv("ecg.csv", header=None)

print(df.head())
print(df.shape)

# Step 3: Separate features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# Step 4: Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Keep only normal ECG records for training
x_train = x_train[y_train == 1]

# Step 6: Normalize the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 7: Build the Autoencoder
input_layer = layers.Input(shape=(x_train.shape[1],))

encoded = layers.Dense(32, activation="relu")(input_layer)
encoded = layers.Dense(16, activation="relu")(encoded)

decoded = layers.Dense(32, activation="relu")(encoded)
output_layer = layers.Dense(x_train.shape[1], activation="sigmoid")(decoded)

model = models.Model(input_layer, output_layer)

# Step 8: Compile the model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Step 9: Train the model using normal ECG records
history = model.fit(
    x_train, x_train,
    epochs=20,
    batch_size=512,
    validation_split=0.1,
    shuffle=True
)

# Step 10: Calculate reconstruction error and threshold
predictions = model.predict(x_train, verbose=0)

train_errors = np.mean((x_train - predictions) ** 2, axis=1)

threshold = np.mean(train_errors) + np.std(train_errors)

print("Anomaly Threshold:", threshold)

# Step 11: Detect anomalies in test data
predictions = model.predict(x_test, verbose=0)

errors = np.mean((x_test - predictions) ** 2, axis=1)

anomalies = (errors > threshold).astype(int)

# Actual labels: 0 = Normal, 1 = Abnormal
y_actual = (y_test != 1).astype(int)

# Step 12: Evaluate the model
print("Detected Anomalies:", np.sum(anomalies))

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
