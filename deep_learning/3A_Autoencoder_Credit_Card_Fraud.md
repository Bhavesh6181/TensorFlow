# Practical 3A – Autoencoder for Credit Card Fraud Detection

```python

# Practical 3A: Credit Card Fraud Detection using Autoencoder

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Step 2: Load dataset
df = pd.read_csv("creditcard.csv")

print(df.head())
print(df.shape)

# Step 3: Separate features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 4: Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Keep only normal transactions for training
x_train = x_train[y_train == 0]

# Step 5: Scale the data
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 6: Build Autoencoder
input_layer = Input(shape=(x_train.shape[1],))

# Encoder
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)

# Decoder
decoded = Dense(16, activation="relu")(encoded)
output_layer = Dense(x_train.shape[1], activation="sigmoid")(decoded)

model = Model(input_layer, output_layer)

# Step 7: Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Step 8: Train autoencoder on normal transactions
history = model.fit(
    x_train, x_train,
    epochs=20,
    batch_size=512,
    validation_split=0.1,
    shuffle=True
)

# Step 9: Calculate reconstruction error and threshold
predictions = model.predict(x_train, verbose=0)
train_errors = np.mean((x_train - predictions) ** 2, axis=1)

threshold = np.mean(train_errors) + np.std(train_errors)
print("Anomaly Threshold:", threshold)

# Step 10: Detect fraud transactions
predictions = model.predict(x_test, verbose=0)
errors = np.mean((x_test - predictions) ** 2, axis=1)

anomalies = (errors > threshold).astype(int)

print("Number of detected fraud transactions:", np.sum(anomalies))

# Step 11: Evaluate model
print("Accuracy:", accuracy_score(y_test, anomalies))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, anomalies))

print("\nClassification Report:")
print(classification_report(y_test, anomalies))

# Step 12: Plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
