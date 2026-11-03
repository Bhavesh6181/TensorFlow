# Practical 3F – Autoencoder for Sign Language MNIST

```python

# Practical 3F: Sign Language MNIST Anomaly Detection using Autoencoder

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Step 2: Load dataset
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

print(train_df.head())
print(train_df.shape)

# Step 3: Separate features and labels
X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values

X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

# Step 4: Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Keep only the remaining 23 classes for training
X_train = X_train[y_train != 24]

# Step 5: Build Autoencoder
input_layer = Input(shape=(X_train.shape[1],))

# Encoder
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)

# Decoder
decoded = Dense(16, activation="relu")(encoded)
output_layer = Dense(X_train.shape[1], activation="sigmoid")(decoded)

model = Model(input_layer, output_layer)

# Step 6: Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Step 7: Train autoencoder on normal classes
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=512,
    validation_split=0.1,
    shuffle=True
)

# Step 8: Calculate reconstruction error and threshold
predictions = model.predict(X_train, verbose=0)
train_errors = np.mean((X_train - predictions) ** 2, axis=1)

threshold = np.mean(train_errors) + np.std(train_errors)

print("Anomaly Threshold:", threshold)

# Step 9: Detect anomalies in test data
predictions = model.predict(X_test, verbose=0)
errors = np.mean((X_test - predictions) ** 2, axis=1)

anomalies = (errors > threshold).astype(int)
print("Number of detected anomalies:", np.sum(anomalies))

# Actual labels: 0 = Normal, 1 = Anomaly
y_actual = (y_test == 24).astype(int)


# Step 10: Evaluate model
print("Accuracy:", accuracy_score(y_actual, anomalies))

print("\nConfusion Matrix:")
print(confusion_matrix(y_actual, anomalies))

print("\nClassification Report:")
print(classification_report(y_actual, anomalies))

# Step 11: Plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
