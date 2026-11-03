# Practical 3E – Autoencoder for NSL-KDD

```python
# Practical 3E: Network Intrusion Detection using Autoencoder

# Step 1: Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# Step 2: Define column names

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


# Step 3: Load dataset

train = pd.read_csv("KDDTrain+.TXT", names=columns)
test = pd.read_csv("KDDTest+.TXT", names=columns)

print(train.head())
print(train.shape)


# Step 4: Separate features and labels

X_test = test.drop(["label", "difficulty"], axis=1)

y_test = test["label"].str.strip().str.rstrip(".")

y_test = (y_test != "normal").astype(int)


normal_data = train[
    train["label"].str.strip().str.rstrip(".") == "normal"
]

X_normal = normal_data.drop(["label", "difficulty"], axis=1)


# Step 5: Split normal data into training and validation

X_train, X_val = train_test_split(
    X_normal, test_size=0.20, random_state=42
)


# Step 6: Encode categorical features and scale numerical features

categorical = ["protocol_type", "service", "flag"]

numerical = [
    col for col in X_train.columns
    if col not in categorical
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), categorical)
])

X_train = preprocessor.fit_transform(X_train)

X_val = preprocessor.transform(X_val)

X_test = preprocessor.transform(X_test)


# Step 7: Build Autoencoder using Functional API

input_layer = Input(shape=(X_train.shape[1],))

encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(16, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(decoded)

output_layer = Dense(
    X_train.shape[1], activation="linear"
)(decoded)

model = Model(input_layer, output_layer)


# Step 8: Compile model

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# Step 9: Train model

history = model.fit(
    X_train, X_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, X_val)
)


# Step 10: Calculate validation reconstruction error

val_predictions = model.predict(X_val)

val_errors = np.mean(
    (X_val - val_predictions) ** 2,
    axis=1
)


# Step 11: Set threshold

threshold = np.percentile(val_errors, 95)

print("Threshold:", threshold)


# Step 12: Calculate test reconstruction error

test_predictions = model.predict(X_test)

test_errors = np.mean(
    (X_test - test_predictions) ** 2,
    axis=1
)


# Step 13: Detect intrusions and evaluate model

y_pred = (test_errors > threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0
))


# Step 14: Plot training and validation loss

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()


# Step 15: Plot reconstruction errors

plt.plot(test_errors, label="Reconstruction Error")

plt.axhline(
    threshold, color="red",
    linestyle="--", label="Threshold"
)

plt.title("Network Intrusion Detection")
plt.xlabel("Test Sample")
plt.ylabel("Reconstruction Error")

plt.legend()
plt.show()
```
