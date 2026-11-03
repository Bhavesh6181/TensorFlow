# Practical 3C – LSTM Autoencoder for Smart Grid

## 1. Objective
The goal is to detect energy consumption anomalies (such as sudden power surges, equipment failures, or meter tampering) in Smart Grid systems using an LSTM Autoencoder on multivariate time-series data.

## 2. Dataset
- **Name:** Smart Grid Energy Consumption dataset (`energy_consumption.csv`)
- **Features (5):** `Electricity_Consumed`, `Temperature`, `Humidity`, `Wind_Speed`, `Avg_Past_Consumption`
- **Metadata:** `Timestamp` column used for chronological time ordering
- **Target:** `Anomaly_Label` where `0 = Normal consumption` and `1 = Anomaly`

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For time-series parsing, sorting by timestamp, and forward-filling missing values. |
| `numpy` | For creating sliding temporal windows and computing reconstruction error matrices. |
| `matplotlib.pyplot` | For graphing training and validation loss trends over epochs. |
| `MinMaxScaler` (sklearn) | For scaling sensor features into `[0, 1]`. |
| `accuracy_score`, `confusion_matrix`, `classification_report` (sklearn) | For evaluating anomaly classification performance against ground truth labels. |
| `Input`, `LSTM`, `RepeatVector`, `TimeDistributed`, `Dense` (tensorflow.keras) | Specialized layers for building sequence-to-sequence LSTM autoencoders. |

## 4. Important Functions and Code
- `df["Timestamp"] = pd.to_datetime(...)` & `df.sort_values("Timestamp")`: Chronologically sorts data to preserve temporal order.
- `df.ffill().fillna(0)`: Handles missing values by carrying the previous valid observation forward.
- Sliding Window Creation: Uses a loop to convert 2D data into 3D tensors of shape `(samples, window_size=10, features=5)`.
- `X_train = X[normal]`: Trains the LSTM Autoencoder **only** on windows that contain entirely normal behavior (`Anomaly_Label == 0`).
- `LSTM(32, activation="relu")(input_layer)`: **Encoder** LSTM reads the 10 time steps and compresses them into a single 32-dimensional context vector.
- `RepeatVector(window)`: Replicates the 32-dimensional context vector 10 times to provide input for each decoded time step.
- `LSTM(32, activation="relu", return_sequences=True)`: **Decoder** LSTM reconstructs temporal sequences step by step.
- `TimeDistributed(Dense(len(features)))`: Applies a Dense layer independently to every time step to reconstruct the 5 original feature values.
- `np.mean((X - predictions) ** 2, axis=(1, 2))`: Computes MSE reconstruction error across both time steps (axis 1) and features (axis 2).

## 5. Important Parameters
- **Window Size (10):** Looks at 10 consecutive time steps per sequence window.
- **Features (5):** Electricity, temperature, humidity, wind speed, past consumption.
- **Latent Units (32):** Number of hidden units in encoder and decoder LSTMs.
- **Epochs (10):** Training passes.
- **Batch Size (32):** Mini-batch size.
- **Optimizer (Adam) & Loss (MSE):** Standard choice for sequence reconstruction tasks.
- **Threshold:** `Mean + 1 Standard Deviation` of normal training window errors.

## 6. Model Architecture
- **Input:** `(10, 5)` tensor representing 10 consecutive readings across 5 features.
- **Encoder:** `LSTM(32, ReLU)` -> Outputs a 1D vector of length 32.
- **Bridge:** `RepeatVector(10)` -> Replicates vector across 10 time steps -> `(10, 32)`.
- **Decoder:** `LSTM(32, ReLU, return_sequences=True)` -> `(10, 32)`.
- **Output Layer:** `TimeDistributed(Dense(5))` -> Reconstructs original shape `(10, 5)`.
- **Data Flow:** `(10, 5) -> LSTM(32) -> RepeatVector(10) -> LSTM(32, seq) -> TimeDistributed(Dense(5))`

## 7. Preprocessing Steps
- **Datetime Sorting:** Ensures observations are strictly chronological; time series must never be randomly shuffled before windowing.
- **Missing Value Handling (`ffill`):** Fills sensor dropouts with the most recent valid sensor measurement.
- **Min-Max Normalization:** Scales all 5 features to `[0, 1]`.
- **Sliding Window Transformation:** Converts flat 2D tabular data into 3D sequential blocks: `(samples, time_steps=10, features=5)`.

## 8. Expected Output
- Initial dataset inspection (head and shape).
- 10 epochs of training progress with decreasing loss.
- Anomaly threshold printed.
- Number of detected smart grid anomalies.
- Accuracy score, Confusion Matrix, and Classification Report.
- Training vs. Validation Loss plot.

## 9. Viva Questions and Answers
**Q1. Why do we use an LSTM Autoencoder instead of a standard Dense Autoencoder here?**
**Ans:** Smart grid data is time-dependent. Standard autoencoders treat each sample independently, whereas LSTM autoencoders capture temporal dependencies and sequential trends across time steps.

**Q2. What is the role of `RepeatVector` in an LSTM Autoencoder?**
**Ans:** The encoder LSTM summarizes the entire 10-step sequence into a single 1D vector. `RepeatVector(10)` replicates this vector 10 times to provide input for the decoder LSTM at each time step.

**Q3. What does `TimeDistributed(Dense(5))` do?**
**Ans:** It applies the same Dense layer with 5 outputs independently to each of the 10 time steps, reconstructing the original 5 features at every point in time.

**Q4. What is a sliding window?**
**Ans:** A technique that groups consecutive observations (here 10 steps) into overlapping sequence blocks to serve as inputs for recurrent networks.

**Q5. Why is `axis=(1, 2)` used when computing reconstruction error?**
**Ans:** Because the predictions are 3D `(samples, time_steps, features)`. Taking the mean over axes 1 and 2 averages errors across all time steps and features, producing a single error scalar per sample.

**Q6. What happens if an anomaly occurs in the smart grid?**
**Ans:** The unusual pattern deviates from the normal consumption dynamics the LSTM learned, causing high reconstruction error that exceeds the threshold.

**Q7. Why must time-series data be sorted before windowing?**
**Ans:** Shuffling time-series data breaks the temporal order, making it impossible to model temporal patterns.

**Q8. What are the 5 features in this dataset?**
**Ans:** Electricity_Consumed, Temperature, Humidity, Wind_Speed, and Avg_Past_Consumption.

**Q9. What activation function is used inside the LSTM layers?**
**Ans:** ReLU activation (`activation="relu"`).

**Q10. Why is the model trained only on normal windows?**
**Ans:** To ensure it only learns normal consumption patterns and fails to reconstruct abnormal surges or faults.

## 10. Quick Revision Flow
`Sort by Timestamp -> Forward-fill NaNs -> Normalize 5 features with MinMaxScaler -> Create 10-step sliding windows -> Filter normal windows -> Build LSTM Autoencoder (Input(10,5) -> LSTM(32) -> RepeatVector(10) -> LSTM(32, seq) -> TimeDistributed(Dense(5))) -> Fit (10 epochs) -> Compute threshold (mean + std) -> Flag errors > threshold -> Evaluate`
