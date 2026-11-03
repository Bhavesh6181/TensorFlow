# Practical 3D – LSTM Autoencoder for NASA Turbofan

## 1. Objective
The goal is to detect operational faults and mechanical degradation in aircraft engines using a multivariate LSTM Autoencoder trained on healthy operating cycles of NASA Turbofan engines.

## 2. Dataset
- **Name:** NASA Turbofan Engine Degradation Simulation Dataset (`nasa_turbofan.csv`)
- **Metadata Columns:** `unit` (engine ID number) and `cycle` (operational flight cycles over time)
- **Features (21):** 21 continuous sensor measurements (`sensor1` to `sensor21` measuring temperature, pressure, fan speeds, etc.)
- **Target:** `Fault_Label` where `0 = Healthy operating state` and `1 = Faulty / degraded state`

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For sorting engine records by engine unit and flight cycle, and handling sensor readings. |
| `numpy` | For constructing per-engine sliding time windows and calculating reconstruction error. |
| `matplotlib.pyplot` | For visualizing training and validation loss curves. |
| `MinMaxScaler` (sklearn) | For scaling all 21 sensor channels into `[0, 1]`. |
| `accuracy_score`, `confusion_matrix`, `classification_report` (sklearn) | For evaluating fault detection accuracy, precision, and recall. |
| `Input`, `LSTM`, `RepeatVector`, `TimeDistributed`, `Dense` (tensorflow.keras) | Layers used to build the sequence-to-sequence LSTM autoencoder. |

## 4. Important Functions and Code
- `df.sort_values(["unit", "cycle"])`: Sorts records first by engine unit, then chronologically by cycle.
- `features = [f"sensor{i}" for i in range(1, 22)]`: Programmatically gathers all 21 sensor columns.
- Per-unit windowing: `for unit in df["unit"].unique(): ...`: Crucial step ensuring sliding windows of size 10 are created **within each engine independently**, preventing data from different engines from mixing together.
- `X_train = X[normal]`: Trains the model exclusively on healthy engine cycles (`Fault_Label == 0`).
- `LSTM(32, activation="relu")(input_layer)`: **Encoder** LSTM compresses 10 time steps across 21 sensors into a 32-dimensional state vector.
- `RepeatVector(window)`: Repeats the 32-element context vector across the 10 time steps.
- `LSTM(32, activation="relu", return_sequences=True)`: **Decoder** LSTM reconstructs the temporal sequence.
- `TimeDistributed(Dense(21))`: Reconstructs all 21 sensor readings at each time step.
- `threshold = np.mean(train_errors) + np.std(train_errors)`: Threshold for flagging mechanical faults.

## 5. Important Parameters
- **Window Size (10):** 10 consecutive operational cycles per evaluation window.
- **Sensor Features (21):** Full array of engine monitoring sensors.
- **LSTM Units (32):** Latent feature dimension for encoder and decoder.
- **Epochs (10):** Training passes.
- **Batch Size (32):** Mini-batch size.
- **Optimizer (Adam) & Loss (MSE):** Optimized for multi-sensor sequence reconstruction.

## 6. Model Architecture
- **Input:** `(10, 21)` tensor representing 10 cycles of 21 sensor readings.
- **Encoder:** `LSTM(32, ReLU)` -> Compresses input into a 32-dimensional vector.
- **Bridge:** `RepeatVector(10)` -> Expands vector to `(10, 32)`.
- **Decoder:** `LSTM(32, ReLU, return_sequences=True)` -> `(10, 32)`.
- **Output Layer:** `TimeDistributed(Dense(21))` -> Reconstructs original shape `(10, 21)`.
- **Data Flow:** `(10, 21) -> LSTM(32) -> RepeatVector(10) -> LSTM(32, seq) -> TimeDistributed(Dense(21))`

## 7. Preprocessing Steps
- **Multi-Level Sorting:** Sorts by `unit` and `cycle` to maintain proper chronological order for each engine.
- **Forward-Fill (`ffill().fillna(0)`):** Imputes any missing sensor values.
- **Min-Max Normalization:** Scales all 21 sensor channels to `[0, 1]`.
- **Per-Engine Sliding Window:** Generates 10-cycle sequences per engine unit so window transitions never cross between two different engines.

## 8. Expected Output
- Dataset head and shape.
- 10 training epochs showing progressive reduction in reconstruction error.
- Calculated fault threshold value.
- Total detected engine faults.
- Classification accuracy, Confusion Matrix, and Classification Report.
- Training vs. Validation Loss plot.

## 9. Viva Questions and Answers
**Q1. What is the real-world application of this practical?**
**Ans:** Predictive maintenance in aviation. Detecting early engine degradation before catastrophic mechanical failure occurs.

**Q2. Why is windowing performed per engine unit (`df["unit"].unique()`)?**
**Ans:** Different engines have different operational histories. If windows crossed between engine boundaries, the end of Engine 1 would mix with the start of Engine 2, creating artificial, invalid transitions.

**Q3. How many sensors are monitored per engine cycle?**
**Ans:** 21 sensors measuring variables like temperatures, fan speeds, and pressures.

**Q4. What is the shape of each input sample fed into the LSTM?**
**Ans:** `(10, 21)` — 10 consecutive operating cycles with 21 sensor readings each.

**Q5. Why is the Autoencoder trained only on healthy engine cycles (`Fault_Label == 0`)?**
**Ans:** To learn the normal operational dynamics of a healthy engine. When mechanical faults develop, sensor patterns change and the model produces high reconstruction error, flagging the fault.

**Q6. What does `TimeDistributed(Dense(21))` do?**
**Ans:** It applies a Dense layer with 21 outputs independently to each of the 10 time steps, reconstructing all 21 sensor measurements across time.

**Q7. How is the fault threshold determined?**
**Ans:** `Threshold = Mean(training errors) + Standard Deviation(training errors)`.

**Q8. What loss function is minimized?**
**Ans:** Mean Squared Error (MSE).

**Q9. Why are LSTMs suitable for engine sensor monitoring?**
**Ans:** Engine wear and degradation develop gradually over time. LSTMs have recurrent memory cells that capture temporal degradation trends across cycles.

**Q10. What does an error above the threshold signify?**
**Ans:** The engine sensors have deviated significantly from healthy operating behavior, indicating a mechanical fault or degradation.

## 10. Quick Revision Flow
`Sort by unit & cycle -> Impute NaNs -> Scale 21 sensors (MinMaxScaler) -> Loop unique units to create 10-step windows -> Filter healthy windows (y == 0) -> Build LSTM Autoencoder (Input(10,21) -> LSTM 32 -> RepeatVector 10 -> LSTM 32 seq -> TimeDistributed Dense 21) -> Train (10 epochs) -> Calculate threshold -> Flag errors > threshold -> Evaluate`
