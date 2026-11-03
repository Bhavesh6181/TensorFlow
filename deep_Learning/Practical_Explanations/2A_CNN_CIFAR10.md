# Practical 2A – CNN using CIFAR-10

## 1. Objective
The goal is to design, implement, and train a Convolutional Neural Network (CNN) to classify 32x32 color images into 10 object categories from the CIFAR-10 dataset, capturing spatial feature hierarchies.

## 2. Dataset
- **Name:** CIFAR-10
- **Files Handled:** The code demonstrates two cases:
  - **Case A:** Pre-separated `train_data.csv` and `test_data.csv`
  - **Case B:** Single `cifar-10.csv` split using `train_test_split` (80% train, 20% test)
- **Features:** 3072 pixel values per sample (32x32 image with 3 color channels: Red, Green, Blue)
- **Target:** Column `label` with 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `tensorflow` | Deep learning platform to build CNN architecture with 2D convolutions and pooling. |
| `pandas` | For loading CSV data and manipulating feature tables. |
| `numpy` | For numerical array manipulation and matrix transpositions. |
| `matplotlib.pyplot` | To graph training and validation accuracy and loss over epochs. |
| `train_test_split` (sklearn) | For partitioning a unified dataset into training and testing subsets (Case B). |
| `Sequential` | Container to stack CNN layers in linear order. |
| `Input`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` | Core CNN layers: `Conv2D` applies learnable image filters, `MaxPooling2D` reduces spatial size, `Flatten` transitions to dense layers, `Dense` performs classification. |

## 4. Important Functions and Code
- `.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)`: Reshapes raw CSV data where color channels (R, G, B) are stored sequentially into channels-first format, then transposes axes to standard channels-last format: `(batch, height=32, width=32, channels=3)`.
- `.astype("float32") / 255.0`: Normalizes RGB values from `[0, 255]` to `[0.0, 1.0]`.
- `Conv2D(32, (3, 3), activation="relu", padding="same")`: Applies 32 learnable 3x3 convolution filters. `padding="same"` ensures output height and width remain 32x32 by padding boundaries with zeros.
- `MaxPooling2D((2, 2))`: Downsamples feature maps by taking the maximum value in 2x2 windows, cutting spatial dimensions in half (32x32 -> 16x16) while retaining dominant features.
- `Conv2D(64, (3, 3), activation="relu", padding="same")`: Second convolution stage extracting higher-level feature combinations with 64 filters.
- `Flatten()`: Flattens the final pooled 3D feature volume into a 1D vector.
- `Dense(128, activation="relu")` & `Dense(10, activation="softmax")`: Fully connected classification head.

## 5. Important Parameters
- **Epochs (10):** Number of full training iterations.
- **Batch Size (64):** Number of samples evaluated before weight updates.
- **Validation Split (0.1):** 10% of training data used to monitor validation loss and accuracy.
- **Filter Sizes (32, 64):** Number of kernels learning low-level edges up to complex patterns.
- **Kernel Size (3x3):** Standard receptive field window scanning across image patches.
- **Pooling Window (2x2):** Downsamples spatial dimensions by 50% along height and width.
- **Optimizer (Adam):** Efficient adaptive optimization algorithm.
- **Loss (`sparse_categorical_crossentropy`):** Cross-entropy for integer labels.

## 6. Model Architecture
- **Input:** (32, 32, 3) color image.
- **Conv Block 1:** Conv2D (32 filters, 3x3, same) + ReLU -> MaxPooling2D (2x2) [output: 16x16x32]
- **Conv Block 2:** Conv2D (64 filters, 3x3, same) + ReLU -> MaxPooling2D (2x2) [output: 8x8x64]
- **Flatten Layer:** Reshapes (8, 8, 64) -> 4096 features.
- **Dense Layer:** 128 neurons, ReLU activation.
- **Output Layer:** 10 neurons, Softmax activation.
- **Data Flow:** `(32, 32, 3) -> Conv2D(32) -> MaxPool(16, 16, 32) -> Conv2D(64) -> MaxPool(8, 8, 64) -> Flatten(4096) -> Dense(128) -> Dense(10)`

## 7. Preprocessing Steps
- **Channel Reordering (`transpose(0, 2, 3, 1)`):** Standardizes CSV storage (which often groups all Red pixels, then Green, then Blue) into the standard `(height, width, channels)` format expected by TensorFlow.
- **Normalization (/255.0):** Scales inputs to `[0, 1]` to maintain stable gradients during backpropagation.

## 8. Expected Output
- Training log showing progressive accuracy gain across 10 epochs.
- Test loss and test accuracy printed.
- Two Matplotlib plots: Accuracy over epochs and Loss over epochs.

## 9. Viva Questions and Answers
**Q1. Why are CNNs superior to FNNs for image classification?**
**Ans:** CNNs preserve spatial 2D relationships, detect local patterns (edges, textures) via shared weights (parameter sharing), and are translation-invariant.

**Q2. What does `padding="same"` do in Conv2D?**
**Ans:** It adds zero-padding around the border so the output feature map has the exact same height and width as the input.

**Q3. What is the purpose of MaxPooling2D?**
**Ans:** It reduces spatial dimensions (downsampling), which cuts computational cost, controls overfitting, and achieves translational invariance.

**Q4. Why is `transpose(0, 2, 3, 1)` necessary in this practical?**
**Ans:** Because the CSV stores pixel values in channels-first order `(samples, channels, height, width)`. Transposing reorders axes to channels-last `(samples, height, width, channels)` as required by Keras on CPU/GPU.

**Q5. What is a convolution filter/kernel?**
**Ans:** A small matrix of learnable weights (here 3x3) that slides across the image to extract visual features like edges, corners, and textures.

**Q6. What happens to the dimensions after a (2, 2) MaxPooling?**
**Ans:** Height and width are halved (e.g., 32x32 becomes 16x16), while the number of channels/feature maps remains unchanged.

**Q7. How many classes are there in CIFAR-10?**
**Ans:** 10 classes of real-world objects.

**Q8. Why do we double the number of filters from 32 to 64 in the second Conv2D layer?**
**Ans:** As spatial dimensions decrease via pooling, deeper layers increase filter count to capture richer, higher-level abstract feature combinations.

**Q9. What is the function of the Flatten layer in a CNN?**
**Ans:** It converts the multidimensional feature volume (e.g., 8x8x64) into a single 1D vector (4096) so fully connected Dense layers can process it.

**Q10. What loss function is used in this practical?**
**Ans:** `sparse_categorical_crossentropy` because the labels are integers (0–9).

## 10. Quick Revision Flow
`Load CSV data -> Reshape & transpose to (32, 32, 3) -> Normalize (/255.0) -> Build CNN (Conv2D -> MaxPool -> Conv2D -> MaxPool -> Flatten -> Dense 128 -> Dense 10) -> Compile (Adam) -> Fit (10 epochs) -> Evaluate -> Plot Accuracy & Loss`
