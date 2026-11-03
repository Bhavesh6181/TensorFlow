# Practical 1B – FNN using Fashion MNIST

## 1. Objective
The goal is to build, train, and evaluate a Feedforward Neural Network (FNN) to classify grayscale images of 10 different clothing items (such as shirts, sneakers, and bags) from the Fashion MNIST dataset.

## 2. Dataset
- **Name:** Fashion MNIST (Zalando clothing dataset)
- **Files Used:** `fashion-mnist_train.csv` and `fashion-mnist_test.csv`
- **Features:** 784 pixel intensity values representing 28x28 images of fashion articles
- **Target:** Column `label` with values 0 to 9 representing 10 clothing categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | To read the CSV datasets containing pixel values and clothing labels into DataFrames. |
| `tensorflow` | Deep learning framework used for layer definitions, network compilation, and model training. |
| `matplotlib.pyplot` | To generate side-by-side plots of loss and accuracy histories across epochs. |
| `Sequential` | Container to stack layers sequentially from input to output. |
| `Input`, `Flatten`, `Dense` | Building blocks: `Input` sets dimensions, `Flatten` reshapes 2D grids to 1D, `Dense` performs fully connected linear transformations with activations. |

## 4. Important Functions and Code
- `train_df.drop("label", axis=1).values`: Isolates the 784 pixel columns from class labels.
- `.reshape(-1, 28, 28)`: Reshapes flattened 784 pixel vectors into 28x28 matrices representing physical image dimensions.
- `.astype("float32") / 255.0`: Converts integer pixel values (0–255) into standardized floats between 0.0 and 1.0.
- `Flatten()`: Unrolls 28x28 matrices into 784 continuous input features for Dense layers.
- `Dense(128, activation="relu")` & `Dense(64, activation="relu")`: Two hidden layers extracting complex non-linear combinations of garment shapes.
- `Dense(10, activation="softmax")`: Produces probability scores across all 10 garment categories.
- `model.compile(...)`: Configures SGD optimizer, sparse categorical cross-entropy loss, and accuracy metrics.
- `model.fit(...)`: Trains the model over 11 epochs with batch size 32 and 10% validation split.
- `model.evaluate(...)`: Calculates final generalization performance on the test dataset.

## 5. Important Parameters
- **Epochs (11):** Total cycles through the training records.
- **Batch Size (32):** Number of images evaluated before weights are updated.
- **Validation Split (0.1):** 10% held-out training data used to monitor loss and validation accuracy.
- **Optimizer (SGD):** Stochastic Gradient Descent for weight optimization.
- **Loss (`sparse_categorical_crossentropy`):** Appropriate for integer class labels from 0 to 9.
- **Activation Functions:** ReLU for intermediate layers (avoids saturation) and Softmax for classification output.

## 6. Model Architecture
- **Input:** 28x28 grayscale clothing image.
- **Flatten Layer:** Transforms (28, 28) into 784 nodes.
- **Hidden Layer 1:** 128 neurons, ReLU activation.
- **Hidden Layer 2:** 64 neurons, ReLU activation.
- **Output Layer:** 10 neurons, Softmax activation.
- **Data Flow:** `(28, 28) -> Flatten(784) -> Dense(128) -> Dense(64) -> Dense(10)`

## 7. Preprocessing Steps
- **Reshaping:** Recreates the 2D layout (28, 28) from flat CSV rows.
- **Normalization (/255.0):** Rescales pixel values to the range `[0, 1]`, ensuring balanced weight updates and numerical stability during backpropagation.

## 8. Expected Output
- Training log printing loss and accuracy for 11 epochs.
- Test loss and test accuracy (typically ~85–88% for standard FNN).
- Two visual plots: Training vs. Validation Loss and Training vs. Validation Accuracy.

## 9. Viva Questions and Answers
**Q1. How is Fashion MNIST different from original MNIST?**
**Ans:** Both have 28x28 grayscale images and 10 classes, but Fashion MNIST contains clothing items instead of digits, making it a more challenging and realistic benchmark.

**Q2. What does an accuracy of 85% on Fashion MNIST mean?**
**Ans:** It means the neural network correctly identifies the garment category for 85 out of every 100 test images.

**Q3. Why do we need `Flatten` before the first `Dense` layer?**
**Ans:** Dense layers only accept 1D feature vectors; Flatten transforms the 2D (28, 28) image matrix into a 784-element 1D vector.

**Q4. Can an FNN easily capture spatial relationships in images?**
**Ans:** FNN flattens images and loses 2D spatial context; that is why Convolutional Neural Networks (CNNs) generally outperform FNNs on image tasks.

**Q5. Why is the output layer size 10?**
**Ans:** There are 10 distinct clothing classes to classify.

**Q6. What happens if we do not normalize the pixel inputs?**
**Ans:** Training becomes unstable or very slow because large inputs (up to 255) cause large gradients and weight oscillations.

**Q7. What does the Softmax function guarantee?**
**Ans:** All output values are between 0 and 1, and their total sum equals exactly 1, representing class probabilities.

**Q8. Why use SGD over Adam here?**
**Ans:** SGD is the classic foundational gradient descent algorithm; while simpler, it demonstrates fundamental neural network optimization.

**Q9. What are the two hidden layer sizes used in this code?**
**Ans:** 128 neurons in the first hidden layer and 64 neurons in the second hidden layer.

**Q10. How is overfitting detected during training?**
**Ans:** When training loss keeps decreasing but validation loss starts increasing.

## 10. Quick Revision Flow
`Load Fashion CSVs -> Extract labels & features -> Reshape to 28x28 -> Normalize (/255.0) -> Sequential model (Flatten -> Dense 128 -> Dense 64 -> Dense 10) -> Compile (SGD) -> Fit (11 epochs) -> Evaluate -> Plot graphs`
