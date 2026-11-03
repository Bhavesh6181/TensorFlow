# Practical 1A – FNN using MNIST

## 1. Objective
The goal is to build, train, and evaluate a Feedforward Neural Network (FNN) to classify 28x28 grayscale images of handwritten digits (0 to 9) into their corresponding numerical labels.

## 2. Dataset
- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Files Used:** `mnist_train.csv` and `mnist_test.csv`
- **Features:** 784 numerical pixel intensity values (representing a 28x28 flattened image)
- **Target:** Column `label` containing values from 0 to 9 (10 classes)

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For reading structured tabular image datasets from CSV files (`pd.read_csv`). |
| `tensorflow` | Deep learning framework used to construct, compile, and train neural network layers. |
| `matplotlib.pyplot` | For visualizing training and validation loss/accuracy curves across epochs. |
| `Sequential` | Keras model container to stack layers sequentially from input to output. |
| `Input`, `Flatten`, `Dense` | Core layer modules: `Input` sets tensor shape, `Flatten` converts 2D images to 1D, `Dense` creates fully-connected neural layers. |

## 4. Important Functions and Code
- `df.drop("label", axis=1).values`: Separates raw pixel feature arrays from the ground-truth target vector.
- `.reshape(-1, 28, 28)`: Reorganizes 784 consecutive pixel columns into two-dimensional spatial image grids (28 rows by 28 columns).
- `.astype("float32") / 255.0`: Scales 8-bit integer pixel brightness values (0–255) to floating-point numbers between 0.0 and 1.0.
- `Flatten()`: Unrolls the 28x28 matrix back into 784 sequential inputs for the dense layers without losing feature information.
- `Dense(128, activation="relu")`: First hidden layer with 128 neurons using Rectified Linear Unit activation to capture non-linear relationships.
- `Dense(10, activation="softmax")`: Final output layer with 10 units that outputs normalized probability distribution over classes 0 to 9.
- `model.compile(...)`: Configures learning parameters including the optimizer, objective loss function, and evaluation metrics.
- `model.fit(...)`: Executes mini-batch gradient descent across 11 epochs while splitting 10% of training data for validation.
- `model.evaluate(x_test, y_test)`: Calculates unbiased final loss and classification accuracy on completely unseen test records.

## 5. Important Parameters
- **Epochs (11):** Number of full passes through the training dataset.
- **Batch Size (32):** Number of samples processed before updating internal model weights.
- **Validation Split (0.1):** 10% of training records reserved to monitor overfitting during training.
- **Optimizer (SGD):** Stochastic Gradient Descent, updates weights along the direction of steepest loss reduction.
- **Loss (`sparse_categorical_crossentropy`):** Used because target labels are integer indices (0–9) rather than one-hot encoded vectors.
- **Activation Functions:** ReLU for hidden layers (prevents vanishing gradients) and Softmax for the output layer (outputs class probabilities that sum to 1).

## 6. Model Architecture
- **Input:** 28x28 grayscale image format.
- **Flatten Layer:** Reshapes (28, 28) into a flat vector of 784 features.
- **Hidden Layer 1:** Dense layer with 128 neurons and ReLU activation.
- **Hidden Layer 2:** Dense layer with 64 neurons and ReLU activation.
- **Output Layer:** Dense layer with 10 neurons and Softmax activation.
- **Data Flow:** `(28, 28) -> Flatten(784) -> Dense(128) -> Dense(64) -> Dense(10)`

## 7. Preprocessing Steps
- **Reshaping:** Converts flat 784-element rows from the CSV into 28x28 image structures so they can pass through Keras input specifications.
- **Normalization (Division by 255):** Keeps input values strictly within `[0.0, 1.0]`. This prevents large pixel values from dominating gradient calculations, ensuring faster and smoother convergence during gradient descent.

## 8. Expected Output
- Model summary showing total and trainable parameters.
- Epoch-wise training progress showing decreasing loss and increasing accuracy (typically reaching 95%+ accuracy).
- Printed test loss and test accuracy.
- Two side-by-side Matplotlib line graphs displaying training vs. validation loss and training vs. validation accuracy over 11 epochs.

## 9. Viva Questions and Answers
**Q1. What is a Feedforward Neural Network (FNN)?**
**Ans:** An artificial neural network where connections between nodes do not form cycles; data moves strictly forward from input to output.

**Q2. Why do we divide pixel values by 255.0?**
**Ans:** Pixel values range from 0 to 255. Dividing by 255 normalizes features into `[0, 1]`, preventing large gradients and accelerating gradient descent convergence.

**Q3. What is the role of the Flatten layer?**
**Ans:** It converts multidimensional inputs (like 28x28 images) into a single 1D vector (784 features) suitable for feeding into fully connected Dense layers.

**Q4. Why is `sparse_categorical_crossentropy` used instead of `categorical_crossentropy`?**
**Ans:** Because the target labels are integer class indices (e.g., 0, 1, 2) rather than one-hot encoded vectors (e.g., [1, 0, 0]).

**Q5. Why do we use ReLU activation in hidden layers?**
**Ans:** ReLU is computationally fast, introduces non-linearity, and helps avoid the vanishing gradient problem common with sigmoid or tanh.

**Q6. What does the Softmax activation function do in the final layer?**
**Ans:** It converts raw logits into a probability distribution over the 10 classes where all output values sum up to 1.

**Q7. What is the difference between an epoch and a batch?**
**Ans:** An epoch is one complete pass through the entire dataset; a batch is a small subset of samples processed before updating weights.

**Q8. What optimizer is used in this practical?**
**Ans:** Stochastic Gradient Descent (`tf.keras.optimizers.SGD()`).

**Q9. What does `validation_split=0.1` mean?**
**Ans:** It automatically reserves 10% of the training data to evaluate model performance after every epoch to check for overfitting.

**Q10. How many total output classes are there in MNIST?**
**Ans:** 10 classes, representing handwritten digits from 0 through 9.

## 10. Quick Revision Flow
`Load CSVs -> Separate X/y -> Reshape to 28x28 -> Normalize (/255.0) -> Define Sequential (Flatten + Dense) -> Compile (SGD, sparse_categorical_crossentropy) -> Fit (11 epochs) -> Evaluate -> Plot Loss & Accuracy`
