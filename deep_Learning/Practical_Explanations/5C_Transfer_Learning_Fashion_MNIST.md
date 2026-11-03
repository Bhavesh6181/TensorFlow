# Practical 5C – Transfer Learning using Fashion MNIST

## 1. Objective
The goal is to classify 10 clothing categories from the Fashion MNIST dataset using Transfer Learning with a pretrained VGG16 network, adapting grayscale clothing images to RGB format and leveraging ImageNet feature representations.

## 2. Dataset
- **Name:** Fashion MNIST (stored in directory format: `fashion mnist/`)
- **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Key Adaptation:** Fashion MNIST is natively 28x28 grayscale. However, in this practical, the images are loaded from directories at `(224, 224)` resolution with `color_mode="rgb"` to satisfy the 3-channel input requirements of VGG16.

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `tensorflow` | Primary framework to build, train, and fine-tune the deep transfer learning model. |
| `matplotlib.pyplot` | For plotting and visualizing test image predictions alongside ground truth labels. |
| `VGG16` (tensorflow.keras.applications) | Pretrained convolutional network used as the feature extraction base. |
| `image_dataset_from_directory` (tf.keras.utils) | Loads images from subfolders, applies validation splitting, resizes to 224x224, and sets `color_mode="rgb"`. |
| `GlobalAveragePooling2D` | Averages 7x7 feature maps into a 512-dimensional vector, avoiding parameter explosion and reducing overfitting. |
| `Input`, `Dense`, `Lambda`, `Model` | Functional layers for assembling the end-to-end model pipeline. |

## 4. Important Functions and Code
- `image_dataset_from_directory(path, validation_split=0.2, subset=..., image_size=(224, 224), batch_size=32, color_mode="rgb")`: Reads images, resizes to 224x224, and explicitly converts them to 3-channel RGB format.
- `base = VGG16(weights="...", include_top=False, input_shape=(224, 224, 3))`: Loads pretrained VGG16 feature extractor.
- `base.trainable = False`: Phase 1 feature extraction: freezes base convolutional layers.
- `GlobalAveragePooling2D()(x)`: Condenses the 7x7x512 feature volume into a 512-dimensional vector.
- `Dense(128, activation="relu")` -> `Dense(len(classes), activation="softmax")`: Custom classification head for the 10 clothing classes.
- `base.trainable = True` + `for layer in base.layers[:-4]: layer.trainable = False`: Phase 2 fine-tuning: unfreezes only the last 4 convolutional layers.
- `tf.keras.optimizers.Adam(learning_rate=0.00001)`: Small learning rate for fine-tuning.

## 5. Important Parameters
- **Input Dimensions:** `(224, 224, 3)` — upsampled to match VGG16 architecture.
- **Color Mode:** `"rgb"` (converts single-channel images into 3 channels by replicating grayscale values).
- **Classes (10):** 10 clothing categories.
- **Batch Size (32):** Mini-batch size.
- **Phase 1 Epochs (5):** Feature extraction with frozen base.
- **Phase 2 Epochs (2):** Fine-tuning the top 4 layers of VGG16.
- **Fine-Tuning Learning Rate (`1e-5`):** Low learning rate to carefully adjust weights.
- **Loss:** `sparse_categorical_crossentropy`.

## 6. Model Architecture
- **Input:** `(224, 224, 3)` RGB-adapted clothing image.
- **Preprocessing:** `Lambda(vgg16.preprocess_input)`.
- **Base Feature Extractor:** VGG16 (`include_top=False`) -> Output `(7, 7, 512)`.
- **Pooling:** `GlobalAveragePooling2D()` -> Output `(512,)`.
- **Dense Layer:** 128 neurons, ReLU activation.
- **Output Layer:** 10 neurons, Softmax activation.
- **Data Flow:** `(224, 224, 3) -> Preprocess -> VGG16 Base -> GlobalAvgPool(512) -> Dense(128) -> Dense(10)`

## 7. Preprocessing Steps
- **Color Mode Conversion (`color_mode="rgb"`):** VGG16 was trained on 3-channel RGB images; single-channel images are duplicated across 3 channels to maintain architecture compatibility.
- **Spatial Upsampling (`image_size=(224, 224)`):** Resizes images to the standard VGG16 input resolution.
- **VGG16 Preprocessing (`preprocess_input`):** Standard zero-centering channel subtraction.

## 8. Expected Output
- Directory scan confirming 10 clothing class subfolders.
- Phase 1 (Feature Extraction): 5 epochs of training progress with increasing accuracy.
- Phase 2 (Fine-Tuning): 2 epochs of fine-tuning logs.
- Final test loss and test accuracy.
- Visual display of a test clothing item with actual and predicted class labels.

## 9. Viva Questions and Answers
**Q1. Fashion MNIST images are natively 28x28 grayscale. Why are they loaded as `(224, 224, 3)` in this practical?**
**Ans:** VGG16 was designed for 3-channel (RGB) images with a minimum spatial resolution (typically 224x224). Resizing and converting to RGB (`color_mode="rgb"`) makes the dataset fully compatible with the pretrained VGG16 architecture.

**Q2. How does `color_mode="rgb"` convert grayscale images?**
**Ans:** It replicates the single grayscale pixel value across all three RGB channels ($R = G = B$), creating a valid 3-channel image without altering the underlying visual content.

**Q3. How does Transfer Learning on Fashion MNIST (5C) compare to training an FNN from scratch (1B)?**
**Ans:** An FNN in 1B trains from scratch and only reaches ~86% accuracy while discarding spatial structures. Transfer learning leverages deep hierarchical filters already learned from ImageNet, yielding better feature extraction and higher accuracy.

**Q4. Why is `GlobalAveragePooling2D` preferred over `Flatten` here?**
**Ans:** `Flatten` creates 25,088 connections into the Dense layer, introducing millions of trainable parameters prone to overfitting. `GlobalAveragePooling2D` averages spatial dimensions down to 512 values, creating a lightweight, robust classification head.

**Q5. What is the two-phase training strategy used here?**
**Ans:**
- **Phase 1 (Feature Extraction):** Base VGG16 is frozen; only the newly added Dense layers are trained for 5 epochs.
- **Phase 2 (Fine-Tuning):** The last 4 layers of VGG16 are unfrozen and trained with a very small learning rate ($10^{-5}$) for 2 epochs.

**Q6. What happens if we unfreeze all layers of VGG16 with a high learning rate?**
**Ans:** Catastrophic forgetting occurs: the large gradients will overwrite and destroy the pretrained ImageNet weights, ruining model performance.

**Q7. What does `include_top=False` accomplish?**
**Ans:** It omits the original 1000-class dense classification layers from ImageNet so we can substitute a 10-class classifier for Fashion MNIST.

**Q8. What are the 10 classes in Fashion MNIST?**
**Ans:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

**Q9. What is the fine-tuning learning rate used in this code?**
**Ans:** $0.00001$ (`1e-5`).

**Q10. Why is `Lambda(vgg16.preprocess_input)` necessary?**
**Ans:** VGG16 requires input pixels to be zero-centered with respect to the ImageNet dataset (mean BGR subtraction). Omitting this preprocessing degrades transfer performance.

## 10. Quick Revision Flow
`Load Fashion MNIST from folders (224x224, color_mode="rgb") -> Load VGG16 (include_top=False) -> Freeze base -> Add classification head (Lambda -> VGG16 -> GlobalAveragePooling2D -> Dense 128 -> Dense 10) -> Train Phase 1 (5 epochs) -> Unfreeze last 4 layers of base -> Recompile with low lr (1e-5) -> Train Phase 2 (2 epochs) -> Evaluate -> Plot prediction sample`
