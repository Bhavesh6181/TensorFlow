# Practical 5A – Transfer Learning using Caltech-101

## 1. Objective
The goal is to classify images from the Caltech-101 dataset using Transfer Learning with a pretrained VGG16 convolutional network, demonstrating both Feature Extraction (frozen base) and Fine-Tuning (unfreezing top layers).

## 2. Dataset
- **Name:** Caltech-101
- **Structure:** Directory `caltech-101/` with subfolders for each object category (e.g., airplanes, faces, motorbikes, etc.)
- **Resolution & Format:** Images resized to standard input dimensions of `(224, 224, 3)`
- **Subset Used:** `train_ds.take(20)` and `val_ds.take(5)` are used to reduce training time during practical demonstration.

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `tensorflow` | Deep learning platform used for model loading, layer customization, and training. |
| `matplotlib.pyplot` | For displaying sample test images with actual vs. predicted class labels. |
| `VGG16` (tensorflow.keras.applications) | Pretrained deep CNN architecture trained on 1.4 million ImageNet images. |
| `image_dataset_from_directory` (tf.keras.utils) | Efficient utility that automatically reads images from folder structures, generates labels, and creates batched datasets. |
| `Input`, `Dense`, `Flatten`, `Lambda`, `Model` | Keras Functional API layers to assemble preprocessing, pretrained base, and custom classification heads. |

## 4. Important Functions and Code
- `image_dataset_from_directory(path, validation_split=0.2, subset=..., image_size=(224, 224), batch_size=32)`: Loads images directly from disk, resizes them to 224x224, and splits them into 80% training and 20% validation.
- `train_ds.take(20)`: Subsets 20 batches to accelerate practical lab execution.
- `VGG16(weights="...", include_top=False, input_shape=(224, 224, 3))`: Loads pretrained VGG16 weights without the original 1000-class ImageNet classification top.
- `base.trainable = False`: **Phase 1 (Feature Extraction):** Freezes all pretrained VGG16 convolutional layers so their learned weights are not modified.
- `Lambda(tf.keras.applications.vgg16.preprocess_input)`: Applies VGG16-specific input preprocessing (channel zero-centering with ImageNet mean BGR subtraction).
- `Flatten()(x) -> Dense(128, relu) -> Dense(len(classes), softmax)`: Custom classification head adapted to Caltech-101 classes.
- **Phase 2 (Fine-Tuning):**
  - `base.trainable = True`: Unfreezes the base model.
  - `for layer in base.layers[:-4]: layer.trainable = False`: Re-freezes all early and middle layers, leaving only the **last 4 convolutional layers** trainable.
  - `Adam(learning_rate=0.00001)`: Recompiles with a very small learning rate (`1e-5`) to carefully fine-tune weights without destroying pretrained feature detectors.
- `next(iter(val_ds))`: Grabs a batch of validation images to run inference and visualize predictions.

## 5. Important Parameters
- **Input Dimensions:** `(224, 224, 3)` — native input size required by VGG16.
- **Batch Size (32):** Mini-batch size.
- **Phase 1 Epochs (5):** Trains only the new custom classification head while base is frozen.
- **Phase 2 Epochs (2):** Fine-tunes the top 4 convolutional layers.
- **Fine-Tuning Learning Rate (`1e-5`):** 100x smaller than standard learning rate to prevent catastrophic forgetting.
- **Loss (`sparse_categorical_crossentropy`):** Multi-class loss for integer directory class labels.

## 6. Model Architecture
- **Input Layer:** `(224, 224, 3)` RGB image.
- **Preprocessing:** `Lambda(vgg16.preprocess_input)`.
- **Pretrained Base:** VGG16 (Block 1 through Block 5 convolutions, `include_top=False`).
- **Classification Head:**
  - `Flatten()` -> Flattens (7, 7, 512) feature maps to 25,088 elements.
  - `Dense(128, ReLU)`
  - `Dense(num_classes, Softmax)`
- **Data Flow:** `(224, 224, 3) -> Preprocess -> VGG16 Base -> Flatten(25088) -> Dense(128) -> Dense(num_classes)`

## 7. Preprocessing Steps
- **Resizing (`224x224`):** Rescales arbitrary Caltech-101 image sizes to the fixed resolution expected by VGG16.
- **VGG16 Preprocessing (`preprocess_input`):** Converts RGB to BGR and subtracts the ImageNet dataset mean from each color channel.
- **Batching and Prefetching:** Handled automatically by `image_dataset_from_directory`.

## 8. Expected Output
- Number of found files and classes logged during dataset loading.
- Phase 1 (Feature Extraction): 5 epochs of training progress.
- Phase 2 (Fine-Tuning): 2 epochs of fine-tuning with small learning rate.
- Final validation loss and validation accuracy printed.
- Visual display of a test image using `plt.imshow` with title: `"Actual: <label>, Predicted: <label>"`.

## 9. Viva Questions and Answers
**Q1. What is Transfer Learning?**
**Ans:** A machine learning technique where a model pretrained on a massive benchmark dataset (e.g., ImageNet) is reused as the starting point for a different but related task (e.g., Caltech-101).

**Q2. Why do we use Transfer Learning instead of training from scratch?**
**Ans:** Training from scratch requires millions of labeled images and days of compute. Transfer learning allows high accuracy with much smaller datasets and significantly shorter training time.

**Q3. What does `include_top=False` mean?**
**Ans:** It excludes the original final fully connected classification layers (which classified 1000 ImageNet categories), allowing us to append our own custom classification head.

**Q4. Why do we freeze the base layers (`base.trainable = False`) in Phase 1?**
**Ans:** The newly added dense layers have random weights. If the entire network were trainable, large initial gradients would destroy the pretrained feature representations (catastrophic forgetting).

**Q5. What is Fine-Tuning (Phase 2)?**
**Ans:** Unfreezing the top few convolutional layers of the base model and training them alongside the custom classifier using a very low learning rate to adapt features to the new dataset.

**Q6. Why is a very small learning rate (`0.00001`) used during fine-tuning?**
**Ans:** To make gentle, incremental adjustments to the pretrained weights without destroying the valuable general visual features learned from ImageNet.

**Q7. Why are only the last 4 layers unfrozen (`base.layers[:-4]`)?**
**Ans:** Early layers learn generic features (edges, textures) that apply to all images. Later layers learn task-specific high-level shapes; fine-tuning only the later layers adapts those shapes to our target domain.

**Q8. What does `train_ds.take(20)` do?**
**Ans:** It limits the training dataset to the first 20 batches to reduce execution time during classroom demonstrations.

**Q9. What is the role of the `Lambda` layer?**
**Ans:** It wraps the arbitrary function `vgg16.preprocess_input` into a Keras layer so input preprocessing runs directly inside the model graph on GPU/CPU.

**Q10. What is VGG16?**
**Ans:** A 16-layer deep convolutional neural network developed by the Visual Geometry Group at Oxford, known for its simple architecture using repeated 3x3 convolutions and 2x2 max pooling.

## 10. Quick Revision Flow
`Load Caltech-101 via image_dataset_from_directory -> Load VGG16 (include_top=False) -> Freeze base (base.trainable=False) -> Add custom head (Lambda -> VGG16 -> Flatten -> Dense 128 -> Dense output) -> Train Phase 1 (5 epochs) -> Unfreeze top 4 layers of base -> Recompile with low lr (1e-5) -> Train Phase 2 (2 epochs) -> Evaluate -> Show sample prediction`
