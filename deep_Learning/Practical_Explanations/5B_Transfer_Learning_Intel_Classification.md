# Practical 5B – Transfer Learning for Intel Classification

## 1. Objective
The goal is to classify natural scene images into 6 distinct landscape categories from the Intel Image Classification dataset using Transfer Learning with a pretrained VGG16 network and Global Average Pooling.

## 2. Dataset
- **Name:** Intel Image Classification dataset
- **Directory:** `Intel Image Classification/`
- **Classes (6):** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
- **Resolution:** Images resized to `(224, 224, 3)`

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `tensorflow` | Deep learning platform to build, compile, and train the transfer learning pipeline. |
| `matplotlib.pyplot` | For plotting and visualizing sample test images alongside predicted labels. |
| `VGG16` (tensorflow.keras.applications) | Pretrained convolutional network backbone. |
| `image_dataset_from_directory` (tf.keras.utils) | Loads and batches images from class subdirectories with automated train/validation splitting. |
| `GlobalAveragePooling2D` | Replaces traditional Flatten; computes average of each feature map, dramatically reducing parameter count. |
| `Input`, `Dense`, `Lambda`, `Model` | Functional layers for assembling the end-to-end model. |

## 4. Important Functions and Code
- `image_dataset_from_directory(path, validation_split=0.2, subset=..., image_size=(224, 224), batch_size=32)`: Automatically reads images from the 6 category folders, applies 80/20 train/validation split, and resizes inputs to 224x224.
- `base = VGG16(weights="...", include_top=False, input_shape=(224, 224, 3))`: Instantiates the VGG16 convolutional feature extractor.
- `GlobalAveragePooling2D()(x)`: **Key distinction from 5A.** Instead of `Flatten()`, it takes the average across the 7x7 spatial dimensions for each of the 512 channels, reducing the feature map directly to a 512-element 1D vector.
- `Dense(128, activation="relu")` -> `Dense(len(classes), activation="softmax")`: Classification layers for the 6 natural scene classes.
- `base.trainable = False`: Phase 1 feature extraction where the base model weights remain locked.
- `base.trainable = True` + `for layer in base.layers[:-4]: layer.trainable = False`: Phase 2 fine-tuning where only the last 4 convolutional layers are updated.
- `tf.keras.optimizers.Adam(learning_rate=0.00001)`: Low learning rate for fine-tuning.

## 5. Important Parameters
- **Input Size:** `(224, 224, 3)`.
- **Classes (6):** Buildings, Forest, Glacier, Mountain, Sea, Street.
- **Batch Size (32):** Mini-batch size.
- **Phase 1 Epochs (5):** Trains only the custom classification head.
- **Phase 2 Epochs (2):** Fine-tunes the last 4 convolutional layers.
- **Fine-Tuning Learning Rate (`1e-5`):** Prevents destructive weight updates to pretrained layers.
- **Optimizer:** Adam.
- **Loss:** `sparse_categorical_crossentropy`.

## 6. Model Architecture
- **Input:** `(224, 224, 3)` natural scene image.
- **Preprocessing:** `Lambda(vgg16.preprocess_input)`.
- **Base Feature Extractor:** VGG16 (`include_top=False`) -> Output shape `(7, 7, 512)`.
- **Pooling:** `GlobalAveragePooling2D()` -> Output shape `(512,)`.
- **Dense Layer:** 128 neurons, ReLU activation.
- **Output Layer:** 6 neurons, Softmax activation.
- **Data Flow:** `(224, 224, 3) -> Preprocess -> VGG16 Base -> GlobalAvgPool(512) -> Dense(128) -> Dense(6)`

## 7. Preprocessing Steps
- **Image Resizing (`224x224`):** Resizes all natural landscape images to uniform dimensions.
- **VGG16 Mean Subtraction (`preprocess_input`):** Zero-centers image channels using ImageNet dataset statistics.
- **Stratified Batch Generation:** Partitioned into batches of 32 using `image_dataset_from_directory`.

## 8. Expected Output
- Class names identified by directory scan (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).
- Phase 1: 5 epochs of training logs showing rapid convergence.
- Phase 2: 2 epochs of fine-tuning logs.
- Test loss and test accuracy printed.
- Display of one test image with actual and predicted class label.

## 9. Viva Questions and Answers
**Q1. What is the difference between `GlobalAveragePooling2D` (used in 5B) and `Flatten` (used in 5A)?**
**Ans:** `Flatten` unrolls the entire `(7, 7, 512)` tensor into 25,088 features, creating a massive number of weights for the subsequent Dense layer. `GlobalAveragePooling2D` averages each 7x7 feature map into a single value, outputting only 512 features. This prevents overfitting and drastically reduces parameter count.

**Q2. What are the 6 classes in the Intel Image Classification dataset?**
**Ans:** Buildings, Forest, Glacier, Mountain, Sea, and Street.

**Q3. Why does VGG16 perform well on natural landscape images?**
**Ans:** VGG16 was trained on 1.4 million ImageNet images, which already contain countless outdoor scenes, trees, bodies of water, and structures. The low- and mid-level feature filters transfer naturally to landscape classification.

**Q4. What is the role of `base.trainable = False`?**
**Ans:** It freezes the pretrained VGG16 weights during Phase 1 so only the newly added Dense layers are trained.

**Q5. Why do we fine-tune with a learning rate of $10^{-5}$ instead of $10^{-3}$?**
**Ans:** A high learning rate would cause drastic weight updates, destroying the pretrained feature representations. A tiny learning rate ($10^{-5}$) makes delicate adjustments suited to the new dataset.

**Q6. What does `include_top=False` do?**
**Ans:** It drops the original ImageNet classifier (the top 3 dense layers), keeping only the convolutional feature extraction base.

**Q7. How many layers are kept trainable during Phase 2?**
**Ans:** Only the last 4 layers of VGG16; all preceding layers remain frozen via `layer.trainable = False`.

**Q8. What is the output shape of `GlobalAveragePooling2D` when input is `(7, 7, 512)`?**
**Ans:** `(512,)`.

**Q9. What loss function is used?**
**Ans:** `sparse_categorical_crossentropy` because class labels are integers (0 to 5).

**Q10. How does Transfer Learning combat overfitting when dataset size is modest?**
**Ans:** Pretrained weights act as strong regularizers. The model does not need to learn edge and texture filters from scratch, reducing the risk of fitting noise in small datasets.

## 10. Quick Revision Flow
`Load Intel dataset -> Load VGG16 (include_top=False) -> Freeze base -> Build model (Lambda -> VGG16 -> GlobalAveragePooling2D -> Dense 128 -> Dense 6) -> Train Phase 1 (5 epochs) -> Unfreeze last 4 layers of base -> Recompile with low lr (1e-5) -> Train Phase 2 (2 epochs) -> Evaluate -> Visualize sample prediction`
