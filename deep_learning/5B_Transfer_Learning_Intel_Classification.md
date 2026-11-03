# Practical 5B – Transfer Learning for Intel Classification

```python
# Practical 5B: Intel Image Classification using VGG16

# Step 1: Import libraries
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model

# Step 2: Load and preprocess dataset
path = "Intel Image Classification"

train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

classes = train_ds.class_names

# Step 3: Load pretrained VGG16 model
base = VGG16(
    weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Step 4: Freeze base layers
base.trainable = False

# Step 5: Add custom classification layers
inp = Input(shape=(224, 224, 3))

x = Lambda(tf.keras.applications.vgg16.preprocess_input)(inp)
x = base(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
out = Dense(len(classes), activation="softmax")(x)

model = Model(inp, out)

# Step 6: Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 7: Train model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Step 8: Fine-tune the top VGG16 layers
base.trainable = True

for layer in base.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2
)

# Step 9: Evaluate final model
loss, accuracy = model.evaluate(val_ds)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Step 10: Visualize prediction
images, labels = next(iter(val_ds))

pred = model.predict(images[:1], verbose=0)
p = pred[0].argmax()

plt.imshow(images[0].numpy().astype("uint8"))
plt.title(f"Actual: {classes[labels[0]]}, Predicted: {classes[p]}")
plt.axis("off")
plt.show()
```
