import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------
# Configuration
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ----------------------------
# Load Dataset
# ----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ----------------------------
# Class Names
# ----------------------------
class_names = train_ds.class_names

print("\nDetected Classes:")
for cls in class_names:
    print(cls)

# ----------------------------
# Performance Optimization
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ----------------------------
# CNN Model
# ----------------------------
model = models.Sequential([

    tf.keras.Input(shape=(224, 224, 3)),

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),

    layers.Dense(
        len(class_names),
        activation='softmax'
    )
])

# ----------------------------
# Compile Model
# ----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# Model Summary
# ----------------------------
model.summary()

# ----------------------------
# Train Model
# ----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# ----------------------------
# Save Model
# ----------------------------
model.save("models/plant_disease_model.keras")

print("\nModel saved successfully!")