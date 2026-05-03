import os
import kagglehub
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# 1. Local Dataset from Zip
dataset_path = r'dataset/flowers'
print("Using local dataset path:", dataset_path)

# 2. Define Classes (Updated to match folders in 'archive (2).zip')
SELECTED_CLASSES = ['bougainvillea', 'daisies', 'tulip']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 3. Data Preparation (Image Generators)
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 # Use 20% for validation
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES, # Only load these 3 classes
    class_mode='categorical',
    subset='training',
    seed=seed_value
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES,
    class_mode='categorical',
    subset='validation',
    seed=seed_value
)

# 4. Model Architecture (3 Classes)
def create_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x) # 3 outputs
    
    return Model(inputs=base_model.input, outputs=predictions)

model = create_model(len(SELECTED_CLASSES))

# 5. Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Initial Training (Frozen Base)
print(f"Starting initial training for classes: {SELECTED_CLASSES}")
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[early_stopping],
    verbose=1
)

# 7. Fine-Tuning (Unfreeze last block)
print("\nUnfreezing last block of VGG16 for fine-tuning...")

# Find the VGG16 base within our model
base_model = None
for layer in model.layers:
    if 'vgg16' in layer.name:
        base_model = layer
        break

if base_model:
    base_model.trainable = True
    # Freeze all layers except the last 4
    for layer in base_model.layers[:-4]:
        layer.trainable = False
else:
    # If VGG16 layers are top-level
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), # Very low learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting fine-tuning...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Save Model
model_name = "flower_classifier_model.h5"
model.save(model_name)
print(f"\nTraining complete! Model saved as {model_name}")

print("\nModel is ready. Classes detected:", train_generator.class_indices)
