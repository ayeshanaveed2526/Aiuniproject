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

# 6. Training
print(f"Starting training for classes: {SELECTED_CLASSES}")
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Note: This will only run if TensorFlow is installed and dataset is complete
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=20,
#     callbacks=[early_stopping],
#     verbose=1
# )

print("\nModel is ready. Classes detected:", train_generator.class_indices)
