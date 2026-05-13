import os
import json
import numpy as np
import random
import shutil
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set random seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# 1. DATASET PATHS
ORIGINAL_DATASET  = r'dataset/flowers'
BALANCED_DATASET  = r'dataset/flowers_balanced'

TARGET_CLASSES    = ['bougainvillea', 'daisies', 'tulip']
SELECTED_CLASSES  = TARGET_CLASSES

IMG_SIZE          = (224, 224)
BATCH_SIZE        = 32
TARGET_SAMPLES    = 300 

def augment_directory(src_dir, dst_dir, target_count, augmentor):
    os.makedirs(dst_dir, exist_ok=True)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in images:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
    count = len(images)
    while count < target_count:
        src_img_name = random.choice(images)
        img = load_img(os.path.join(src_dir, src_img_name), target_size=IMG_SIZE)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        for batch in augmentor.flow(x, batch_size=1, save_to_dir=dst_dir, save_prefix='aug', save_format='jpeg'):
            count += 1
            break
        if count >= target_count: break
    print(f"  {os.path.basename(dst_dir):20s}: {count} images")

def copy_all(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in images:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
    print(f"  {'other':20s}: {len(images)} images")

print("\n[INFO] Building balanced dataset ...")
balance_augmentor = ImageDataGenerator(
    rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)

if os.path.exists(BALANCED_DATASET): shutil.rmtree(BALANCED_DATASET)

for cls in TARGET_CLASSES:
    augment_directory(os.path.join(ORIGINAL_DATASET, cls), os.path.join(BALANCED_DATASET, cls), TARGET_SAMPLES, balance_augmentor)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 3. DATA GENERATORS
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=35,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    BALANCED_DATASET, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES, class_mode='categorical', subset='training', seed=seed_value
)
val_generator = val_datagen.flow_from_directory(
    BALANCED_DATASET, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES, class_mode='categorical', subset='validation', seed=seed_value
)

# Persist class order for inference consistency
class_indices = train_generator.class_indices
ordered_classes = [None] * len(class_indices)
for name, idx in class_indices.items():
    ordered_classes[idx] = name
with open('flower_classes.json', 'w', encoding='utf-8') as f:
    json.dump(ordered_classes, f, ensure_ascii=True, indent=2)

from sklearn.utils.class_weight import compute_class_weight
class_weights_arr = compute_class_weight('balanced', classes=np.unique(train_generator.labels), y=train_generator.labels)
class_weights = dict(enumerate(class_weights_arr))

# 4. MODEL - MobileNetV2 (Fast)
def create_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)

model = create_model(len(SELECTED_CLASSES))
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6),
    ModelCheckpoint('flower_classifier_model.h5', monitor='val_accuracy', save_best_only=True)
]

print("\nTraining top layers...")
model.fit(train_generator, validation_data=val_generator, epochs=15, class_weight=class_weights, callbacks=callbacks)

print("\nFine-tuning...")
model.load_weights('flower_classifier_model.h5')

# Find MobileNetV2 base model by searching layers
mobilenet_base = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name.lower():
        mobilenet_base = layer
        break

if mobilenet_base:
    mobilenet_base.trainable = True
    # Freeze all but the last 50 layers of the base model
    for layer in mobilenet_base.layers[:-50]:
        layer.trainable = False
    print(f"  [INFO] Unfrozen last 50 layers of {mobilenet_base.name}")
else:
    print("  [WARN] Could not find MobileNetV2 base layer. Skipping fine-tune.")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
model.fit(train_generator, validation_data=val_generator, epochs=10, class_weight=class_weights, callbacks=callbacks)

# Final evaluation
print("\n[OK] Final Model Evaluation:")
loss, acc = model.evaluate(val_generator)
print(f"  Validation Accuracy: {acc*100:.2f}%")
print(f"  Validation Loss:     {loss:.4f}")
print("\n[OK] Model training complete and saved as 'flower_classifier_model.h5'")
