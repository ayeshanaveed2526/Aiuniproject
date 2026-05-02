import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.applications import VGG16, EfficientNetB3, DenseNet121, ResNet50, ResNet152V2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- NOTE: You need to define these functions or data for the script to run ---
def create_model(base_model):
    # Placeholder: Define your model architecture here
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1)(x) # Assuming regression based on your metrics
    return Model(inputs=base_model.input, outputs=predictions)

def custom_loss(y_true, y_pred):
    # Placeholder: Define your custom loss here
    return Huber()(y_true, y_pred)

# Load your data here
# train_data = pd.read_csv("your_data.csv") 
# test_data = pd.read_csv("your_test_data.csv")

# ------------------------------------------------------------------------------

# Define the base models
base_models = {
    'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'EfficientNet': EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'DenseNet': DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'ResNet152V2': ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
}

# Set up the K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store training and validation losses
avg_train_losses = []
avg_val_losses = []
df_predictions = pd.DataFrame()

# Loop over each base model name
for base_model_name in base_models.keys():
    base_model = base_models[base_model_name]
    train_losses_fold = []
    val_losses_fold = []
    test_losses_fold = []
    test_mae_fold = []
    fold_predictions = []

    # Loop over each fold
    # WARNING: This requires 'train_data' to be defined
    for fold, (train_index, val_index) in enumerate(kfold.split(train_data), 1):
        print(f"Fold: {fold}  Model: {base_model_name}")
        
        train_data_fold = train_data.iloc[train_index]
        val_data_fold = train_data.iloc[val_index]

        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        datagen.seed = 42

        train_generator = datagen.flow_from_dataframe(
            dataframe=train_data_fold,
            x_col='Left_image',
            y_col='normalized_score',
            batch_size=8,
            seed=42,
            shuffle=True,
            class_mode='raw',
            target_size=(224, 224)
        )

        val_generator = datagen.flow_from_dataframe(
            dataframe=val_data_fold,
            x_col='Left_image',
            y_col='normalized_score',
            batch_size=8,
            seed=42,
            shuffle=True,
            class_mode='raw',
            target_size=(224, 224)
        )

        model = create_model(base_model)
        model.compile(optimizer='adam', loss=custom_loss, metrics=[MeanAbsoluteError()])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=50,
            verbose=2,
            callbacks=[early_stopping]
        )

        train_losses_fold.append(history.history['loss'])
        val_losses_fold.append(history.history['val_loss'])

        # NOTE: Update this path for local Windows use
        model_save_path = f'run1_{base_model_name}_fold{fold}.h5'
        model.save_weights(model_save_path)

        test_generator = datagen.flow_from_dataframe(
            dataframe=test_data,
            x_col='Left_image',
            y_col='normalized_score',
            batch_size=len(test_data),
            seed=42,
            shuffle=False,
            class_mode='raw',
            target_size=(224, 224)
        )

        test_images, test_labels = test_generator.next()
        test_loss, test_mae = model.evaluate(test_images, test_labels, verbose=2)
        
        test_losses_fold.append(test_loss)
        test_mae_fold.append(test_mae)

        test_predictions = model.predict(test_images)
        
        fold_predictions_fold = pd.DataFrame({
            'Predicted': test_predictions.flatten(),
            'Real': test_labels
        })
        fold_predictions.append(fold_predictions_fold)
        print(f"Complete training for Fold {fold}  Model: {base_model_name}")

    avg_train_losses.append(np.mean(train_losses_fold, axis=0))
    avg_val_losses.append(np.mean(val_losses_fold, axis=0))
    
    # Save results
    pd.DataFrame(train_losses_fold).to_csv(f'run1_{base_model_name}_train_losses.csv', index=False)
    pd.DataFrame(val_losses_fold).to_csv(f'run1_{base_model_name}_val_losses.csv', index=False)
    
    df_predictions_fold = pd.concat(fold_predictions)
    df_predictions[base_model_name] = df_predictions_fold['Predicted'].values
    
    df_performance_fold = pd.DataFrame({
        'Fold': range(1, len(test_losses_fold) + 1),
        'Test Loss': test_losses_fold,
        'Test MAE': test_mae_fold
    })
    df_performance_fold.to_csv(f'run1_{base_model_name}_performance.csv', index=False)
