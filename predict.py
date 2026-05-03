import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# 1. Load the trained model
model_path = 'flower_classifier_model.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please run deep_learning_model.py first.")
    sys.exit()

model = tf.keras.models.load_model(model_path)

# 2. Define classes (Must match the training order)
CLASSES = ['bougainvillea', 'daisies', 'tulip']
IMG_SIZE = (224, 224)

def predict_flower(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale like training

    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100

    print(f"\n--- Prediction Result ---")
    print(f"Flower: {CLASSES[class_idx]}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"-------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_flower(sys.argv[1])
