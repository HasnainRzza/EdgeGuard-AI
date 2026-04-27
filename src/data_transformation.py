import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils import load_config

config = load_config()
IMG_SIZE = config['training']['img_size']

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img.astype(np.float32)

def transform_data(image_paths):
    """Processes a list of image paths into a numpy array of images."""
    X = []
    for path in image_paths:
        try:
            X.append(preprocess_image(path))
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
    return np.array(X)

def get_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
