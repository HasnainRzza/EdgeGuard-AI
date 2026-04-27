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

def transform_data(image_paths, labels=None):
    """Processes a list of image paths into a numpy array of images.
    If labels are provided, filters the labels to match the successfully loaded images.
    """
    X = []
    y_filtered = []
    for i, path in enumerate(image_paths):
        try:
            img = preprocess_image(path)
            X.append(img)
            if labels is not None:
                y_filtered.append(labels[i])
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
            
    X = np.array(X)
    if labels is not None:
        y_filtered = np.array(y_filtered)
        assert len(X) == len(y_filtered), f"Data cardinality mismatch: X has {len(X)}, y has {len(y_filtered)}"
        return X, y_filtered
    return X

def get_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
