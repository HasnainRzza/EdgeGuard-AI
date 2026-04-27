# =========================================
# FIRE DETECTION - EDGE OPTIMIZED SCRIPT
# =========================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# =========================================
# CONFIG
# =========================================
IMG_SIZE = 128
CLASSES = ["fire", "no_fire"]
ALPHA = 0.5

# 👉 CHANGE THIS PATH IF NEEDED
DATASET_PATH = r"fire_dataset"


# =========================================
# PREPROCESSING
# =========================================
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img.astype(np.float32)


# =========================================
# LOAD DATASET
# =========================================
def load_dataset(path):
    X, y = [], []

    folder_map = {
        "fire_images": 0,
        "non_fire_images": 1
    }

    for folder_name, label in folder_map.items():
        folder_path = os.path.join(path, folder_name)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"{folder_path} not found!")

        for file in tqdm(os.listdir(folder_path), desc=folder_name):
            img_path = os.path.join(folder_path, file)

            try:
                X.append(preprocess_image(img_path))
                y.append(label)
            except:
                continue

    return np.array(X), np.array(y)


# =========================================
# DATA AUGMENTATION
# =========================================
def get_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])


# =========================================
# MODEL BLOCK
# =========================================
def depthwise_block(x, filters, stride=1):
    filters = int(filters * ALPHA)

    x = layers.DepthwiseConv2D(3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    return x


# =========================================
# BUILD MODEL
# =========================================
def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = get_augmentation()(inputs)

    x = layers.Conv2D(int(32 * ALPHA), 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    x = depthwise_block(x, 32)
    x = depthwise_block(x, 64, stride=2)

    x = depthwise_block(x, 64)
    x = depthwise_block(x, 128, stride=2)

    x = depthwise_block(x, 128)
    x = depthwise_block(x, 256, stride=2)

    x = depthwise_block(x, 256)

    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================================
# TRAIN + EVALUATE
# =========================================
def train_and_evaluate():
    print("📦 Loading dataset...")
    X, y = load_dataset(DATASET_PATH)

    print("Dataset shape:", X.shape)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print("🧠 Building model...")
    model = build_model()
    model.summary()

    print("🚀 Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=16
    )

    print("📊 Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    print("💾 Saving model...")
    model.save("fire_model.h5")

    return model, X_train


# =========================================
# TFLITE CONVERSION
# =========================================
def convert_to_tflite(model, X_train):
    print("⚡ Converting to TFLite INT8...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data():
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1]]

    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open("fire_model_int8.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ TFLite model saved!")


# =========================================
# MAIN
# =========================================
def main():
    model, X_train = train_and_evaluate()
    convert_to_tflite(model, X_train)


if __name__ == "__main__":
    main()