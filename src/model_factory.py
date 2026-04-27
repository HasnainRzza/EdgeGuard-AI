import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_config

config = load_config()
IMG_SIZE = config['training']['img_size']
ALPHA = config['training']['alpha']
LEARNING_RATE = config['training']['learning_rate']

def depthwise_block(x, filters, stride=1):
    filters = int(filters * ALPHA)

    x = layers.DepthwiseConv2D(3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    return x

def build_model(augmentation_layer=None):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = inputs
    if augmentation_layer:
        x = augmentation_layer(x)

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
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
