import tensorflow as tf
import os
from utils import get_project_root

def convert_to_tflite(model, X_train, output_path=None):
    if output_path is None:
        output_path = os.path.join(get_project_root(), "models", "fire_model_int8.tflite")
    
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ TFLite model saved at {output_path}!")
    return output_path
