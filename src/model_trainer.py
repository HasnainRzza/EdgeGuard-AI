import mlflow
import mlflow.tensorflow
import os
import psutil
import platform
import numpy as np
import tensorflow as tf
from utils import load_config, get_project_root

config = load_config()

def get_system_info():
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "ram": f"{round(psutil.virtual_memory().total / (1024.0 **3))} GB",
        "python_version": platform.python_version()
    }
    return info

def train_model(model, X_train, y_train, X_val, y_val):
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Get model parameters
    trainable_params = int(sum([np.prod(v.shape) for v in model.trainable_variables]))
    total_params = model.count_params()

    with mlflow.start_run():
        # Log parameters & system info
        mlflow.log_params(config['training'])
        mlflow.log_params(get_system_info())
        mlflow.log_param("trainable_params", trainable_params)
        mlflow.log_param("total_params", total_params)

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size']
        )

        # Log metrics
        for i in range(len(history.history['accuracy'])):
            mlflow.log_metric("accuracy", history.history['accuracy'][i], step=i)
            mlflow.log_metric("loss", history.history['loss'][i], step=i)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][i], step=i)
            mlflow.log_metric("val_loss", history.history['val_loss'][i], step=i)

        # Save model
        model_dir = os.path.join(get_project_root(), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "fire_model.h5")
        model.save(model_path)
        
        # Log model to MLflow
        mlflow.tensorflow.log_model(model, artifact_path="model")

    return model, model_path
