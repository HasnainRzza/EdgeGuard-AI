import numpy as np
import time
from sklearn.metrics import classification_report
from utils import load_config
import mlflow

config = load_config()

def evaluate_model(model, X_test, y_test):
    print("📊 Evaluating...")
    
    # 1. Standard Metrics
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    
    # 2. Latency and Throughput
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    num_samples = len(X_test)
    
    latency_ms = (total_time / num_samples) * 1000
    throughput = num_samples / total_time
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Inference Latency: {latency_ms:.2f} ms/image")
    print(f"Throughput: {throughput:.2f} images/sec")

    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=config['training']['classes']
    )
    
    # Log to MLflow
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("latency_ms", latency_ms)
    mlflow.log_metric("throughput", throughput)
    
    return {
        "accuracy": acc,
        "loss": loss,
        "latency_ms": latency_ms,
        "throughput": throughput,
        "report": report
    }
