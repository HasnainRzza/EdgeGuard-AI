import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_ingestion import load_dataset, get_data_ingestion_config
from data_transformation import transform_data, get_augmentation
from model_factory import build_model
from model_trainer import train_model
from model_evaluation import evaluate_model
from model_pusher import convert_to_tflite
from monitoring_service import log_to_prometheus
from performance_analysis import analyze_performance

def run_pipeline():
    # 1. Ingestion
    print("📦 Loading dataset...")
    data_path = get_data_ingestion_config()
    image_paths, labels = load_dataset(data_path)
    
    # 2. Splitting
    X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(
        X_temp_paths, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # 3. Transformation
    print("🛠 Transforming data...")
    X_train, y_train = transform_data(X_train_paths, y_train)
    X_val, y_val = transform_data(X_val_paths, y_val)
    X_test, y_test = transform_data(X_test_paths, y_test)
    
    # 4. Build Model
    print("🧠 Building model...")
    augmentation = get_augmentation()
    model = build_model(augmentation_layer=augmentation)
    total_params = model.count_params()
    
    # 5. Train
    print("🚀 Training...")
    model, model_path = train_model(model, X_train, y_train, X_val, y_val)
    
    # 6. Evaluate
    print("📊 Evaluating...")
    eval_metrics = evaluate_model(model, X_test, y_test)
    eval_metrics['total_params'] = total_params
    
    # 7. Statistical Analysis
    analyze_performance(eval_metrics)
    
    # 8. Prometheus Monitoring
    log_to_prometheus(eval_metrics)
    
    # 9. Push / Convert
    print("💾 Saving and Converting to TFLite...")
    tflite_path = convert_to_tflite(model, X_train)
    
    print(f"✅ Pipeline completed successfully! TFLite model: {tflite_path}")

if __name__ == "__main__":
    run_pipeline()
