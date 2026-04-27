import numpy as np
import yaml
from utils import load_config

config = load_config()

HISTORICAL_MODELS = {
    "VGG-like (Initial)": 58370850,
    "MobileNet-V1 (Base)": 676878
}

def analyze_performance(current_metrics):
    """
    Performs statistical comparison and efficiency analysis.
    """
    current_params = current_metrics.get('total_params', 0)
    
    print("\n📈 --- Statistical Performance Analysis ---")
    
    for name, params in HISTORICAL_MODELS.items():
        reduction = (1 - (current_params / params)) * 100
        print(f"🔹 Comparison with {name}:")
        print(f"   - Parameter Reduction: {reduction:.2f}%")
        print(f"   - Compression Factor: {params / current_params:.2f}x")
    
    # Calculate Parameter Efficiency (Accuracy per Million Parameters)
    efficiency = current_metrics['accuracy'] / (current_params / 1e6)
    print(f"\n🚀 Efficiency Metric:")
    print(f"   - Accuracy per Million Params: {efficiency:.2f}")
    
    # Latency Analysis
    latency = current_metrics['latency_ms']
    print(f"\n⏱️ Latency Analysis:")
    if latency < 5:
        print(f"   - Status: Ultra-Low Latency (Edge Ready)")
    elif latency < 20:
        print(f"   - Status: Low Latency (Mobile Ready)")
    else:
        print(f"   - Status: Moderate Latency")
    
    print("-------------------------------------------\n")

if __name__ == "__main__":
    # Mock data for testing
    mock_metrics = {
        'total_params': 40530,
        'accuracy': 0.95,
        'latency_ms': 2.5
    }
    analyze_performance(mock_metrics)
