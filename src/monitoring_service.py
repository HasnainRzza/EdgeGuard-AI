from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from utils import load_config

config = load_config()

def log_to_prometheus(metrics_dict):
    """
    Pushes metrics to Prometheus Pushgateway.
    Requires Pushgateway to be running (usually on port 9091).
    """
    registry = CollectorRegistry()
    
    # Define metrics
    g_accuracy = Gauge('model_accuracy', 'Accuracy of the trained model', registry=registry)
    g_loss = Gauge('model_loss', 'Loss of the trained model', registry=registry)
    g_latency = Gauge('model_latency_ms', 'Inference latency in milliseconds', registry=registry)
    g_throughput = Gauge('model_throughput_ips', 'Inference throughput in images per second', registry=registry)
    g_params = Gauge('model_parameters', 'Number of model parameters', registry=registry)

    # Set values
    g_accuracy.set(metrics_dict.get('accuracy', 0))
    g_loss.set(metrics_dict.get('loss', 0))
    g_latency.set(metrics_dict.get('latency_ms', 0))
    g_throughput.set(metrics_dict.get('throughput', 0))
    g_params.set(metrics_dict.get('total_params', 0))

    try:
        # Assuming Pushgateway is running at localhost:9091 (standard port)
        # You can update this in config.yaml
        push_gateway_url = config.get('monitoring', {}).get('pushgateway_url', 'localhost:9091')
        push_to_gateway(push_gateway_url, job='edgeguard_pipeline', registry=registry)
        print(f"✅ Metrics pushed to Prometheus at {push_gateway_url}")
    except Exception as e:
        print(f"⚠️ Could not push to Prometheus: {e}")
