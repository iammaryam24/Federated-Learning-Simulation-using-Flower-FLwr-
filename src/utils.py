"""
Utility functions for Federated Learning project
"""

import logging
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Log file path (optional)
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def calculate_metrics(metrics_history, metric_name='accuracy'):
    """
    Calculate summary metrics from training history
    
    Args:
        metrics_history: List of metric dictionaries
        metric_name: Name of the metric to analyze
        
    Returns:
        Dictionary of summary metrics
    """
    if not metrics_history:
        return {}
    
    values = [m.get(metric_name, 0) for m in metrics_history]
    
    return {
        'final': values[-1] if values else 0,
        'max': max(values) if values else 0,
        'min': min(values) if values else 0,
        'mean': np.mean(values) if values else 0,
        'std': np.std(values) if values else 0,
        'num_rounds': len(values)
    }

def save_experiment_results(results, experiment_name, output_dir="results"):
    """
    Save experiment results to file
    
    Args:
        results: Dictionary of results
        experiment_name: Name of the experiment
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dumps(results, f, indent=2, default=str)
    
    return filepath

def create_comparison_dataframe(federated_metrics, centralized_metrics):
    """
    Create a comparison DataFrame for federated vs centralized
    
    Args:
        federated_metrics: Federated learning metrics
        centralized_metrics: Centralized learning metrics
        
    Returns:
        Pandas DataFrame with comparison
    """
    comparison_data = {
        'Metric': ['Final Accuracy', 'Training Time (s)', 'Rounds/Epochs'],
        'Federated Learning': [
            federated_metrics.get('final_accuracy', 0),
            federated_metrics.get('training_time', 0),
            len(federated_metrics.get('history', []))
        ],
        'Centralized Learning': [
            centralized_metrics.get('final_accuracy', 0),
            centralized_metrics.get('training_time', 0),
            centralized_metrics.get('epochs', 0)
        ]
    }
    
    return pd.DataFrame(comparison_data)