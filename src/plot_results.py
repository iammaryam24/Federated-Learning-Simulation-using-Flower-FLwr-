"""
Visualization and plotting for Federated Learning results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class ResultPlotter:
    """Plot and visualize federated learning results"""
    
    def __init__(self, output_dir="results"):
        """
        Initialize plotter
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_accuracy_comparison(self, federated_history, centralized_history,
                                  save=True, show=False):
        """
        Plot accuracy comparison between federated and centralized
        
        Args:
            federated_history: Federated learning accuracy history
            centralized_history: Centralized learning accuracy history
            save: Save plot to file
            show: Display plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Federated learning accuracy
        if federated_history:
            rounds = [m['round'] for m in federated_history]
            fed_accuracy = [m.get('accuracy', 0) * 100 for m in federated_history]
            ax1.plot(rounds, fed_accuracy, 'b-o', linewidth=2, markersize=8)
            ax1.set_xlabel('Communication Round', fontsize=14)
            ax1.set_ylabel('Accuracy (%)', fontsize=14)
            ax1.set_title('Federated Learning Accuracy', fontsize=16)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 100])
        
        # Centralized learning accuracy
        if centralized_history:
            epochs = range(1, len(centralized_history) + 1)
            cent_accuracy = [m.get('accuracy', 0) * 100 for m in centralized_history]
            ax2.plot(epochs, cent_accuracy, 'r-s', linewidth=2, markersize=8)
            ax2.set_xlabel('Epoch', fontsize=14)
            ax2.set_ylabel('Accuracy (%)', fontsize=14)
            ax2.set_title('Centralized Learning Accuracy', fontsize=16)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 100])
        
        plt.suptitle('Federated vs Centralized Learning: Accuracy Comparison', 
                     fontsize=18, y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'accuracy_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_curves(self, federated_history, centralized_history,
                        save=True, show=False):
        """
        Plot loss curves comparison
        
        Args:
            federated_history: Federated learning loss history
            centralized_history: Centralized learning loss history
            save: Save plot to file
            show: Display plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Federated learning loss
        if federated_history:
            rounds = [m['round'] for m in federated_history]
            fed_loss = [m.get('loss', 0) for m in federated_history]
            ax1.plot(rounds, fed_loss, 'b-o', linewidth=2, markersize=8)
            ax1.set_xlabel('Communication Round', fontsize=14)
            ax1.set_ylabel('Loss', fontsize=14)
            ax1.set_title('Federated Learning Loss', fontsize=16)
            ax1.grid(True, alpha=0.3)
        
        # Centralized learning loss
        if centralized_history:
            epochs = range(1, len(centralized_history) + 1)
            cent_loss = [m.get('loss', 0) for m in centralized_history]
            ax2.plot(epochs, cent_loss, 'r-s', linewidth=2, markersize=8)
            ax2.set_xlabel('Epoch', fontsize=14)
            ax2.set_ylabel('Loss', fontsize=14)
            ax2.set_title('Centralized Learning Loss', fontsize=16)
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Federated vs Centralized Learning: Loss Comparison', 
                     fontsize=18, y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'loss_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_time_comparison(self, times_dict, save=True, show=False):
        """
        Plot training time comparison
        
        Args:
            times_dict: Dictionary with training times
            save: Save plot to file
            show: Display plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(times_dict.keys())
        times = list(times_dict.values())
        
        bars = ax.bar(methods, times, color=['blue', 'red', 'green'])
        ax.set_xlabel('Training Method', fontsize=14)
        ax.set_ylabel('Training Time (seconds)', fontsize=14)
        ax.set_title('Training Time Comparison', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f}s',
                   ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'time_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_strategy_comparison(self, fedavg_metrics, fedprox_metrics,
                                 save=True, show=False):
        """
        Plot comparison between FedAvg and FedProx strategies
        
        Args:
            fedavg_metrics: FedAvg metrics history
            fedprox_metrics: FedProx metrics history
            save: Save plot to file
            show: Display plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        if fedavg_metrics and fedprox_metrics:
            rounds_fedavg = [m['round'] for m in fedavg_metrics]
            acc_fedavg = [m.get('accuracy', 0) * 100 for m in fedavg_metrics]
            rounds_fedprox = [m['round'] for m in fedprox_metrics]
            acc_fedprox = [m.get('accuracy', 0) * 100 for m in fedprox_metrics]
            
            ax1.plot(rounds_fedavg, acc_fedavg, 'b-o', linewidth=2, 
                    markersize=8, label='FedAvg')
            ax1.plot(rounds_fedprox, acc_fedprox, 'g-s', linewidth=2, 
                    markersize=8, label='FedProx')
            ax1.set_xlabel('Communication Round', fontsize=14)
            ax1.set_ylabel('Accuracy (%)', fontsize=14)
            ax1.set_title('Strategy Accuracy Comparison', fontsize=16)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 100])
            
            # Loss comparison
            loss_fedavg = [m.get('loss', 0) for m in fedavg_metrics]
            loss_fedprox = [m.get('loss', 0) for m in fedprox_metrics]
            
            ax2.plot(rounds_fedavg, loss_fedavg, 'b-o', linewidth=2, 
                    markersize=8, label='FedAvg')
            ax2.plot(rounds_fedprox, loss_fedprox, 'g-s', linewidth=2, 
                    markersize=8, label='FedProx')
            ax2.set_xlabel('Communication Round', fontsize=14)
            ax2.set_ylabel('Loss', fontsize=14)
            ax2.set_title('Strategy Loss Comparison', fontsize=16)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('FedAvg vs FedProx: Strategy Comparison', 
                     fontsize=18, y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'strategy_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_summary_comparison(self, summary_data, save=True, show=False):
        """
        Plot summary comparison bar chart
        
        Args:
            summary_data: Dictionary with summary comparison
            save: Save plot to file
            show: Display plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = list(summary_data.keys())
        fed_values = [summary_data[m].get('fed_avg', 0) for m in metrics]
        fedprox_values = [summary_data[m].get('fed_prox', 0) for m in metrics]
        cent_values = [summary_data[m].get('centralized', 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        bars1 = ax.bar(x - width, fed_values, width, label='FedAvg', color='blue')
        bars2 = ax.bar(x, fedprox_values, width, label='FedProx', color='green')
        bars3 = ax.bar(x + width, cent_values, width, label='Centralized', color='red')
        
        ax.set_xlabel('Performance Metrics', fontsize=14)
        ax.set_ylabel('Values', fontsize=14)
        ax.set_title('Federated vs Centralized: Performance Summary', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'summary_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()