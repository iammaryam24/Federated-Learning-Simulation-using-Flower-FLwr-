"""
Federated Learning Simulation using Flower
A comparative study of Federated vs Centralized Learning
"""

__version__ = "1.0.0"
__author__ = "Team Fireweed"

from .data_loader import FederatedDataLoader
from .model import FederatedModel
from .utils import setup_logging, calculate_metrics