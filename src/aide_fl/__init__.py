"""
AIDE-FL: Secure Federated Learning Framework
"""

__version__ = "0.4.0"
__author__ = "IDLab-Discover"

# Export key classes/functions for easier imports
from aide_fl.dataset_cache.api import DatasetCache
from aide_fl.model_server.api import ModelServer
from aide_fl.experiments import apfed  # Enable import of experiments
