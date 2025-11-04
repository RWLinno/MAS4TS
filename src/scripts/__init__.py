"""
Training and evaluation scripts for MAS4TS
"""

from .train_mas4ts import main as train_main
from .evaluate_mas4ts import main as evaluate_main

__all__ = ['train_main', 'evaluate_main']

