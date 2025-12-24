"""
Models Factory Package

This package implements the Factory Pattern to decouple model instantiation 
from the main execution logic. It routes requests to specific architecture 
definitions based on the configuration provided.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .factory import get_model