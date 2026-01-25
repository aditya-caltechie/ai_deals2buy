"""
Backward-compatible shim.

Neural network model + inference moved to `models.neural_network`.
"""

from models.neural_network import (  # noqa: F401
    DeepNeuralNetwork,
    DeepNeuralNetworkInference,
    ResidualBlock,
)
