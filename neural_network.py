"""
Lightweight feed-forward neural network using pure NumPy.
Architecture: INPUT -> HIDDEN1 -> HIDDEN2 -> OUTPUT
"""

import numpy as np
from typing import List, Tuple, Optional
import os


class NeuralNetwork:
    """
    Simple feed-forward neural network with configurable layer sizes.
    Uses ReLU activation for hidden layers, tanh for output.
    """

    def __init__(self, layer_sizes: List[int]):
        """
        Initialize network with random weights.
        
        Args:
            layer_sizes: list of neuron counts per layer, e.g. [9, 16, 16, 2]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Xavier/Glorot initialization for better gradient flow
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        for i in range(self.num_layers - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: input array of shape (1, input_size) or (input_size,)
        
        Returns:
            output array of shape (1, output_size) or (output_size,)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        current = x
        
        for i in range(self.num_layers - 1):
            current = current @ self.weights[i] + self.biases[i]
            
            # ReLU for hidden layers, tanh for output
            if i < self.num_layers - 2:
                current = np.maximum(0, current)  # ReLU
            else:
                current = np.tanh(current)  # tanh output: [-1, 1]
        
        return current
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward pass."""
        return self.forward(x)
    
    def get_flat_weights(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1D array."""
        arrays = []
        for i in range(self.num_layers - 1):
            arrays.append(self.weights[i].flatten())
            arrays.append(self.biases[i].flatten())
        return np.concatenate(arrays)
    
    def set_flat_weights(self, flat: np.ndarray) -> None:
        """Set weights from a flat 1D array."""
        idx = 0
        for i in range(self.num_layers - 1):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            
            self.weights[i] = flat[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
            
            self.biases[i] = flat[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size
    
    def copy(self) -> 'NeuralNetwork':
        """Create a deep copy of this network."""
        nn = NeuralNetwork(self.layer_sizes)
        for i in range(self.num_layers - 1):
            nn.weights[i] = self.weights[i].copy()
            nn.biases[i] = self.biases[i].copy()
        return nn
    
    def mutate(self, rate: float, strength: float) -> None:
        """
        Mutate weights in place.
        
        Args:
            rate: probability of mutating each weight
            strength: standard deviation of mutation noise
        """
        for i in range(self.num_layers - 1):
            # Mutate weights
            mask_w = np.random.random(self.weights[i].shape) < rate
            noise_w = np.random.randn(*self.weights[i].shape) * strength
            self.weights[i] += mask_w * noise_w
            
            # Mutate biases
            mask_b = np.random.random(self.biases[i].shape) < rate
            noise_b = np.random.randn(*self.biases[i].shape) * strength
            self.biases[i] += mask_b * noise_b
    
    def save(self, filepath: str) -> None:
        """Save network weights to a file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        np.savez(filepath, *self.weights, *self.biases)
    
    def load(self, filepath: str) -> None:
        """Load network weights from a file."""
        data = np.load(filepath)
        arrays = list(data.values())
        
        w_idx = 0
        b_idx = self.num_layers - 1  # biases start after all weights
        
        for i in range(self.num_layers - 1):
            self.weights[i] = arrays[w_idx]
            w_idx += 1
        
        for i in range(self.num_layers - 1):
            self.biases[i] = arrays[b_idx]
            b_idx += 1
    
    def get_weight_shape_info(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get shapes of all weight matrices and bias vectors."""
        info = []
        for i in range(self.num_layers - 1):
            info.append((
                self.weights[i].shape,
                self.biases[i].shape
            ))
        return info


def crossover(parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
    """
    Create offspring by blending parent weights.
    Uses uniform crossover: each weight randomly from one parent.
    
    Args:
        parent1: first parent network
        parent2: second parent network
    
    Returns:
        new child network
    """
    child = parent1.copy()
    
    for i in range(child.num_layers - 1):
        # Uniform crossover mask
        mask_w = np.random.random(child.weights[i].shape) > 0.5
        child.weights[i] = np.where(mask_w, parent1.weights[i], parent2.weights[i])
        
        mask_b = np.random.random(child.biases[i].shape) > 0.5
        child.biases[i] = np.where(mask_b, parent1.biases[i], parent2.biases[i])
    
    return child
