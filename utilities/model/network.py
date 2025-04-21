"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Gesture Classification Module
-------------------------------------
This module implements a neural network for classifying gestures based on input features.
It provides a simple feedforward architecture that can be trained to recognize different
gesture patterns from preprocessed sensor data. The parameters were chosen through a parameter
search process using Optuna.

The implementation uses PyTorch for creating and training the neural network model.
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim

class GestureClassifier(nn.Module):
    """
    A neural network model for classifying gestures.
    
    This classifier uses a simple feedforward architecture with batch normalization,
    tanh activation functions, and dropout for regularization. It is designed to 
    process fixed-length feature vectors extracted from gesture data.
    
    Attributes:
        net (nn.Sequential): The sequential neural network layers.
    
    Args:
        input_dim (int, optional): Dimension of the input feature vector. Defaults to 63.
        num_classes (int, optional): Number of gesture classes to predict. Defaults to 8.
    """
    
    def __init__(self, input_dim=63, num_classes=8):
        """
        A simple feedforward network for gesture classification.
        """
        super(GestureClassifier, self).__init__()
        
        # Define the neural network architecture as a sequential model
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.3966),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
                containing the feature vectors to classify.
                
        Returns:
            torch.Tensor: Raw logits output from the network, shape (batch_size, num_classes).
        """
        
        return self.net(x)
