import torch
import torch.nn as nn
import torch.optim as optim

# ----- Define the PyTorch Model -----
class GestureClassifier(nn.Module):
    def __init__(self, input_dim=63, num_classes=2):
        """
        A simple feedforward network for gesture classification.
        """
        super(GestureClassifier, self).__init__()
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
        return self.net(x)
