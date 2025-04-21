"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Hyperparameter Optimization Module for Gesture Classification
------------------------
This module implements hyperparameter tuning for a gesture classification model
using Optuna. It includes:
1. A configurable neural network model for gesture classification
2. Functions for training and evaluating the model
3. An Optuna study to find optimal hyperparameters
4. CSV logging of trial results

The optimization process uses early stopping to prevent overfitting and
saves the best model configuration found during the trials.

This file was used to determine the best hyperparameters for the model at different times of the
data collection process. As more data was added, some parameters were fixed based on previous trials, 
and some were optimized, leading to a more efficient search space.
"""

# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import optuna
from train import GestureDataset  # Your custom dataset
from tqdm import tqdm
import time
import csv

# ----------------- Model -------------------
class GestureClassifier(nn.Module):
    """
    Neural network model for classifying hand gestures from landmark data.
    
    This model consists of a configurable feedforward neural network with
    options for different activation functions, dropout rates, and batch
    normalization.
    
    Args:
        input_dim (int): Dimension of input feature vector. Defaults to 63 (21 landmarks × 3 coordinates).
        hidden_dim1 (int): Size of first hidden layer. Defaults to 128.
        hidden_dim2 (int): Size of second hidden layer. Defaults to 64.
        dropout_rate (float): Probability of dropout for regularization. Defaults to 0.3.
        activation (nn.Module): Activation function to use. Defaults to nn.ReLU.
        use_batchnorm (bool): Whether to use batch normalization. Defaults to True.
        num_classes (int): Number of gesture classes to predict. Defaults to 8.
    """
    
    def __init__(
        self,
        input_dim=63,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout_rate=0.3,
        activation=nn.ReLU,
        use_batchnorm=True,
        num_classes=8,
    ):
        super(GestureClassifier, self).__init__()
        
        # Build the network architecture as a sequence of layers
        layers = [nn.Linear(input_dim, hidden_dim1)]  # Input layer

        # Add batch normalization if specified
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim1))

        # Add activation, dropout, and remaining layers
        layers.append(activation())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim1, hidden_dim2))  # Hidden layer
        layers.append(activation())
        layers.append(nn.Linear(hidden_dim2, num_classes))  # Output layer

        # Create the sequential model from the layer list
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Model predictions of shape (batch_size, num_classes)
        """
        
        return self.net(x)

# ----------------- Optimizer Helper -------------------
def get_optimizer(optimizer_name, model_params, lr):
    """
    Creates the specified optimizer with the given parameters.
    
    Args:
        optimizer_name (str): Name of the optimizer ('adam', 'sgd')
        model_params (iterable): Model parameters to optimize
        lr (float): Learning rate
        
    Returns:
        torch.optim.Optimizer: The initialized optimizer
        
    Raises:
        ValueError: If an unsupported optimizer name is provided
    """
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# ----------------- Training and Evaluation -------------------
def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, device, max_epochs=10, patience=3):
    """
    Trains the model and evaluates it on the validation set, with early stopping.
    
    Args:
        model (nn.Module): The model to train
        optimizer (torch.optim.Optimizer): The optimizer for training
        criterion (nn.Module): Loss function
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        device (torch.device): Device to use for training (CPU/GPU)
        max_epochs (int): Maximum number of training epochs. Defaults to 10.
        patience (int): Number of epochs with no improvement before early stopping. Defaults to 3.
        
    Returns:
        tuple: (best_val_acc, best_epoch) - Best validation accuracy and the epoch it was achieved
    """
    
    model.to(device)
    
    best_val_acc = 0.0
    best_epoch = 0
    no_improvement_epochs = 0
    
    for epoch in range(max_epochs):
        model.train()  # Set the model to training mode
        
        # Wrap the training loop with tqdm for progress bar visualization
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{max_epochs}") as pbar:
            running_loss = 0.0
            correct = 0
            total = 0

            for x_batch, y_batch in pbar:
                
                # Send batch to device (GPU/CPU)
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()  # Clear the gradients
                out = model(x_batch)  # Forward pass
                
                loss = criterion(out, y_batch)  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                # Update running statistics for progress display
                running_loss += loss.item()
                _, predicted = out.max(1)  # Get the predicted class
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                # Update progress bar description with the loss and accuracy
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), acc=100 * correct / total)

        # Evaluate model on validation set after each epoch
        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Check if the validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improvement_epochs = 0
            
            # Save the model with the best validation accuracy
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improvement_epochs += 1

        # If no improvement for 'patience' epochs, stop early
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break
    
    return best_val_acc, best_epoch


def evaluate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    
    Args:
        model (nn.Module): Model to evaluate
        val_loader (DataLoader): DataLoader for validation data
        device (torch.device): Device to use for evaluation
        
    Returns:
        float: Validation accuracy as a percentage
    """
    
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            _, predicted = out.max(1)  # Get predicted class indices
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    # Calculate the accuracy
    val_acc = 100 * correct / total
    return val_acc

# File path for logging trial results
LOG_FILE = "optuna_results.csv"

def log_trial_to_csv(trial_number, config, best_val_acc, time_elapsed):
    """
    Logs the trial results to a CSV file.
    
    Args:
        trial_number (int): Current trial number
        config (dict): Configuration parameters used for the trial
        best_val_acc (float): Best validation accuracy achieved
        time_elapsed (float): Time taken for the trial in seconds
    """
    
    # Open file in append mode
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["trial", *config.keys(), "accuracy", "time_seconds"])
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
            
        # Prepare and write the row data
        row = {
            "trial": trial_number,
            **config,
            "accuracy": round(best_val_acc, 4),
            "time_seconds": round(time_elapsed, 2),
        }
        writer.writerow(row)

# ----------------- Optuna Objective -------------------
def objective(trial):
    """
    Objective function for Optuna optimization.
    
    This function:
    1. Creates a model with hyperparameters suggested by Optuna
    2. Trains and evaluates the model
    3. Logs results
    4. Returns the validation accuracy for Optuna to maximize
    
    Args:
        trial (optuna.trial.Trial): Optuna trial object
        
    Returns:
        float: Best validation accuracy achieved in this trial
    """
    
    # Define hyperparameters to optimize (some are fixed based on previous trials with less data)
    config = {
        "hidden_dim1": 128,  # fixed from best trial
        "hidden_dim2": 64,   # fixed
        "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.5),
        "activation": trial.suggest_categorical("activation", [nn.ReLU, nn.Tanh]),
        "use_batchnorm": True,  # fixed
        "lr": trial.suggest_float("lr", 0.001, 0.1),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "batch_size": trial.suggest_categorical("batch_size", [64]),
    }

    # Display trial configuration
    print(f"\n🔧 Starting trial {trial.number} with configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model with the trial's hyperparameters
    model = GestureClassifier(
        hidden_dim1=config["hidden_dim1"],
        hidden_dim2=config["hidden_dim2"],
        dropout_rate=config["dropout_rate"],
        activation=config["activation"],
        use_batchnorm=config["use_batchnorm"],
    )

    # Set up optimizer and loss function
    optimizer = get_optimizer(config["optimizer"], model.parameters(), config["lr"])
    criterion = nn.CrossEntropyLoss()

    # Create data loaders with the trial's batch size
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    # Determine the appropriate device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 
                          "cuda" if torch.cuda.is_available() else "cpu")

    # Train and evaluate the model, timing the process
    start_time = time.time()
    best_val_acc, best_epoch = train_and_evaluate(
        model, optimizer, criterion, train_loader, val_loader, device, max_epochs=32, patience=5
    )
    time_elapsed = time.time() - start_time

    # Display results
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch+1}")

    # Log trial results to CSV
    log_trial_to_csv(trial.number, config, best_val_acc, time_elapsed)

    return best_val_acc

# ----------------- Main -------------------
def main():
    """
    Main function that sets up and runs the hyperparameter optimization.
    
    This function:
    1. Sets up the device (GPU/CPU)
    2. Loads and splits the dataset
    3. Creates and runs the Optuna study
    4. Displays the best trial results
    """
    
    global train_dataset, val_dataset, device  # Make these variables accessible to the objective function

    # Set up device for training (MPS for Apple Silicon, CUDA for NVIDIA GPU, or CPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using GPU")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up dataset paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(current_dir, "..", "training_data"))

    # Load and split dataset
    dataset = GestureDataset(dataset_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Display best trial results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    print("  Best params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()