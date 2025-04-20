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
        layers = [nn.Linear(input_dim, hidden_dim1)]

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim1))

        layers.append(activation())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim1, hidden_dim2))
        layers.append(activation())
        layers.append(nn.Linear(hidden_dim2, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------- Optimizer Helper -------------------
def get_optimizer(optimizer_name, model_params, lr):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(model_params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# ----------------- Training and Evaluation -------------------

def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, device, max_epochs=10, patience=3):
    model.to(device)
    
    best_val_acc = 0.0
    best_epoch = 0
    no_improvement_epochs = 0
    
    for epoch in range(max_epochs):
        model.train()  # Set the model to training mode
        
        # Wrap the training loop with tqdm for progress bar
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

                # Update running stats
                running_loss += loss.item()
                _, predicted = out.max(1)  # Get the predicted class
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                # Update progress bar description with the loss and accuracy
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), acc=100 * correct / total)

        # Validation step after each epoch
        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Check if the validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improvement_epochs = 0
            # Save the best model state
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improvement_epochs += 1

        # If no improvement for 'patience' epochs, stop early
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break
    
    return best_val_acc, best_epoch


def evaluate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            _, predicted = out.max(1)  # Get the predicted class
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    # Calculate the accuracy
    val_acc = 100 * correct / total
    return val_acc

LOG_FILE = "optuna_results.csv"

def log_trial_to_csv(trial_number, config, best_val_acc, time_elapsed):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["trial", *config.keys(), "accuracy", "time_seconds"])
        if not file_exists:
            writer.writeheader()
        row = {
            "trial": trial_number,
            **config,
            "accuracy": round(best_val_acc, 4),
            "time_seconds": round(time_elapsed, 2),
        }
        writer.writerow(row)

# ----------------- Optuna Objective -------------------
def objective(trial):
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

    print(f"\n🔧 Starting trial {trial.number} with configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    model = GestureClassifier(
        hidden_dim1=config["hidden_dim1"],
        hidden_dim2=config["hidden_dim2"],
        dropout_rate=config["dropout_rate"],
        activation=config["activation"],
        use_batchnorm=config["use_batchnorm"],
    )

    optimizer = get_optimizer(config["optimizer"], model.parameters(), config["lr"])
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 
                          "cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    best_val_acc, best_epoch = train_and_evaluate(
        model, optimizer, criterion, train_loader, val_loader, device, max_epochs=32, patience=5
    )
    time_elapsed = time.time() - start_time

    print(f"Best Validation Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch+1}")

    # Log results
    log_trial_to_csv(trial.number, config, best_val_acc, time_elapsed)

    return best_val_acc

# ----------------- Main -------------------
def main():
    global train_dataset, val_dataset, device  # So objective() can see them

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using GPU")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(current_dir, "..", "training_data"))

    dataset = GestureDataset(dataset_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    print("  Best params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()