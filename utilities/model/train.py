"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Gesture Recognition Data Processing and Training Module
----------------------
This module handles the complete pipeline for training a gesture recognition model:
1. Data loading from image files containing hand gestures
2. Feature extraction using MediaPipe Hands
3. Dataset creation and management
4. Model training and validation
5. Model saving

The system extracts 3D landmarks from hand images and uses them to train a neural network
classifier that can recognize different gesture classes.
"""

# Imports
import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from network import GestureClassifier

# ----- A function to extract landmarks from an image -----
def extract_landmarks(image_path, hands):
    """
    Extracts hand landmarks from an image using MediaPipe.
    
    Args:
        image_path (str): Path to the input image file.
        hands (mp.solutions.hands.Hands): Initialized MediaPipe Hands object.
        
    Returns:
        np.ndarray: A 63-dimensional vector (21 landmarks × 3 coordinates) representing 
                   the hand landmarks. Returns a zero vector if no hand is detected.
    """
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        # Return zeros if image cannot be read
        return np.zeros(63, dtype=np.float32)
    
    # Convert BGR to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect hands and extract landmarks
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # If at least one hand is detected, use the first one
        landmarks = results.multi_hand_landmarks[0]
        # Extract x, y, z coordinates for each landmark and flatten into a 1D array
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32).flatten()
    else:
        # Return zeros if no hand is detected
        return np.zeros(63, dtype=np.float32)

# ----- Custom Dataset Class -----
class GestureDataset(Dataset):
    """
    PyTorch Dataset for loading and processing hand gesture images.
    
    This dataset expects a directory structure where each subdirectory represents
    a gesture class and contains images of that gesture:
      root_dir/
        gesture1/
          image1.jpg
          image2.jpg
          ...
        gesture2/
          image1.jpg
          ...
    
    The dataset uses MediaPipe Hands to extract 3D landmark features from each image.
    
    Attributes:
        samples (list): List of tuples (image_path, label_index).
        labels (list): List of label names (directory names).
        hands (mp.solutions.hands.Hands): MediaPipe Hands processor.
    """
    
    def __init__(self, root_dir):
        """
        Initialize the GestureDataset.
        
        Args:
            root_dir (str): Root directory containing class subdirectories with images.
        """
        
        self.samples = []
        self.labels = []
        
        # Get sorted list of subdirectory names as labels (alphabetical order)
        self.labels = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Create a mapping from label names to indices
        label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Collect all valid image files with their labels
        for label in self.labels:
            folder = os.path.join(root_dir, label)
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder, filename), label_to_idx[label]))

        # Initialize MediaPipe Hands with static image mode (optimized for images, not video)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,  # We only look for one hand
            min_detection_confidence=0.5  # Minimum confidence for detection
        )

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (landmarks_tensor, label_tensor) where landmarks_tensor is a
                  tensor of shape (63,) and label_tensor is a scalar tensor.
        """
        
        # Get image path and label
        img_path, label = self.samples[idx]
        
        # Extract landmarks from the image
        landmarks = extract_landmarks(img_path, self.hands)
        
        # Convert to PyTorch tensors
        return torch.tensor(landmarks), torch.tensor(label)

# ----- Main Training Loop -----
def main():
    """
    Main function to run the training pipeline.
    
    This function:
    1. Sets up the dataset and dataloaders
    2. Initializes the model, loss function, and optimizer
    3. Trains the model for a fixed number of epochs
    4. Evaluates model performance on a validation set
    5. Saves the trained model weights
    """
    
    # Path to dataset folder - subfolders are going to get read as labels here
    # NOTE: Once the model trains, everytime we add a new subfolder, we need to rearrange 
    # the class dictionary in evaluate.py because it needs to follow alphabetical order 
    # to correctly display labels
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "..", "training_data")
    dataset_dir = os.path.abspath(dataset_dir)

    # Create dataset
    dataset = GestureDataset(dataset_dir)
    
    # Split dataset into training and validation sets (80% / 20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Initialize model with correct number of output classes
    num_classes = len(dataset.labels)
    model = GestureClassifier(num_classes=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0019)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        
        # Set model to training mode
        model.train()
        running_loss = 0.0
        
        # Train on batches with progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad() # Zero the gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            running_loss += loss.item() * inputs.size(0) # Accumulate loss
        epoch_loss = running_loss / train_size # Average loss for the epoch
        print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

        # Validation step
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs) # Forward pass
                _, preds = torch.max(outputs, 1) # Get predicted classes
                total += labels.size(0) # Total samples
                correct += (preds == labels).sum().item() # Correct predictions
        test_acc = correct / total # Calculate accuracy
        print(f"Epoch {epoch+1} Validation Accuracy: {test_acc*100:.2f}%")

    # Save the trained model weights
    save_path = os.path.join(current_dir, "gesture_classifier_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete and model saved to {save_path}")

if __name__ == "__main__":
    main()
