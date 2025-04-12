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

# ----- Define a function to extract landmarks from an image -----
def extract_landmarks(image_path, hands):
    """
    Reads an image, processes it with MediaPipe Hands to extract landmarks,
    and returns a 63-dimensional vector (21 landmarks * 3 coordinates).
    If no hand is detected, returns a vector of zeros.
    """
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(63, dtype=np.float32)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        # Use the first detected hand.
        landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32).flatten()
    else:
        return np.zeros(63, dtype=np.float32)

# ----- Custom Dataset Class -----
class GestureDataset(Dataset):
    def __init__(self, root_dir):
        """
        Expects a folder structure:
          root_dir/
            label1/
              image1.jpg
              image2.jpg
              ...
            label2/
              image1.jpg
              ...
        It uses MediaPipe Hands to extract landmark features.
        """
        self.samples = []
        self.labels = []
        # Get sorted list of subfolder names as labels
        self.labels = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        for label in self.labels:
            folder = os.path.join(root_dir, label)
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder, filename), label_to_idx[label]))

        # Initialize MediaPipe Hands once (static_image_mode=True for images)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        landmarks = extract_landmarks(img_path, self.hands)
        return torch.tensor(landmarks), torch.tensor(label)

# ----- Main Training Loop -----
def main():
    # Path to dataset folder - subfolders are ghoing to get read as labels here
    # Once the model trains, everytime we add a new subfolder, we need to rearrange the class dictionay in evaluate.py because it needs to follow alphabetical order to correclty display labels
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "..", "training_data")
    dataset_dir = os.path.abspath(dataset_dir)

    # Create dataset and DataLoader
    dataset = GestureDataset(dataset_dir)
    # For simplicity, use 80% for training and 20% for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    num_classes = len(dataset.labels)
    model = GestureClassifier(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / train_size
        print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct / total
        print(f"Epoch {epoch+1} Validation Accuracy: {val_acc*100:.2f}%")

    # Save the trained model weights
    save_path = os.path.join(current_dir, "gesture_classifier_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete and model saved to {save_path}")

if __name__ == "__main__":
    main()
