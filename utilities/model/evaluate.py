"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Gesture Recognition Real-time Evaluation Module
-------------------------
This module performs real-time hand gesture recognition using a webcam. It:
1. Loads a pre-trained gesture classification model
2. Captures video from the default camera
3. Processes each frame to detect hands using MediaPipe
4. Extracts hand landmarks and classifies the gesture
5. Displays the results in real-time with visual feedback

The system can recognize 8 different hand gestures and displays the 
recognized gesture name next to the detected hand.
"""

# Imports
import cv2
import mediapipe as mp
import torch
import numpy as np
from network import GestureClassifier
import os

# ----- Model Loading and Setup -----
# Define the number of gesture classes the model can recognize
num_classes = 8  # This will need adjustment as more gestures are added
model = GestureClassifier(num_classes=num_classes)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the trained model weights
weights_path = os.path.join(script_dir, "gesture_classifier_weights.pth")

# Load the pre-trained model weights
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode (disables dropout, etc.)

# ----- Class Mapping Setup -----
# Define a mapping from class indices to gesture names
# Note: This mapping must be updated when new gestures are added
class_names = {
    0: "grab_move_left", 
    1: "grab_move_right", 
    2: "hold_left", 
    3: "hold_right", 
    4: "scale_left_hand",
    5: "scale_right_hand", 
    6: "swipe_left_hand", 
    7: "swipe_right_hand"
}

# ----- MediaPipe Setup -----
# Initialize MediaPipe Hands for real-time hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # For video streams, not static images
    max_num_hands=2,          # Track up to 2 hands simultaneously
    min_detection_confidence=0.5,  # Minimum confidence for initial detection
    min_tracking_confidence=0.5    # Minimum confidence for tracking continuation
)
# Initialize MediaPipe drawing utilities for visualizing hand landmarks
mp_drawing = mp.solutions.drawing_utils

# ----- Camera Setup -----
# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)


# ----- Main Processing Loop -----
while True:
    
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame from BGR (OpenCV format) to RGB (MediaPipe format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # If hands are detected, process each hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the 3D coordinates of each landmark and flatten into a vector
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Add x, y, z coordinates
            
            # Convert to numpy array with appropriate data type
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Convert to PyTorch tensor and add batch dimension for model input
            input_tensor = torch.tensor(landmarks).unsqueeze(0)
            
            # Run inference with the model
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = model(input_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()  # Get class with highest score
            
            # Get the gesture name from the class index
            gesture_name = class_names.get(predicted_class, "Unknown")
            
            # Calculate position to display the gesture name (near the wrist)
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[0]  # Wrist is the first landmark
            cx, cy = int(wrist.x * w), int(wrist.y * h)  # Convert normalized coordinates to pixel coordinates
            
            # Add the gesture name text to the frame
            cv2.putText(frame, gesture_name, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Print the prediction to console for debugging
            print(f"Predicted Gesture: {gesture_name}")

    # Display the processed frame
    cv2.imshow("Gesture Recognition", frame)

    # Check for 'Esc' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ----- Cleanup -----
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
