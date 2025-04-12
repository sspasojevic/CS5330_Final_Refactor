import cv2
import mediapipe as mp
import torch
import numpy as np
from network import GestureClassifier
import os

# Load your trained model weights
num_classes = 8  # We will adjust trhis as we increase the number of gestures
model = GestureClassifier(num_classes=num_classes)

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, "gesture_classifier_weights.pth")

model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Define a mapping from class indices to gesture names
class_names = {0: "grab_move_left", 1: "grab_move_right", 2: "hold_left", 3: "hold_right", 4: "scale_left_hand",
               5: "scale_right_hand", 6: "swipe_left_hand", 7: "swipe_right_hand"} # This will also get adjusted as we add more gestures

# Set up MediaPipe Hands for video stream (for multi-hand detection)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Open video stream using OpenCV
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract landmarks and flatten them into a 63-dimensional vector
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks, dtype=np.float32)
            # Convert to PyTorch tensor and add batch dimension
            input_tensor = torch.tensor(landmarks).unsqueeze(0)
            # Run inference
            with torch.no_grad():
                outputs = model(input_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
            gesture_name = class_names.get(predicted_class, "Unknown")
            # Get position to display the gesture (using wrist landmark)
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, gesture_name, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Predicted Gesture: {gesture_name}")

    # Display the video stream
    cv2.imshow("Gesture Recognition", frame)

    # Press 'Esc' key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
