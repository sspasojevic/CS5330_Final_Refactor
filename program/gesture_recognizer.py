"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Gesture Recognizer Module
------------------------
This class performs real-time gesture recognition using MediaPipe hand tracking and a 
pre-trained PyTorch model. It classifies gestures from a single hand in a video 
frame and maps them to specific movement commands (e.g., move, scale, rotate).
This module supports smoothing of gesture detection using a short history of 
past classifications and includes utility functions for calculating gesture-based 
motion deltas like translation, scaling, and rotation. It takes a state changer object
during initialization and it changes the the deltas inside the state changer
based on the gesture recognized. This same state changer will be used to 
render the object with updated position, scale, rotation in scene.py 
"""

# Imports
from utilities.model.network import GestureClassifier
import os
import torch
import mediapipe as mp
import cv2
import numpy as np
import threading
import time
import math
import random

class GestureRecognizer:
    """
    Performs real-time hand gesture recognition for 3D object manipulation using MediaPipe and PyTorch.
    
    This class detects and classifies hand gestures from video frames and maps them to specific
    movement commands (move, scale, rotate). It uses a pre-trained PyTorch model for gesture 
    classification and MediaPipe for hand landmark detection. The class maintains gesture history
    for smoothing detection and includes utilities for calculating gesture-based motion deltas.
    
    The recognized gestures include:
        - Grab and move (left/right hand)
        - Hold (left/right hand)
        - Scale (left/right hand)
        - Swipe (left/right hand)
    
    These gestures are mapped to four movement types:
        - Translation (move)
        - Scaling
        - Rotation (clockwise/counterclockwise)
        - No movement (hold)
    
    Attributes:
        state_changer: Object that maintains the position, scale, and rotation state of the 3D object
        gesture_history_size (int): Size of the gesture history queue for smoothing detection
        consecutive_threshold (int): Required consecutive detections to confirm a gesture
        gestures (dict): Mapping of class indices to gesture names
        model (GestureClassifier): Pre-trained PyTorch model for gesture classification
        hands (mediapipe.solutions.hands.Hands): MediaPipe hand tracking solution
    """
    
    def __init__(self, state_changer):
        # State change control
        self.state_changer = state_changer
        
        # Last known motion-related states
        self.last_distance_scale = 0
        self.last_distance_rotation = 0
        self.last_x = 0
        self.last_y = 0
        self.last_index_position = 0
        self.first_index_frame = True
        self.first_index_frame_move = True
        
        # Gesture history tracking
        self.gesture_history_size = 10  # size of the gesture history queue
        self.gesture_history = []  # queue of recent gesture classifications
        
        # Thresholds for gesture switching
        self.consecutive_threshold = 4  # required consecutive detections to switch to another gesture
        
        # Current state tracking
        self.current_gesture = ""

        # -------- Gestures and movements ---------
        self.gestures = {0: "grab_move_left", 1: "grab_move_right", 2: "hold_left", 3: "hold_right", 4: "scale_left_hand",
               5: "scale_right_hand", 6: "swipe_left_hand", 7: "swipe_right_hand"}

        # Map gesture classes to movements
        movements = {0: "scale", 1: "move", 2: "rotate_Y_clockwise", 3: "rotate_Y_counterclockwise"}

        # -------- Initialize the model from the saved weights ---------
        self.num_classes = 8
        self.model = GestureClassifier(num_classes=self.num_classes)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, "../utilities/model", "gesture_classifier_weights.pth")

        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()  # set to evaluation mode

        # -------- Initialize the hands from MediaPipe ---------
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def classify_gesture(self, frame):
        """
        Classifies the hand gesture in the given frame using the trained model. Draws the hand landmarks and writes the name.
        
        Args:
            frame (np.ndarray): Input BGR frame from the camera.
        
        Returns:
            dict: Mapping of predicted gesture name to MediaPipe hand landmark object.
        """

        results = {}

        # Convert frame to RGB as required by MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gestures = self.hands.process(frame_rgb)

        # Check if any hands are detected
        if gestures.multi_hand_landmarks:
            
            # Iterate through detected hands
            for hand in gestures.multi_hand_landmarks:

                # Draw the landmark
                self.mp_drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)


                # -------- Get the 63 landmarks and flatten ---------
                landmarks = []
                
                # Extract x, y, z coordinates of each landmark
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks, dtype=np.float32)
                # Convert to PyTorch tensor and add batch dimension
                input_tensor = torch.tensor(landmarks).unsqueeze(0)

                # -------- Classify gesture ---------
                with torch.no_grad():
                    outputs = self.model(input_tensor) # get model outputs
                    predicted_class = torch.argmax(outputs, dim=1).item() # get the predicted class index
                gesture_name = self.gestures.get(predicted_class, "Unknown") # get the gesture name from mapping

                results[gesture_name] = hand # Store the hand landmarks in the results dictionary

                # -------- Writes the name in frame ---------
                h, w, _ = frame.shape
                wrist = hand.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, gesture_name, (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return results

    def get_movement(self, gesture_name):
        """
        Maps a gesture name to the corresponding movement type.
        
        Args:
            gesture_name (str): The gesture being performed.
        
        Returns:
            str: Type of movement ("scale", "move", "rotate_Y_clockwise", etc.).
        """
        
        movement = ""

        if gesture_name == "scale_left_hand" or gesture_name == "scale_right_hand":
            movement = "scale"
        elif gesture_name == "grab_move_left" or gesture_name == "grab_move_right":
            movement = "move"
        elif gesture_name == "swipe_right_hand":
            movement = "rotate_Y_clockwise"
        elif gesture_name == "swipe_left_hand":
            movement = "rotate_Y_counterclockwise"
        elif gesture_name == "hold_left" or gesture_name == "hold_right":
            movement = "no_movement"

        return movement
    
    def calculate_scale_delta(self, results, frame, gesture_name):
        """
        Calculates the change in distance between thumb and index for scaling.
        
        Args:
            results (dict): Gesture recognition results.
            frame (np.ndarray): Input image frame.
            gesture_name (str): The detected gesture name.
        
        Returns:
            float: Change in distance indicating scale motion.
        """

        h, w, _ = frame.shape

        # Get thumb and index tip coordinates
        index_tip = results[gesture_name].landmark[8]
        thumb_tip = results[gesture_name].landmark[4]

        # Convert to pixel coordinates
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

        # Calculate distance between thumb and index (Euclidean distance)
        distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

        # Check if this is the first frame in the sequence
        if self.last_distance_scale < 0.5 :
            self.last_distance_scale = distance
            return 0

        # Calculate the change in distance
        delta = (distance - self.last_distance_scale)
        delta = round(delta)

        # Assign the new distance for the next frame
        self.last_distance_scale = distance
        
        return delta

    def calculate_rotation_delta(self, results, frame, gesture_name):
        """
        Calculates the change in X-axis position of index finger for rotation motion.
        
        Args:
            results (dict): Gesture recognition results.
            frame (np.ndarray): Input image frame.
            gesture_name (str): The detected gesture name.
        
        Returns:
            float: Change in horizontal position indicating a rotation.
        """

        h, w, _ = frame.shape

        # Get the index finger tip coordinates
        index_tip = results[gesture_name].landmark[8]
        
        # Convert to pixel coordinates
        index_x = int(index_tip.x * w)

        # Check if this is the first frame in the sequence
        if self.first_index_frame:
            self.first_index_frame = False
            self.last_index_position = index_x # assign the first position for next frame
            return 0
        
        # Calculate the change in X-axis position
        distance = index_x - self.last_index_position
        delta = (distance - self.last_distance_rotation)
        
        # Make sure that left hand cannot to clockwise and right hand counterclockwise
        if delta >= 0 and gesture_name == "swipe_left_hand":
            self.last_index_position = 0
            self.first_index_frame = True
            return 0
        elif delta <= 0 and gesture_name == "swipe_right_hand":
            self.last_index_position = 0
            self.first_index_frame = True
            return 0

        # Assign the new position for the next frame
        self.last_distance_rotation = distance
        
        return delta

    def calculate_translation_delta(self, results, frame, gesture_name):
        """
        Calculates the translation delta (change in x and y position) of a gesture
        from the previous frame to the current frame.

        Args:
            results (dict): Dictionary containing gesture name mapped to landmark data.
            frame (np.ndarray): The video frame where the gesture is detected.
            gesture_name (str): The name of the gesture to track.

        Returns:
            tuple: (delta_x, delta_y) representing movement in pixels.
                    Returns 0 on the first call to initialize tracking.
        """

        h, w, _ = frame.shape

        # Get the coordinates of the bottom of third finger
        fist_center = results[gesture_name].landmark[9]

        # Convert to pixel coordinates
        fist_center_x, fist_center_y = int(fist_center.x * w), int(fist_center.y * h)

        # Check if this is the first frame in the sequence
        if self.first_index_frame_move:
            self.first_index_frame_move = False
            self.last_x = fist_center_x # assign the first position for next frame
            self.last_y = fist_center_y # assign the first position for next frame
            return 0

        # Calculate the change in position
        delta_x = (self.last_x - fist_center_x)
        delta_y = (self.last_y - fist_center_y)
        self.last_x = fist_center_x # assign the new position for the next frame
        self.last_y = fist_center_y # assign the new position for the next frame

        return delta_x, delta_y
    
    def get_consecutive_count(self, queue):
        """
        Counts how many times the last element appears consecutively at the end of the list.

        Args:
            queue (list): List of gesture names (strings).

        Returns:
            int: Number of times the last element repeats consecutively at the end.
        """
    
        # Check if the queue is empty
        if not queue:
            return 0
        
        # Get the last item in the queue
        last_item = queue[-1]
        count = 0
        
        # Count backwards from the end
        for i in range(len(queue) - 1, -1, -1):
            if queue[i] == last_item:
                count += 1
            else:
                break
                
        return count
    
    def reset_variables(self, resets = ["none"]):
        """
        Reset specific internal state variables related to movement, scaling, or rotation.

        Args:
            resets (list of str): List of movement types to reset. Options: "scale", "move", "rotate"
        """
        
        # Reset the last known distances and positions based on the specified resets
        if "scale" in resets:
            self.last_distance_scale = 0
        if "move" in resets:
            self.last_x = 0
            self.last_y = 0
            self.first_index_frame_move = True
        if "rotate" in resets:
            self.last_distance_rotation = 0
            self.last_index_position = 0
            self.first_index_frame = True
            
        # Reset the state changer deltas
        self.state_changer.reset()
    
    def process(self, frame):
        """
        Main logic function that processes a frame to detect gestures and apply
        the corresponding movement (scale, translate, rotate).

        This function:
            Classifies the gesture in the frame.
            Tracks history and confirms gesture stability.
            Calculates deltas (scale, translation, rotation).
            Applies deltas if threshold is met.
            Resets variables if no stable gesture is detected.

        Args:
            frame (np.ndarray): The current video frame.

        Returns:
            None
        """

        # Returns dictionary of "gesture_name: 21 landmarks"
        results = self.classify_gesture(frame)

        # Will get first key name if it exists
        gesture_name = next(iter(results), "")
        
        # Add the gesture name to the history
        self.gesture_history.append(gesture_name)
        
        # Limit the size of the gesture history
        if len(self.gesture_history) > self.gesture_history_size:
            self.gesture_history.pop(0)
        
        # Check for consecutive identical gestures at the end of the queue
        consecutive_gesture = self.get_consecutive_count(self.gesture_history)
        
        # If the frame is a hold gesture, the user "let the object go", takes priority over other gestures
        if gesture_name == "hold_left" or gesture_name == "hold_right":
            movement = "no movement"

        # If the current gesture is empty and the new gesture is not empty
        elif self.current_gesture == "" and gesture_name != "":
            
            # If the new gesture has been held for the required number of frames
            if consecutive_gesture >= self.consecutive_threshold:
                movement = self.get_movement(gesture_name) # get the movement type

        # If the current gesture is not empty (gesture is being held)
        elif self.current_gesture != "":
            
            # If the current gesture is not the same as the new gesture and the new gesture has been held for the required number of frames
            if self.current_gesture != gesture_name and consecutive_gesture >= self.consecutive_threshold:
                movement = self.get_movement(gesture_name) # get the movement type
        
        # ------- Process for the current gesture -------
        
        if movement == "scale":
            delta = self.calculate_scale_delta(results, frame, gesture_name) # get the delta
            
            # Check if the delta is significant enough to apply scaling
            if abs(delta) >= 3.5:
                self.state_changer.update_scale_delta(delta)
            else:
                self.reset_variables(["move", "rotate"])
                
        elif movement == "move":
            x_delta, y_delta = self.calculate_translation_delta(results, frame, gesture_name) # get the deltas
            
            # Check if the deltas are significant enough to apply translation
            if abs(x_delta) >= 3 and abs(y_delta) >= 3:
                self.state_changer.update_translation_delta(x_delta, y_delta)
            else:
                self.reset_variables(["scale", "rotate"])
            
        elif movement == "rotate_Y_counterclockwise":
            delta = self.calculate_rotation_delta(results, frame, gesture_name) # get the delta
                
            # Check if the delta is significant enough to apply rotation
            if abs(delta) >= 3:
                self.state_changer.update_rotation_delta(delta)
            else:
                self.reset_variables(["scale", "move"])    
                self.last_distance_rotation = 0

        elif movement == "rotate_Y_clockwise":
            delta = self.calculate_rotation_delta(results, frame, gesture_name) # get the delta
                
            # Check if the delta is significant enough to apply rotation
            if abs(delta) >= 3:
                self.state_changer.update_rotation_delta(delta)
            else:
                self.reset_variables(["scale", "move"])    
                self.last_distance_rotation = 0
                
        # If the gesture is not recognized or is a hold gesture, reset the deltas, position states
        else:
            self.reset_variables(["scale", "move", "rotate"])  