"""
    Does the gesture recognition and checks if there is one hand, two hands, 
    which hands are up and deduces if we’re doing a motion.
    
    Will handle if grab is on the screen for one second, and only if so, 
    will change the motion to the appropriate one, and change the state of the state_changer so
    that we’re doing this motion.
"""

from utilities.model.network import GestureClassifier
import os
import torch
import mediapipe as mp
import cv2
import numpy as np


class GestureRecognizer:
    def __init__(self, state_changer):
        self.state_changer = state_changer
        
        gestures = {0: "grab_move_left", 1: "grab_move_right", 2: "hold_left", 3: "hold_right", 4: "scale_left_hand",
               5: "scale_right_hand", 6: "swipe_left_hand", 7: "swipe_right_hand"}
        
        movements = {0: "scale", 1: "move", 2: "rotate_Y_clockwise", 3: "rotate_Y_counterclockwise"}
        
        # -------- Initialize the model from the saved weights ---------
        self.num_classes = 8 
        self.model = GestureClassifier(num_classes=self.num_classes)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, "../utilities/model", "gesture_classifier_weights.pth")

        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()  # Set to evaluation mode
        
        # -------- Initialize the hands from MediaPipe ---------
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def classify_gesture(self, frame):
        """
            This will take the frame and return 1 or 2 gestures in the frame. 
        """
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gestures = self.hands.process(frame_rgb)
        
        if gestures.multi_hand_landmarks:
            for hand in gestures.multi_hand_landmarks:
                
                # Draw the landmark; can erase later
                self.mp_drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        
        
        # NEED TO DO CLASSIFICATION
        

    def get_movement(self):
        """
            This will return the name of the movement we're performing.
            
            hold_left + swipe_right_hand will be rotate_Y_clockwise
            hold_right + swipe_left_hand will be rotate_Y_counterclockwise
            grab_move_left OR grab_move_right will be move
            scale_left_hand OR scale_right_hand will be scale
        """
    
        pass
        
    def check_is_move_active(self):
        """
            We need to check if we "held" the grab hand for 1 second before we enable moving. 
            This is up for discussion on how to implement.
        """
        
        pass
        
    def calculate_scale_delta(self):
        """
            Grab the landmarks and calculate the change. Will need to take into account previous position
            of landmarks.
        """
        
        # bla bla bla here
        
        scale_delta = 0 # example
        
        self.state_changer.update_scale_delta(scale_delta)
    
    def calculate_rotation_delta(self):
        """
            Grab the landmarks and calculate the change. Will need to take into account previous position
            of landmarks.
        """
        
        rotation_delta = 0 # example
        
        self.state_changer.update_rotation_delta(rotation_delta)
        
        pass
    
    def calculate_translation_delta(self):
        """
            Grab the landmarks and calculate the change. Will need to take into account previous position
            of landmarks.
        """
        
        x_delta = 0 # example
        y_delta = 0 # example
        
        self.state_changer.update_translation_delta(x_delta, y_delta)
        
        pass

    def process(self, frame):
        """
            Function that will run the entire logic and call the above function.
            This one will also update the state_changer.
            
            Note: have to think about handling instances of incorrectly classified 1 frame in a series of other
            correctly classified ones. 
            Example: when rotating, sometimes a hand reads as scale - need to disable switching to scale.
        """
        
        self.frame = frame
        # We might need to set up whatever we will memorize in terms of previous frame or previous delta...
        
        self.classify_gesture(self.frame)
    
    