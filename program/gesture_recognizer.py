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
import threading
import time
import math


class GestureRecognizer:
    def __init__(self, state_changer):
        self.state_changer = state_changer
        self.last_update = 0        # Keeps track of the time of the most recent update
        self.last_distance = 0      # Keeps track of the last distance between the index and thumb finger
        self.lock = threading.Lock() # For state updates

        # -------- Gestures and movements ---------
        self.gestures = {0: "grab_move_left", 1: "grab_move_right", 2: "hold_left", 3: "hold_right", 4: "scale_left_hand",
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
            Takes a frame, finds a hand, classifies the gesture, draws it on screen, returns gesture name and landmarks.
        """

        results = {}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gestures = self.hands.process(frame_rgb)

        if gestures.multi_hand_landmarks:
            for hand in gestures.multi_hand_landmarks:

                # Draw the landmark; can erase later
                self.mp_drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)


                # -------- Get the 63 landmarks and flatten ---------
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks, dtype=np.float32)
                # Convert to PyTorch tensor and add batch dimension
                input_tensor = torch.tensor(landmarks).unsqueeze(0)

                # -------- Classify gesture ---------
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                gesture_name = self.gestures.get(predicted_class, "Unknown")

                results[gesture_name] = hand

                # -------- Writes the name in frame ---------
                h, w, _ = frame.shape
                wrist = hand.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, gesture_name, (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return results

    def get_movement(self, gesture_name):
        """
            This will return the name of the movement we're performing.
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

    def check_is_move_active(self):
        """
            We need to check if we "held" the grab hand for 1 second before we enable moving.
            This is up for discussion on how to implement.
        """

        pass

    def calculate_scale_delta(self, results, frame, gesture_name):
        """
            Grab the landmarks and calculate the change. Will need to take into account previous position
            of landmarks.
        """

        # bla bla bla here

        h, w, _ = frame.shape

        index_tip = results[gesture_name].landmark[8]
        thumb_tip = results[gesture_name].landmark[4]

        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

        distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

        if self.last_distance < 0.5 :
            print("First distance")
            print(distance)
            self.last_distance = distance
            return 0

        delta = (distance - self.last_distance)
        # round to nearest integer
        delta = round(delta)

        self.last_distance = distance
        return delta

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
        # We might need to set up whatever we will memorize in terms of previous frame or previous delta...

        results = self.classify_gesture(frame) # Returns dictionary of "gesture_name: 21 landmarks"

        # Will get first key name if it exists
        gesture_name = next(iter(results), "")

        now = time.time()

        # if now - self.last_update > 0.5:        # Adds a time buffer to dampen changes in state as the system processes it

        #     movement = self.get_movement(gesture_name)
        #     print(movement) # Debugging

        #     with self.lock:         # Lock only while the state is being updated
        #         if movement == "scale":
        #             self.calculate_scale_delta()
        #         elif movement == "move":
        #             self.calculate_translation_delta()
        #         elif movement == "rotate":
        #             self.calculate_rotation_delta()

        #     self.last_update = now

        # else: #### Debugging
        #     print(f"Skipped {gesture_name} due to inside buffer time")



        movement = self.get_movement(gesture_name)
        # print(movement) # Debugging

        # with self.lock:         # Lock only while the state is being updated
        #     if movement == "scale":

        #         print("Scale detected", results[gesture_name])
        #         self.calculate_scale_delta()
        #     elif movement == "move":
        #         self.calculate_translation_delta()
        #     elif movement == "rotate":
        #         self.calculate_rotation_delta()

        # self.last_update = now



        if movement == "scale":
            delta = self.calculate_scale_delta(results, frame, gesture_name)
            if abs(delta) >= 3:
                self.state_changer.update_scale_delta(delta)
                print(self.state_changer.scale_delta)
            else:
                self.state_changer.reset()
        elif movement == "move":
            self.calculate_translation_delta()
        elif movement == "rotate":
            self.calculate_rotation_delta()
        else:
            self.state_changer.reset()
