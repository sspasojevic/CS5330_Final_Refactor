"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

State Changer Module
------------------
This module provides a communication mechanism between gesture recognition and 3D object manipulation.
It serves as a shared state container that is updated by gesture recognition components and
consumed by rendering components to apply transformations to 3D objects in real-time.

The StateChanger class acts as a bridge in the gesture-based object manipulation pipeline,
maintaining deltas for scale, rotation, and translation that are derived from hand gestures
and applied during the rendering process.
"""


class StateChanger:
    """
    A shared state container for coordinating gesture-based object transformations.
    
    This class serves as an intermediary between gesture recognition and object rendering
    components. It stores transformation deltas (scale, rotation, translation) that are
    updated by gesture recognizers and accessed by scene objects during rendering.
    
    The class maintains mutually exclusive transformation states - only one type of
    transformation (scale, rotation, or translation) can be active at a time.
    
    Attributes:
        current_state (any): Optional field for storing current application state
        scale_delta (float): Change in scale to be applied to objects
        rotation_delta (float): Change in rotation (degrees) to be applied to objects
        translation_delta (tuple): Change in position (x, y) to be applied to objects
    """
    
    def __init__(self):
        """
        Initializes the StateChanger with default values.
        
        Sets all transformation deltas to their neutral values (no change)
        and initializes the current_state attribute to None.
        """

        # Transformation deltas - only one is active at a time
        self.scale_delta = 0  # Change in scale factor
        self.rotation_delta = 0  # Change in rotation (degrees)
        self.translation_delta = (0, 0)  # Change in position (x, y)

    def update_scale_delta(self, new_delta):
        """
        Updates the scale delta and resets other transformation deltas.
        
        Sets the scale_delta to the specified value and resets rotation_delta
        and translation_delta to ensure mutual exclusion of transformations.
        
        Args:
            new_delta (float): The new scale delta value
        """
        
        # Set scale delta and reset other deltas for mutual exclusion
        self.scale_delta = new_delta
        self.rotation_delta = 0
        self.translation_delta = (0, 0)

    def update_rotation_delta(self, new_rotation):
        """
        Updates the rotation delta and resets other transformation deltas.
        
        Sets the rotation_delta to the specified value and resets scale_delta
        and translation_delta to ensure mutual exclusion of transformations.
        
        Args:
            new_rotation (float): The new rotation delta value in degrees
        """
        
        # Set rotation delta and reset other deltas for mutual exclusion
        self.scale_delta = 0
        self.rotation_delta = new_rotation
        self.translation_delta = (0, 0)

    def update_translation_delta(self, x, y):
        """
        Updates the translation delta and resets other transformation deltas.
        
        Sets the translation_delta to the specified x and y values and resets
        scale_delta and rotation_delta to ensure mutual exclusion of transformations.
        
        Args:
            x (float): The new x-axis translation delta
            y (float): The new y-axis translation delta
        """
        
        # Set translation delta and reset other deltas for mutual exclusion
        self.scale_delta = 0
        self.rotation_delta = 0
        self.translation_delta = (x, y)

    def reset(self):
        """
        Resets all transformation deltas to their neutral values.
        
        This method is typically called when no gesture is detected or
        when transitioning between different gesture types.
        """
        
        # Reset all deltas to neutral values (no change)
        self.scale_delta = 0
        self.rotation_delta = 0
        self.translation_delta = (0, 0)
