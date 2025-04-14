"""
    StateChanger will be passed into gesture and is updated by gestureRecognizer.
    
    Then on initialization in Scene, we can also pass this same StateChanger as a parameter to SceneObject, 
    and it can grab the updates for rendering directly from here.
    
    SceneObject needs to be modified if we go with this architecture.
    Another option is to use this to craft the handle_gesture.
"""


class StateChanger:
    def __init__(self):
        self.current_state = None
        
        self.scale_delta = 0 # since I'm assuming this might be a multiplier, we might need to talk through the logic  
        self.rotation_delta = 0 # not sure if this is a multiplier or degree amount, will need to check and talk logic
        self.translation_delta = (0, 0) # tuple, x, y translation respectively (only supporting 2D right now)
        
    def update_scale_delta(self, new_delta):
        self.scale_delta = new_delta
        self.rotation_delta = 40
        self.translation_delta = (0, 0)
        
    def update_rotation_delta(self, new_rotation):
        self.scale_delta = 1
        self.rotation_delta = new_rotation
        self.translation_delta = (0, 0)
        
    def update_translation_delta(self, x, y):
        self.scale_delta = 1
        self.rotation_delta = 0
        self.translation_delta = (x, y)
        
    
        