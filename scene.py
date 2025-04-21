"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Scene Module
-----------
This module provides the main rendering environment for a 3D application that combines 
OpenCV-based gesture recognition with ModernGL 3D rendering. It establishes a multi-threaded
architecture where webcam input, gesture recognition, and 3D rendering operate concurrently.

The Scene class extends ModernGL's WindowConfig to create a window with 3D rendering capabilities,
handling user input from both keyboard controls and gesture recognition to manipulate objects
in the 3D environment.
"""

# Imports
import cv2
import numpy as np
import moderngl_window as mglw
import threading
import queue
from pyrr import Matrix44, Vector3  
from moderngl_window import WindowConfig
from pathlib import Path
from program.orbit_camera import OrbitCamera
from program.shader_program import ShaderProgram
from program.scene_object import SceneObject
from program.gesture_recognizer import GestureRecognizer
from program.state_changer import StateChanger
import time

class Scene(WindowConfig):
    """
    Main rendering environment combining OpenCV gesture recognition with ModernGL 3D rendering.
    
    This class establishes a multi-threaded application where webcam input, gesture recognition,
    and 3D rendering operate concurrently. It handles the rendering pipeline, camera controls,
    object manipulation via both keyboard and gesture recognition, and resource management.
    
    The class uses three threads:
    1. Main thread: Handles rendering and keyboard input
    2. Processing thread: Captures frames from webcam
    3. Model thread: Processes frames for gesture recognition
    
    Attributes:
        title (str): Window title
        window_size (tuple): Initial window dimensions
        gl_version (tuple): OpenGL version requirement
        resource_dir (Path): Directory containing resources (models, textures, shaders)
        samples (int): Multi-sampling level for anti-aliasing
        resizable (bool): Whether the window can be resized
        vsync (bool): Whether vertical sync is enabled
        input_queue (Queue): Thread-safe queue for frames from webcam to model
        output_queue (Queue): Thread-safe queue for processed frames from model to display
        state_changer (StateChanger): Shared state for gesture-based transformations
        gesture_recognizer (GestureRecognizer): Processes hand gestures from webcam frames
        shader_program (ShaderProgram): Manager for OpenGL shader programs
        object (SceneObject): Main object to manipulate in the scene
        floor (SceneObject): Floor/ground plane object in the scene
        cam (OrbitCamera): Controllable camera for viewing the scene
    """
    
    title = "OpenCV + ModernGL"
    window_size = (1024, 768)
    gl_version = (3, 3)
    resource_dir = (Path(__file__).parent / 'utilities' / 'render_data').resolve()
    samples = 4 # multi-sampling
    resizable = False
    vsync = True

    def __init__(self, **kwargs):
        """
        Initializes the program and its components.
        
        Sets up the rendering context, threading architecture, gesture recognition,
        shader programs, 3D objects, textures, and camera controls.
        
        Args:
            **kwargs: Arguments passed to WindowConfig parent class, including
                     modernGL context, window size, aspect ratio, etc.
        """
        
        # Initialize the parent WindowConfig class
        super().__init__(**kwargs)
        self.wnd.ctx.error

        # Create thread-safe queues for frame passing between threads
        self.input_queue = queue.Queue(1)  # Queue for frames from webcam to model (size 1)
        self.output_queue = queue.Queue(1)  # Queue for processed frames from model to display
        self.frame = None  # Current frame to display

        # Create state changer and gesture recognizer for object manipulation
        self.state_changer = StateChanger()
        self.gesture_recognizer = GestureRecognizer(self.state_changer)

        # Set up multi-threading architecture
        self.lock = threading.Lock()  # Thread synchronization
        self.processing_active = True  # Flag to control thread execution
        
        # Create threads for webcam capture and gesture recognition
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)  # Daemon ensures the thread ends with the main program
        self.model_thread = threading.Thread(target=self.run_gesture_model, daemon=True)

        # Launch the threads
        self.processing_thread.start()
        self.model_thread.start()

        # Set up the scene vertex shaders
        self.shader_program = ShaderProgram(self.ctx)
        assert Path(self.resource_dir, "shaders/vertex.glsl").exists(), "Vertex shader program not found"
        assert Path(self.resource_dir, "shaders/fragment.glsl").exists(), "Fragment shader program not found"

        # Load shader program
        self.prog = self.shader_program.load_shader(
            name = "crate",
            vertex_path=self.resource_dir / 'shaders' / 'vertex.glsl',
            fragment_path=self.resource_dir / 'shaders' / 'fragment.glsl'
        )
        print(f"Loaded shader program successfully")

        # Load the scene; Verify 3D model and texture resources exist
        assert Path(self.resource_dir, "models/crate.obj").exists(), "obj file not found"
        assert Path(self.resource_dir, "textures/crate.jpg").exists(), "texture not found"

        # Load main object (crate)
        obj_mesh = self.load_scene("models/crate.obj").root_nodes[0].mesh.vao
        obj_tex = self.load_texture_2d("textures/crate.jpg")
        self.object = SceneObject(obj_mesh, obj_tex)

        # Load floor object
        floor_mesh = self.load_scene("models/floor.obj").root_nodes[0].mesh.vao
        floor_tex = self.load_texture_2d("textures/tile_floor.jpg")
        self.floor = SceneObject(floor_mesh, floor_tex)
        self.floor.position = list([0, -0.01, 0]) # Place the floor just slightly below the object

        # Setup orbit camera for scene navigation
        self.cam = OrbitCamera(radius=1)
        self.cam_speed = 2.5 # Camera speed when moving

    def process_frames(self):
        """
        Secondary thread that continuously captures frames from the webcam.
        
        This method runs in its own thread, capturing frames from the webcam at native
        frame rate. It adds frames to the input queue if there's space available.
        If the queue is full, it discards the current frame and captures a new one.
        """
        
        # OpenCV webcam
        self.cap = cv2.VideoCapture(0)

        # Loop while the application is running
        while self.processing_active:
            
            # Capture frame from webcam
            ret, frame = self.cap.read()
            if ret:
                if not self.input_queue.full():           # Will only add a frame to the queue if it's empty and ready to be examined by the model
                    self.input_queue.put(frame.copy())    # Copy will create a standalone frame to save here, rather than passing a reference to the original frame.


    def run_gesture_model(self):
        """
        Tertiary thread that processes frames for gesture recognition.
        
        This method runs in its own thread, taking frames from the input queue,
        processing them with the gesture recognizer to detect hand gestures,
        and placing the processed frames (with annotations) in the output queue
        for display. If gesture detection fails, it passes the unaltered frame.
        """

        # Loop while the application is running
        while self.processing_active:
            try:
                
                # Get a frame from the input queue
                frame = self.input_queue.get()
                
                # Process the frame with gesture recognition
                self.gesture_recognizer.process(frame)

                # Add processed frame to output queue if space is available (for render of real time webcam)
                if not self.output_queue.full():
                    self.output_queue.put(frame.copy())

            # Empty queue will raise exception
            except:
                # If no frame is available or processing fails, try to pass the frame unmodified
                if not self.output_queue.full():
                    self.output_queue.put(frame.copy())
                continue                # Otherwise, continue for next iteration

    def on_render(self, time:float , frame_time: float) -> None:
        """
        The main rendering pipeline, called each frame.
        
        Handles camera movement input, applies gesture-based transformations to objects,
        renders the 3D scene, and displays the webcam feed with gesture annotations.
        
        Args:
            time (float): The time since application start
            frame_time (float): The time elapsed since the last frame
        """

        # Camera event listener.
        # WASD will move camera orbit camera Up/Down/Left/Right
        # Q/E will zoom in/out
        # Up/Down/Left/Right arrows will pan the camera to a new position as well as orbit new point.
        # Panning is relative to the camera axis projected onto world X-Z for natural
        self.handle_movement(frame_time)

        # Apply gesture-based transformations to the object
        self.handle_gesture(self.object, frame_time)

        # Clear the screen and enable depth testing for proper 3D rendering
        self.ctx.clear(0.1, 0.1, 0.1)  # Dark gray background
        self.ctx.enable(self.ctx.DEPTH_TEST)

        # Builds the view and projection matrices. Model will be created at each mesh's render
        view = self.cam.get_view_matrix()
        proj = Matrix44.perspective_projection(
            fovy=45.0,
            aspect=self.wnd.aspect_ratio,
            near=0.1,
            far=100.0,
            dtype='f4'
        )

        # Establish the uniforms for the view and projection matrices
        self.prog['view'].write(view.astype('f4').tobytes())
        self.prog['proj'].write(proj.astype('f4').tobytes())

        # Renders each mesh individually. Internally, it will determine its texture and create a model matrix
        self.object.render(self.prog, texture_unit=0)
        self.floor.render(self.prog, texture_unit=1, uv_scale=1)

        # Updates the webcam frame to most recent one available
        if self.output_queue.full():
            self.frame = self.output_queue.get()

        # Display webcam frame, if available
        if self.frame is not None:
            cv2.imshow("Webcam", cv2.flip(self.frame, 1))
            cv2.waitKey(1)

    
    def handle_movement(self, dt: float) -> None:
        """
        Processes keyboard input to control the orbital camera.
        
        Allows the camera to be rotated around its central point, panned across
        the scene, and zoomed in or out using keyboard controls.
        
        Args:
            dt (float): Time elapsed since the previous frame, for time-based movement
        """
        
        # Get keyboard state
        keys = self.wnd.keys
        
        # Calculate movement speeds based on frame time
        speed = self.cam_speed * dt
        rot_speed = 90.0 * dt
        zoom_speed = 2.0 * dt

        # Pan control
        if self.wnd.is_key_pressed(keys.UP):
            self.cam.pan(0, 0, 1, speed)
        if self.wnd.is_key_pressed(keys.DOWN):
            self.cam.pan(0, 0, -1, speed)
        if self.wnd.is_key_pressed(keys.LEFT):
            self.cam.pan(-1, 0, 0, speed)
        if self.wnd.is_key_pressed(keys.RIGHT):
            self.cam.pan(1, 0, 0, speed)

        # Orbit cam rotation control
        if self.wnd.is_key_pressed(keys.W):
            self.cam.rotate(0, -rot_speed)
        if self.wnd.is_key_pressed(keys.S):
            self.cam.rotate(0, rot_speed)
        if self.wnd.is_key_pressed(keys.A):
            self.cam.rotate(rot_speed, 0)
        if self.wnd.is_key_pressed(keys.D):
            self.cam.rotate(-rot_speed, 0)

        # Zoom Controls
        if self.wnd.is_key_pressed(keys.Q):
            self.cam.zoom(-zoom_speed)
        if self.wnd.is_key_pressed(keys.E):
            self.cam.zoom(zoom_speed)

    def handle_gesture(self, object: SceneObject, dt: float):
        """
        Applies gesture-based transformations to a scene object.
        
        Reads transformation deltas from the state_changer (updated by gesture recognition)
        and applies them to the specified scene object's position, scale, and rotation.
        
        Args:
            object (SceneObject): The object to transform
            dt (float): Time elapsed since the previous frame, for time-based transformations
        """
        
        # Read transformation deltas from state_changer with thread safety
        with self.lock:                                     # Lock prevents reading while being updated by the model
            scale_delta = self.state_changer.scale_delta
            rotation_delta = self.state_changer.rotation_delta
            translation_delta = self.state_changer.translation_delta

        if self.state_changer:
            # Apply scaling to all axes with smoothing
            for i in range(3):
                target_scale = object.scale[i] + scale_delta * dt
                
                # Smooth scaling with weighted average (10% new value, 90% old value); clipped to limit scaling
                object.scale[i] = min(10, max(0.1, object.scale[i] * (1 - 0.1) + target_scale * 0.1))

            # Apply rotation around Y axis only
            object.rotation[1] += rotation_delta * dt * 2

            # Apply translation on X and Y axes
            object.position[2] += translation_delta[0] * dt / 10
            object.position[1] += translation_delta[1] * dt / 10


    def destroy(self):
        """
        Cleans up resources when the application is closing.
        
        Signals background threads to terminate, releases the webcam,
        and closes any open OpenCV windows.
        """
        
        # Signal background threads to terminate
        self.processing_active = False
        
        # Release webcam if open
        if self.cap.isOpened():
            self.cap.release()
            
        # Close all OpenCV windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mglw.run_window_config(Scene)
