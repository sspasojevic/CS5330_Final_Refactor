import cv2
import numpy as np
import moderngl_window as mglw
import threading
import queue
from pyrr import Matrix44, Vector3  # For matrix math
from moderngl_window import WindowConfig
from pathlib import Path
from program.orbit_camera import OrbitCamera
from program.shader_program import ShaderProgram
from program.scene_object import SceneObject
from program.gesture_recognizer import GestureRecognizer
from program.state_changer import StateChanger
import time



# pip install moderngl moderngl-window pywavefront moderngl-window[imgui]



class Scene(WindowConfig):
    title = "OpenCV + ModernGL"
    window_size = (1024, 768)
    gl_version = (3, 3)
    resource_dir = (Path(__file__).parent / 'utilities' / 'render_data').resolve()
    sampels = 4 # multi-sampling
    resizable = False
    vsync = True

    def __init__(self, **kwargs):
        """Initializes the program and its components. Args include the modernGL context, window size,
        aspect ratio, etc. It will also create and load a shader program, an controllable orbit camera,
        the objects to render, and their textures.
        """
        super().__init__(**kwargs)
        self.wnd.ctx.error

        self.input_queue = queue.Queue(1) # Create a queue to hold the most recent frame
        self.output_queue = queue.Queue(1)
        self.frame = None

        # State changer and gesture recognizer will modify the object parameters
        self.state_changer = StateChanger()
        self.gesture_recognizer = GestureRecognizer(self.state_changer)

        # Set up for multi-threading
        self.lock = threading.Lock()

        self.processing_active = True
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True) # Daemon ensures the thread ends with the main program.
        self.model_thread = threading.Thread(target=self.run_gesture_model, daemon=True)

        self.processing_thread.start()
        self.model_thread.start()


        self.shader_program = ShaderProgram(self.ctx)
        assert Path(self.resource_dir, "shaders/vertex.glsl").exists(), "Vertex shader program not found"
        assert Path(self.resource_dir, "shaders/fragment.glsl").exists(), "Fragment shader program not found"

        self.prog = self.shader_program.load_shader(
            name = "crate",
            vertex_path=self.resource_dir / 'shaders' / 'vertex.glsl',
            fragment_path=self.resource_dir / 'shaders' / 'fragment.glsl'
        )

        print(f"Loaded shader program successfully")

        # Verify the model and texture exist
        assert Path(self.resource_dir, "models/crate.obj").exists(), "obj file not found"
        assert Path(self.resource_dir, "textures/crate.jpg").exists(), "texture not found"

        # Load the crate
        obj_mesh = self.load_scene("models/crate.obj").root_nodes[0].mesh.vao
        obj_tex = self.load_texture_2d("textures/crate.jpg")
        self.object = SceneObject(obj_mesh, obj_tex, self.state_changer)

        # Load the floor
        floor_mesh = self.load_scene("models/floor.obj").root_nodes[0].mesh.vao
        floor_tex = self.load_texture_2d("textures/tile_floor.jpg")
        self.floor = SceneObject(floor_mesh, floor_tex)
        self.floor.position = list([0, -0.01, 0]) # Place the floor just slightly below the object

        # Setup orbit camera params
        self.cam = OrbitCamera(radius=1)
        self.cam_speed = 2.5 # Camera speed when moving

    def process_frames(self):
        # OpenCV webcam
        self.cap = cv2.VideoCapture(0)

        while self.processing_active:
            ret, frame = self.cap.read()
            if ret:
                # frame = cv2.flip(frame, 1)
                # cv2.imshow("Webcam", frame)  # display the frame in another window
                # self.frame = frame

                if not self.input_queue.full():           # Will only add a frame to the queue if it's empty and ready to be examined by the model
                    self.input_queue.put(frame.copy())    # Copy will create a standalone frame to save here, rather than passing a reference to the original frame.

                # time.sleep(5) ### Add artificial lag here if desired for testing

    def run_gesture_model(self):
        """Tertiary thread that runs the gesture recognizer model. Will try to intake an image from the queue and process it
        Otherwise it will skip and continue to the next loop
        """
        last_update = 0

        while self.processing_active:
            try:
                frame = self.input_queue.get()    # optionally, add timeout=1 to give a 1 second delay to wait for a new frame
                self.gesture_recognizer.process(frame)

                if not self.output_queue.full():
                    self.output_queue.put(frame.copy())

            # Empty queue will raise exception. Could also change this flow to check for empty queue before taking to avoid
            # error handling as control flow
            except:
                # if no gestures, pass the frame through unannotated
                if not self.output_queue.full():
                    self.output_queue.put(frame.copy())
                continue                # Otherwise, continue for another loop

    def on_render(self, time:float , frame_time: float) -> None:
        """The rendering pipeline for this program.

        Args:
            time (float): The time of the start of the rendering.
            frame_time (float): The time since the last frame
        """

        # Camera event listener.
        # WASD will move camera orbit camera Up/Down/Left/Right
        # Q/E will zoom in/out
        # Up/Down/Left/Right arrows will pan the camera to a new position as well as orbit new point.
        # Panning is relative to the camera axis projected onto world X-Z for natural
        self.handle_movement(frame_time)

        # Object event listener.
        # O/K/L/; will translate the object along the X- or Z- axis (absolute, world scales)
        # RT/FG/VB will rotate the model about the X, Y, Z axes, respectively
        # YU/HJ/NM will scale the model in the X, Y, Z axes, respectively
        self.handle_object(self.object, frame_time) # Keyboard inputs

        ##### Model command inputs go here
        # Send commands from gesture to the object... The declaration can change
        self.handle_gesture(self.object, frame_time)


        #####

        # This sets the background color and enables a depth test to improve rendering
        self.ctx.clear(0.1, 0.1, 0.1)
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

        # Update to most recent frame
        if self.output_queue.full():
            self.frame = self.output_queue.get()
        # Display frame if available
        if self.frame is not None:
            cv2.imshow("Webcam", cv2.flip(self.frame, 1))
            cv2.waitKey(1)

    def handle_object(self, object: SceneObject, dt:float) -> None:
        """Key listener to adjust scene object parameters. Currently only supports adjusting one object
        at any time.

        Args:
            object (SceneObject): The object to manipulate
            dt (float): The delta time from the last frame.
        """
        keys = self.wnd.keys
        speed = object.translation_speed * dt
        rot_speed = object.rotation_speed * dt
        scale_speed = object.scale_speed * dt

        # Translations
        if self.wnd.is_key_pressed(keys.O): object.position[2] -= speed # Forward
        if self.wnd.is_key_pressed(keys.L): object.position[2] += speed # Back

        if self.wnd.is_key_pressed(keys.K): object.position[0] -= speed # Left
        if self.wnd.is_key_pressed(keys.SEMICOLON): object.position[0] += speed # Right

        if self.wnd.is_key_pressed(keys.I): object.position[1] += speed # Up
        if self.wnd.is_key_pressed(keys.P): object.position[1] -= speed # Down

        # Rotations
        if self.wnd.is_key_pressed(keys.R): object.rotation[0] += rot_speed # ',' Rotate about X axis
        if self.wnd.is_key_pressed(keys.T): object.rotation[0] -= rot_speed # '.' Rotate about X axis

        if self.wnd.is_key_pressed(keys.F): object.rotation[1] += rot_speed # ',' Rotate about Y axis
        if self.wnd.is_key_pressed(keys.G): object.rotation[1] -= rot_speed # '.' Rotate about Y axis

        if self.wnd.is_key_pressed(keys.V): object.rotation[2] += rot_speed # ',' Rotate about Y axis
        if self.wnd.is_key_pressed(keys.B): object.rotation[2] -= rot_speed # '.' Rotate about Y axis

        # Scales
        if self.wnd.is_key_pressed(keys.Y): object.scale[0] = max(0.1, object.scale[0] - scale_speed)
        if self.wnd.is_key_pressed(keys.U): object.scale[0] = min(10.0, object.scale[0] + scale_speed)

        if self.wnd.is_key_pressed(keys.H): object.scale[1] = max(0.1, object.scale[1] - scale_speed)
        if self.wnd.is_key_pressed(keys.J): object.scale[1] = min(10.0, object.scale[1] + scale_speed)

        if self.wnd.is_key_pressed(keys.N): object.scale[2] = max(0.1, object.scale[2] - scale_speed)
        if self.wnd.is_key_pressed(keys.M): object.scale[2] = min(10.0, object.scale[2] + scale_speed)

        if self.wnd.is_key_pressed(keys.BACKSPACE):
            object.position = [0, 0, 0]
            object.rotation = [0, 0, 0]
            object.scale = [1, 1, 1]



    def handle_movement(self, dt: float) -> None:
        """Key listener to control the orbital camera. It may be rotated about its central point,
        panned across the scene, or brought in closer/further from the center (zoom)

        Args:
            dt (float): time elapsed since the previous frame
        """
        keys = self.wnd.keys
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
        # Update by reading the state from state_changer
        with self.lock:                                     # Lock prevents reading while being updated by the model
            scale_delta = self.state_changer.scale_delta
            rotation_delta = self.state_changer.rotation_delta
            translation_delta = self.state_changer.translation_delta

        if object.state_changer:
            # Scale in all 3 axes at once
            for i in range(3):
                target_scale = object.scale[i] + scale_delta * dt
                object.scale[i] = min(10, max(0.1, object.scale[i] * (1 - 0.1) + target_scale * 0.1))

            # Rotation about Y axis only
            object.rotation[1] += rotation_delta * dt * 2

            # Translate on X, Y axes only
            object.position[2] += translation_delta[0] * dt / 10
            object.position[1] += translation_delta[1] * dt / 10




    def destroy(self):
        """Cleans up memory upon shutdown
        """
        # Clean up OpenCV when window closes
        self.processing_active = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mglw.run_window_config(Scene)
