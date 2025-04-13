import cv2
import numpy as np
import moderngl_window as mglw
from pyrr import Matrix44, Vector3  # For matrix math
from moderngl_window import WindowConfig
from pathlib import Path
from program.orbit_camera import OrbitCamera
from program.shader_program import ShaderProgram
from program.scene_object import SceneObject
from program.gesture_recognizer import GestureRecognizer
from program.state_changer import StateChanger
# from imgui_bundle import imgui
# from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer


# pip install moderngl moderngl-window pywavefront moderngl-window[imgui]



class Scene(WindowConfig):
    title = "OpenCV + ModernGL"
    window_size = (1024, 768)
    gl_version = (3, 3)
    resource_dir = (Path(__file__).parent / 'utilities' / 'render_data').resolve()
    sampels = 4 # multi-sampling
    resizable = False
    vsync = True
    use_imgui = True

    def __init__(self, **kwargs):
        """Initializes the program and its components. Args include the modernGL context, window size,
        aspect ratio, etc. It will also create and load a shader program, an controllable orbit camera,
        the objects to render, and their textures.
        """
        super().__init__(**kwargs)
        self.wnd.ctx.error
        
        self.state_changer = StateChanger()
        self.gesture_recognizer = GestureRecognizer(self.state_changer)

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
        assert Path(self.resource_dir, "models/crate.obj").exists(), "Bunny obj file not found"
        assert Path(self.resource_dir, "textures/crate.jpg").exists(), "Bunny texture not found"

        # Load the crate
        crate_mesh = self.load_scene("models/crate.obj").root_nodes[0].mesh.vao
        crate_tex = self.load_texture_2d("textures/crate.jpg")
        self.object = SceneObject(crate_mesh, crate_tex, editable=True)

        # Load the floor
        floor_mesh = self.load_scene("models/floor.obj").root_nodes[0].mesh.vao
        floor_tex = self.load_texture_2d("textures/tile_floor.jpg")
        self.floor = SceneObject(floor_mesh, floor_tex)
        self.floor.position = list([0, -0.01, 0])

        # Setup orbit camera params
        self.cam = OrbitCamera(radius=2)
        self.cam_speed = 2.5 # Camera speed when moving

        # OpenCV webcam
        self.cap = cv2.VideoCapture(0)

    def on_render(self, time:float , frame_time: float) -> None:
        """The rendering pipeline for this program.

        Args:
            time (float): The time of the start of the rendering.
            frame_time (float): The time since the last frame
        """
        # Read a frame from webcam
        ret, frame = self.cap.read()
        if ret:
            
            self.gesture_recognizer.process(frame)
            cv2.imshow("Webcam", frame)

            # if cv2.waitKey(1) & 0xFF == 27: # ESC key
            #     self.wnd.close()

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

    def handle_object(self, object: SceneObject, dt:float) -> None:
        """Key listener to adjust scene object parameters. Currently only supports adjusting one object
        at any time.

        Args:
            object (SceneObject): The object to manipulate
            dt (float): The delta time from the last frame.
        """
        keys = self.wnd.keys
        speed = 2.0 * dt
        rot_speed = 45.0 * dt
        scale_speed = 0.5 * dt

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

    def handle_gesture(self, object, frame_time):
        # Would look something like the key event handler above
        pass

    def destroy(self):
        """Cleans up memory upon shutdown
        """
        # Clean up OpenCV when window closes
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mglw.run_window_config(Scene)
