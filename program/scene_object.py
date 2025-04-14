from pyrr import Matrix44, Vector3
from moderngl import Context, Program, Texture
from math import radians
from state_changer import StateChanger

class SceneObject:
    def __init__(self, vao, texture: Texture, state_changer: StateChanger=None):
        """Initializes an object to be rendered via it's mesh VAO. The position, scale, and rotation are
        automatically set to pos=[0,0,0], scale=[1,1,1], and rotation=[0,0,0]

        Args:
            vao (Mesh VAO): From the mglw scene loader, the mesh.vao object
            texture (Texture): The texture associated with the object
            editable (bool, optional): If the object will be modifiable by the user. Defaults to False.
        """

        self.vao = vao
        self.texture = texture
        self.state_changer = state_changer

        self.position = [0.0, 0.0, 0.0] 
        self.scale = [1.0, 1.0, 1.0]    
        self.rotation = [0.0, 0.0, 0.0]
        
        # For keyboard controls
        self.scale_speed = 0.5
        self.rotation_speed = 45 # deg
        self.translation_speed = 2.0

    def get_model_matrix(self) -> Matrix44:
        """Calculates and returns the 4x4 model matrix by computing M = T * R * S

        Returns:
            Matrix44: The model matrix
        """
        Txyz = Matrix44.from_translation(self.position)
        Rx = Matrix44.from_x_rotation(radians(self.rotation[0]))
        Ry = Matrix44.from_y_rotation(radians(self.rotation[1]))
        Rz = Matrix44.from_z_rotation(radians(self.rotation[2]))
        Sxyz = Matrix44.from_scale(self.scale)

        # Matrix multiplication L <- R
        return Sxyz @ Rx @ Ry @ Rz @ Txyz

    def render(self, prog:Program, texture_unit=0, uv_scale=1.0):
        """Renders the object onto the scene.

        Args:
            prog (Program): The shader program to use
            texture_unit (int, optional): The texture unit location. Defaults to 0.
            uv_scale (float, optional): The uv scale to use. Defaults to 1.0.
        """
        self.texture.use(location=texture_unit)
        prog["Texture"] = texture_unit
        prog["uv_scale"].value = uv_scale
        prog["model"].write(self.get_model_matrix().astype('f4').tobytes())
        self.vao.render(prog)
