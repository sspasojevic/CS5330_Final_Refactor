"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Scene Object Module
------------------
This module provides a 3D object representation for rendering in a moderngl-based graphics system.
It handles the coupling of mesh geometry (VAO) with textures and maintains transform properties
(position, scale, rotation) for the object. The module supports both programmatic and
user-interactive manipulation of 3D objects within a scene.

The SceneObject class manages all aspects of 3D object transformation and rendering,
providing a convenient interface for scene composition and object manipulation.
"""

# Imports
from pyrr import Matrix44, Vector3
from moderngl import Context, Program, Texture
from math import radians
from program.state_changer import StateChanger

class SceneObject:
    """
    A renderable 3D object in a scene with transformation capabilities.
    
    This class represents a 3D object with its own mesh geometry (VAO), texture,
    and transform properties (position, scale, rotation). It provides methods for
    calculating its model matrix and rendering itself with a specified shader program.
    
    Attributes:
        vao: The vertex array object containing the mesh geometry
        texture (Texture): The texture applied to the object
        position (list): 3D position [x, y, z] of the object in world space
        scale (list): Scale factors [sx, sy, sz] for the object
        rotation (list): Rotation angles [rx, ry, rz] in degrees around each axis
    """
    
    def __init__(self, vao, texture: Texture):
        """Initializes an object to be rendered via it's mesh VAO. The position, scale, and rotation are
        automatically set to pos=[0,0,0], scale=[1,1,1], and rotation=[0,0,0]

        Args:
            vao (Mesh VAO): From the mglw scene loader, the mesh.vao object
            texture (Texture): The texture associated with the object
        """

        # Store core rendering components
        self.vao = vao # Vertex Array Object containing mesh geometry
        self.texture = texture # Texture to apply to the object

        # Initialize transform properties
        self.position = [5, 0.0, 0.0]  # Starting position offset on X-axis
        self.scale = [1.0, 1.0, 1.0]  # Unit scale on all axes
        self.rotation = [0.0, 0.0, 0.0]  # No initial rotation

    def get_model_matrix(self) -> Matrix44:
        """Calculates and returns the 4x4 model matrix by computing M = T * R * S

        Returns:
            Matrix44: The model matrix
        """
        
        # Create individual transformation matrices
        Txyz = Matrix44.from_translation(self.position)  # Translation matrix
        Rx = Matrix44.from_x_rotation(radians(self.rotation[0]))  # X-axis rotation
        Ry = Matrix44.from_y_rotation(radians(self.rotation[1]))  # Y-axis rotation
        Rz = Matrix44.from_z_rotation(radians(self.rotation[2]))  # Z-axis rotation
        Sxyz = Matrix44.from_scale(self.scale)  # Scale matrix

        # Combine matrices in correct order (right-to-left multiplication)
        # First scale, then rotate (X, Y, Z order), then translate
        return Sxyz @ Rx @ Ry @ Rz @ Txyz

    def render(self, prog:Program, texture_unit=0, uv_scale=1.0):
        """Renders the object onto the scene.

        Args:
            prog (Program): The shader program to use
            texture_unit (int, optional): The texture unit location. Defaults to 0.
            uv_scale (float, optional): The uv scale to use. Defaults to 1.0.
        """
        
        # Bind the texture to the specified texture unit
        self.texture.use(location=texture_unit)
        
        # Set shader uniforms
        prog["Texture"] = texture_unit  # Texture sampler uniform
        prog["uv_scale"].value = uv_scale  # UV scale uniform
        
        # Update model matrix uniform with current transformation
        prog["model"].write(self.get_model_matrix().astype('f4').tobytes())
        
        # Render the mesh with the configured shader
        self.vao.render(prog)
