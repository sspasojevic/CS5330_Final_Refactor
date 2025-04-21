"""
Sara Spasojevic, Matej Zecic, Benjamin Northrop
3D Object Manipulation via Hand Gesture Recognition
CS5330 Final Project
April 20th 2025

Shader Program Module
-------------------
This module provides a manager for OpenGL shader programs using the ModernGL framework.
It handles the loading, compilation, storage, and retrieval of shader programs from
vertex and fragment shader source files. The module simplifies shader management by
providing a centralized repository for all shader programs used in a graphics application.

The ShaderProgram class offers a convenient interface for working with multiple shader
programs in a ModernGL-based rendering system.
"""

# Imports
from moderngl import Context, Program
from typing import Optional

class ShaderProgram:
    """
    A manager for OpenGL shader programs in a ModernGL context.
    
    This class handles the loading, compilation, and management of shader programs
    from vertex and fragment shader source files. It maintains a dictionary of 
    named shader programs that can be retrieved by name, allowing for organized
    shader management in graphics applications.
    
    Attributes:
        ctx (Context): The ModernGL context used for creating shader programs
        programs (dict): Dictionary mapping names to compiled shader programs
    """
    
    def __init__(self, ctx: Context):
        """Initializes the shader program to hold one or more programs

        Args:
            ctx (Context): The modernGL context
        """
        
        # Store the ModernGL context
        self.ctx = ctx
        
        # Dictionary to store compiled shader programs by name
        self.programs = {}

    def load_shader(self, name:str , vertex_path:str , fragment_path: str) -> Program:
        """
        Loads and compiles a shader program from source files, saves it by name.
        
        Reads vertex and fragment shader source code from files, compiles them into
        a shader program using the ModernGL context, and stores the program in the
        internal dictionary for later retrieval.
        
        Args:
            name (str): The shader program name for retrieval
            vertex_path (str): File path to the vertex shader source
            fragment_path (str): File path to the fragment shader source
        
        Returns:
            Program: The compiled shader program
        """
        
        # Return existing program if already loaded
        if name in self.programs:
            return self.programs[name]

        # Read vertex shader source code from file
        with open(vertex_path, 'r') as vert:
            vertex_src = vert.read()
            
        # Read fragment shader source code from file
        with open(fragment_path, 'r') as frag:
            fragment_src = frag.read()

        # Compile the shader program using ModernGL
        program = self.ctx.program(
            vertex_shader=vertex_src,
            fragment_shader=fragment_src,
        )
        
        # Store the compiled program in the dictionary
        self.programs[name] = program
        
        return program

    def get(self, name) -> Optional[Program]:
        """Returns the program from memory if it's found. Otherwise None

        Args:
            name (str): The name of the program

        Returns:
            Program | None: The shader program requested
        """
        
        # Return the program from the dictionary, or None if not found
        return self.programs.get(name)
