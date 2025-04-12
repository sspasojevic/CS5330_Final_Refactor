from moderngl import Context, Program
from typing import Optional

class ShaderProgram:
    def __init__(self, ctx: Context):
        """Initializes the shader program to hold one or more programs

        Args:
            ctx (Context): The modernGL context
        """
        self.ctx = ctx
        self.programs = {}

    def load_shader(self, name:str , vertex_path:str , fragment_path: str) -> Program:
        """Loads and compiles a shader program from files, saves it by name

        Args:
            name (str): the shader program name
            vertex_path (str): path to the vertex shader
            fragment_path (str): path to the fragment shader

        Returns: the shader program
        """
        if name in self.programs:
            return self.programs[name]

        with open(vertex_path, 'r') as vert:
            vertex_src = vert.read()
        with open(fragment_path, 'r') as frag:
            fragment_src = frag.read()

        program = self.ctx.program(
            vertex_shader=vertex_src,
            fragment_shader=fragment_src,
        )
        self.programs[name] = program
        return program

    def get(self, name) -> Optional[Program]:
        """Returns the program from memory if it's found. Otherwise None

        Args:
            name (str): The name of the program

        Returns:
            Program | None: The shader program requested
        """
        return self.programs.get(name)
