// Fragment Shader
#version 330  // Using GLSL version 330 (OpenGL 3.3)

// Uniforms (values provided by the application)
uniform sampler2D Texture;  // Input texture to sample from

// Input variables from the vertex shader
in vec2 v_uv;  // Texture coordinates

// Output variables
out vec4 f_color;  // Final fragment color output

// Main shader function
void main() {
    // Sample the texture at the provided UV coordinates
    // and assign the resulting color to the fragment
    f_color = texture(Texture, v_uv);
}