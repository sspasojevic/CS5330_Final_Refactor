// Vertex Shader
#version 330  // Using GLSL version 330 (OpenGL 3.3)

// Transformation matrices
uniform mat4 model;      // Model matrix - transforms vertices from model to world space
uniform mat4 view;       // View matrix - transforms from world to camera space
uniform mat4 proj;       // Projection matrix - transforms from camera to clip space
uniform float uv_scale;  // Scale factor for texture coordinates

// Input vertex attributes
in vec3 in_position;     // Vertex position in model space
in vec2 in_texcoord_0;   // Texture coordinates for the first UV map

// Output variables to fragment shader
out vec2 v_uv;           // Texture coordinates passed to fragment shader

void main() {
    // Scale the texture coordinates and pass to fragment shader
    v_uv = in_texcoord_0 * uv_scale;
    
    // Transform vertex position through the full MVP matrix chain
    // The multiplication order is important: first model, then view, then projection
    gl_Position = proj * view * model * vec4(in_position, 1.0);
}