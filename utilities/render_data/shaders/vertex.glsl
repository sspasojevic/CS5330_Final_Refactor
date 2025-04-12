#version 330

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform float uv_scale;

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 v_uv;

void main() {
    v_uv = in_texcoord_0 * uv_scale;
    gl_Position = proj * view * model * vec4(in_position, 1.0);
}
