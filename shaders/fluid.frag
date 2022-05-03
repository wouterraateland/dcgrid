#version 420 core
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_bindless_texture : require

#include "../extern/L/shaders/fwd_ubo_rendering.glsl"

uniform sampler2D uTex;

layout(location = 0) out vec4 color;

void main()
{
	color = texture(uTex, gl_FragCoord.xy / uViewportDim);
}
