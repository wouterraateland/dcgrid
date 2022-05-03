#version 420 core

layout(location = 0) in vec2 InVertex;

void main()
{
	gl_Position = vec4(InVertex * 10, InVertex.y, 1.0);
}
