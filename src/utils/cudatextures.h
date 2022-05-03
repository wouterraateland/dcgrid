#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <gl/GL.h>

struct cudaGraphicsResource;

class TextureResource {
public:
  GLuint m_GLTexture;
  cudaGraphicsResource *m_CudaResource;

  int getInternalFormat(const size_t &numChannels);
  int getFormat(const size_t &numChannels);

  void init(const int2 &resolution, const size_t &numChannels,
            const GLint &wrap = GL_CLAMP_TO_BORDER,
            const float &borderColor = 0.f);
  void init(const int3 &resolution, const size_t &numChannels,
            const GLint &wrap = GL_CLAMP_TO_BORDER,
            const float &borderColor = 0.f);
  void destroy();

  cudaSurfaceObject_t mapAsSurface();
  cudaTextureObject_t mapAsTexture(const bool &normalizedCoords = true);
  void unmap();

private:
  bool m_mapped;
  bool m_mappedAsSurface;

  cudaSurfaceObject_t m_mappedSurface;
  cudaTextureObject_t m_mappedTexture;
};

class DoubleTextureResource {
private:
  TextureResource res0, res1;
  TextureResource *current, *next;

public:
  ~DoubleTextureResource() { destroy(); }
  void init(const int3 &resolution, const size_t &numChannels,
            const GLint &wrap = GL_CLAMP_TO_BORDER,
            const float &borderColor = 0.f);
  void destroy() {
    res0.destroy();
    res1.destroy();
  }

  cudaTextureObject_t mapCurrentAsTexture();
  cudaSurfaceObject_t mapCurrentAsSurface();
  cudaTextureObject_t mapNextAsTexture();
  cudaSurfaceObject_t mapNextAsSurface();
  void unmapAll();

  GLuint getCurrentGLTexture() { return current->m_GLTexture; }

  void swap() { std::swap(current, next); }
};
