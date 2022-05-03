#include "cudatextures.h"
#include <cuda_gl_interop.h>
#include <helper_cuda.h>

#include <glm/gtc/random.hpp>

int TextureResource::getInternalFormat(const size_t &numChannels) {
  switch (numChannels) {
  case 2:
    return GL_RG32F;
  case 3:
    return GL_RGB32F;
  case 4:
    return GL_RGBA32F;
  case 1:
  default:
    return GL_R32F;
  }
}

int TextureResource::getFormat(const size_t &numChannels) {
  switch (numChannels) {
  case 2:
    return GL_RG;
  case 3:
    return GL_RGB;
  case 4:
    return GL_RGBA;
  case 1:
  default:
    return GL_RED;
  }
}

void TextureResource::init(const int2 &resolution, const size_t &numChannels,
                           const GLint &wrap, const float &borderColor) {
  glGenTextures(1, &m_GLTexture);
  glBindTexture(GL_TEXTURE_2D, m_GLTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, wrap);
  float borderColorPerSide[] = {borderColor, borderColor, borderColor,
                                borderColor};
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColorPerSide);

  const int internalFormat = getInternalFormat(numChannels);
  const int format = getFormat(numChannels);

  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, resolution.x, resolution.y, 0,
               format, GL_FLOAT, 0);

  checkCudaErrors(
      cudaGraphicsGLRegisterImage(&m_CudaResource, m_GLTexture, GL_TEXTURE_2D,
                                  cudaGraphicsRegisterFlagsSurfaceLoadStore));

  glBindTexture(GL_TEXTURE_2D, 0);
  m_mapped = false;
}

void TextureResource::init(const int3 &resolution, const size_t &numChannels,
                           const GLint &wrap, const float &borderColor) {
  glGenTextures(1, &m_GLTexture);
  glBindTexture(GL_TEXTURE_3D, m_GLTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrap);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrap);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap);
  float borderColorPerSide[] = {borderColor, borderColor, borderColor,
                                borderColor};
  glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColorPerSide);

  const int internalFormat = getInternalFormat(numChannels);
  const int format = getFormat(numChannels);

  glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, resolution.x, resolution.y,
               resolution.z, 0, format, GL_FLOAT, 0);

  checkCudaErrors(
      cudaGraphicsGLRegisterImage(&m_CudaResource, m_GLTexture, GL_TEXTURE_3D,
                                  cudaGraphicsRegisterFlagsSurfaceLoadStore));

  glBindTexture(GL_TEXTURE_3D, 0);
  m_mapped = false;
}

void TextureResource::destroy() {
  cudaGraphicsUnregisterResource(m_CudaResource);
  glDeleteTextures(1, &m_GLTexture);
}

cudaSurfaceObject_t TextureResource::mapAsSurface() {
  checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaResource));

  cudaArray_t viewCudaArray;
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray,
                                                        m_CudaResource, 0, 0));

  cudaResourceDesc viewCudaArrayResourceDesc;
  memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
  {
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
  }
  checkCudaErrors(
      cudaCreateSurfaceObject(&m_mappedSurface, &viewCudaArrayResourceDesc));

  m_mapped = true;
  m_mappedAsSurface = true;

  return m_mappedSurface;
}

cudaTextureObject_t
TextureResource::mapAsTexture(const bool &normalizedCoords) {
  checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaResource));

  cudaArray_t viewCudaArray;
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray,
                                                        m_CudaResource, 0, 0));

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = viewCudaArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.normalizedCoords = normalizedCoords ? 1 : 0;

  checkCudaErrors(
      cudaCreateTextureObject(&m_mappedTexture, &resDesc, &texDesc, NULL));

  m_mapped = true;
  m_mappedAsSurface = false;

  return m_mappedTexture;
}

void TextureResource::unmap() {
  if (!m_mapped)
    return;

  if (m_mappedAsSurface)
    checkCudaErrors(cudaDestroySurfaceObject(m_mappedSurface));
  else
    checkCudaErrors(cudaDestroyTextureObject(m_mappedTexture));

  checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaResource));

  m_mapped = false;
}

void DoubleTextureResource::init(const int3 &resolution,
                                 const size_t &numChannels, const GLint &wrap,
                                 const float &borderColor) {
  res0.init(resolution, numChannels, wrap, borderColor);
  res1.init(resolution, numChannels, wrap, borderColor);
  current = &res0;
  next = &res1;
}

cudaTextureObject_t DoubleTextureResource::mapCurrentAsTexture() {
  return current->mapAsTexture();
}

cudaSurfaceObject_t DoubleTextureResource::mapCurrentAsSurface() {
  return current->mapAsSurface();
}

cudaTextureObject_t DoubleTextureResource::mapNextAsTexture() {
  return next->mapAsTexture();
}

cudaSurfaceObject_t DoubleTextureResource::mapNextAsSurface() {
  return next->mapAsSurface();
}

void DoubleTextureResource::unmapAll() {
  current->unmap();
  next->unmap();
}
