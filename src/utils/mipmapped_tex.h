#pragma once
#include "helper_math.h"
#include "utils/grid_math.cuh"
#include <boost/filesystem.hpp>
#include <stb/stb_image.h>

#undef STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>
#include <stb/stb_image_write.h>

template <class T> class MipmappedTex {
public:
  MipmappedTex(const int2 resolution);
  ~MipmappedTex();

  void loadFromBuffer(T *src);
  void loadFromFile(const char *filename);
  void generateMipmap();

  size_t getIdx(const size_t &x, const size_t &y,
                const size_t &level = 0) const;
  T &get(const size_t &x, const size_t &y, const size_t &level = 0) const;
  void set(const size_t &x, const size_t &y, const T &value,
           const size_t &level = 0);
  void saveToJPG(const char *filename) const;

  int2 getResolution() const;

  size_t width = 0;
  size_t height = 0;
  size_t numCells = 0;
  T *data = nullptr;
};

template <class T>
MipmappedTex<T>::MipmappedTex(const int2 resolution)
    : width(resolution.x), height(resolution.y),
      numCells(mipmapCells(width, height)) {
  data = new T[numCells];
}

template <class T> MipmappedTex<T>::~MipmappedTex() { delete[] data; }

template <class T> void MipmappedTex<T>::loadFromFile(const char *filename) {
  struct stat buffer;
  const int numChannels = sizeof(T) / sizeof(float);
  if (stat(filename, &buffer) == 0) {
    stbi_set_flip_vertically_on_load(0);
    int srcWidth;
    int srcHeight;
    int srcNumChannels;
    float *srcData = stbi_loadf(filename, &srcWidth, &srcHeight,
                                &srcNumChannels, numChannels);

    stbir_resize_float(srcData, srcWidth, srcHeight, 0, (float *)data, width,
                       height, 0, numChannels);

    delete[] srcData;

    generateMipmap();
  }
}

template <class T> void MipmappedTex<T>::loadFromBuffer(T *src) {
  memcpy(data, src, numCells * sizeof(T));
}

template <class T> void MipmappedTex<T>::generateMipmap() {
  size_t iSrc = 0;
  size_t iDest = width * height;
  for (int s = 2; width % s == 0 && height % s == 0; s *= 2)
    for (int y = 0; y < height / s; y++, iSrc += width / (s / 2))
      for (int x = 0; x < width / s; x++, iSrc += 2, iDest += 1) {
        const T &v00 = data[iSrc];
        const T &v01 = data[iSrc + 1];
        const T &v10 = data[iSrc + width / (s / 2)];
        const T &v11 = data[iSrc + width / (s / 2) + 1];
        data[iDest] = (v00 + v01 + v10 + v11) * .25f;
      }
}

template <class T>
size_t MipmappedTex<T>::getIdx(const size_t &x, const size_t &y,
                               const size_t &level) const {
  const size_t scale = 1 << level;
  size_t offset = 0;
  for (size_t s = 1; s < scale; s *= 2)
    offset += (width * height) / (s * s);

  return offset + (y / scale) * (width / scale) + (x / scale);
}

template <class T>
T &MipmappedTex<T>::get(const size_t &x, const size_t &y,
                        const size_t &level) const {
  return data[getIdx(x, y, level)];
}

template <class T>
void MipmappedTex<T>::set(const size_t &x, const size_t &y, const T &value,
                          const size_t &level) {
  data[getIdx(x, y, level)] = value;
}

template <class T> int2 MipmappedTex<T>::getResolution() const {
  return make_int2(width, height);
}

template <class T> void MipmappedTex<T>::saveToJPG(const char *filename) const {
  const int numChannels = sizeof(T) / sizeof(float);
  uint8_t *dest =
      (uint8_t *)malloc(width * height * numChannels * sizeof(uint8_t));
  float *floatData = (float *)data;

  for (size_t i = 0; i < width * height * numChannels; i++)
    dest[i] = int(floatData[i]);

  stbi_flip_vertically_on_write(0);
  stbi_write_jpg(filename, width, height, numChannels, dest,
                 width * numChannels * sizeof(int));

  free(dest);
}