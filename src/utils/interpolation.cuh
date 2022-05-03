#pragma once
#include <cuda_runtime.h>

template <typename T>
inline __device__ T bilinearInterpolation(const float &dx, const float &dz,
                                          const T &tex00, const T &tex01,
                                          const T &tex10, const T &tex11) {
  const float dxInv = 1.f - dx;
  const T c0 = tex00 * dxInv + tex10 * dx;
  const T c1 = tex01 * dxInv + tex11 * dx;

  return c0 * (1.f - dz) + c1 * dz;
}

template <typename T>
inline __device__ T trilinearInterpolation(const float &dx, const float &dy,
                                           const float &dz, const T &tex000,
                                           const T &tex001, const T &tex010,
                                           const T &tex011, const T &tex100,
                                           const T &tex101, const T &tex110,
                                           const T &tex111) {
  const float dxInv = 1.f - dx;
  const T c00 = tex000 * dxInv + tex100 * dx;
  const T c01 = tex001 * dxInv + tex101 * dx;
  const T c10 = tex010 * dxInv + tex110 * dx;
  const T c11 = tex011 * dxInv + tex111 * dx;

  const float dyInv = 1.f - dy;
  const T c0 = c00 * dyInv + c10 * dy;
  const T c1 = c01 * dyInv + c11 * dy;

  return c0 * (1.f - dz) + c1 * dz;
}