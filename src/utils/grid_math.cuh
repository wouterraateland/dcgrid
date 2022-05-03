#pragma once
#include "helper_math.h"
#include <cuda_runtime.h>

inline __host__ __device__ int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define arrayIdx2d(px, pz) ((pz)*params.gx + (px))
#define arrayIdx3d(px, py, pz) (((pz)*params.gy + (py)) * params.gx + (px))

#define clampPos(p)                                                            \
  make_int3(clamp(p.x, 0, params.gx - 1), clamp(p.y, 0, params.gy - 1),        \
            clamp(p.z, 0, params.gz - 1))

inline __host__ __device__ size_t mipmapCells(const size_t &w,
                                              const size_t &h) {
  size_t numCells = 0;
  for (size_t s = 1; w % s == 0 && h % s == 0; s *= 2)
    numCells += (w * h) / (s * s);
  return numCells;
}

inline __host__ __device__ size_t mipmapCells(const size_t &w, const size_t &h,
                                              const size_t &d) {
  size_t numCells = 0;
  for (size_t s = 1; w % s == 0 && h % s == 0 && d % s == 0; s *= 2)
    numCells += (w * h * d) / (s * s * s);
  return numCells;
}

__inline__ __host__ __device__ size_t mipmapIdx(const int &x, const int &z,
                                                const size_t &w,
                                                const size_t &h,
                                                const size_t &scale) {
  size_t offset = 0;
  for (size_t s = 1; s < scale; s *= 2)
    offset += (w * h) / (s * s);

  return offset + z * (w / scale) + x;
}

inline __host__ __device__ size_t mipmapIdx(const int &x, const int &y,
                                            const int &z, const size_t &w,
                                            const size_t &h, const size_t &d,
                                            const size_t &scale) {
  size_t offset = 0;
  for (size_t s = 1; s < scale; s *= 2)
    offset += (w * h * d) / (s * s * s);

  return offset + (z * (h / scale) + y) * (w / scale) + x;
}
