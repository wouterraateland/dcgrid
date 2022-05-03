#pragma once
#include "utils/grid_math.cuh"
#include <cuda_runtime.h>

#define INIT_XYZ                                                               \
  const int x = blockIdx.x * blockDim.x + threadIdx.x;                         \
  const int y = blockIdx.y * blockDim.y + threadIdx.y;                         \
  const int z = blockIdx.z * blockDim.z + threadIdx.z;                         \
  if (x >= params.gx || y >= params.gy || z >= params.gz)                      \
    return;

#define INIT_XYZ_MIPMAP                                                        \
  const int x = blockIdx.x * blockDim.x + threadIdx.x;                         \
  const int y = blockIdx.y * blockDim.y + threadIdx.y;                         \
  const int z = blockIdx.z * blockDim.z + threadIdx.z;                         \
  if (x >= (params.gx >> mipmapLevel) || y >= (params.gy >> mipmapLevel) ||    \
      z >= (params.gz >> mipmapLevel))                                         \
    return;

#define INIT_IDX const size_t idx = arrayIdx3d(x, y, z);

#define INIT_STENCIL_IDX                                                       \
  const size_t idx = arrayIdx3d(x, y, z);                                      \
  const size_t idxl = x > 0 ? idx - 1 : idx;                                   \
  const size_t idxr = x < params.gx - 1 ? idx + 1 : idx;                       \
  const size_t idxd = y > 0 ? idx - params.gx : idx;                           \
  const size_t idxu = y < params.gy - 1 ? idx + params.gx : idx;               \
  const size_t idxb = z > 0 ? idx - params.gx * params.gy : idx;               \
  const size_t idxf = z < params.gz - 1 ? idx + params.gx * params.gy : idx;

struct UniformGrid {
  // Permanent
  float *density = nullptr;
  float3 *velocity = nullptr;
  float *fluidity = nullptr;
  float *pressure = nullptr;

  // Temporary
  void *temporary = nullptr;

  // Aliasses of temporary
  float3 *t_velocity = nullptr;
  float *t_pressure = nullptr;
  float *divergence = nullptr;
  float *t_density = nullptr;
};

__global__ void k_uniform_init_aliasses(UniformGrid *grid);
__global__ void k_uniform_init(UniformGrid *grid);
__global__ void k_uniform_set_solidity_ratio(UniformGrid *grid,
                                             int mipmapLevel);
__global__ void k_uniform_advect_velocity(UniformGrid *grid);
__global__ void k_uniform_calc_divergence(UniformGrid *grid);
__global__ void k_uniform_restrict(UniformGrid *grid, int mipmapLevel);
__global__ void k_uniform_jacobi(UniformGrid *grid, int mipmapLevel);
__global__ void k_uniform_jacobi_inv(UniformGrid *grid, int mipmapLevel);
__global__ void k_uniform_prolongate(UniformGrid *grid, int mipmapLevel);
__global__ void k_uniform_apply_pressure(UniformGrid *grid);
__global__ void k_uniform_advect_density(UniformGrid *grid);

__global__ void k_uniform_render(UniformGrid *grid, cudaSurfaceObject_t dest,
                                 int2 resolution, float3 eyeOrigin,
                                 float3 eyeDir, float fov, float3 sunDir);

__global__ void k_uniform_debug_stats(UniformGrid *grid, float *stats,
                                      size_t numBins);