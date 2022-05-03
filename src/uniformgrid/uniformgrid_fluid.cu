#include "uniformgrid.h"
#include "utils/grid_math.cuh"
#include "utils/sim_utils.h"
#include <helper_cuda.h>
#include <helper_math.h>

#define INIT_SAMPLE                                                            \
  const float x = pos.x - .5f, y = pos.y - .5f, z = pos.z - .5f;               \
  const float xf = floorf(x), yf = floorf(y), zf = floorf(z);                  \
  const int x0 = (int)xf, y0 = (int)yf, z0 = (int)zf;                          \
  const int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;                             \
  const int x0c = clamp(x0, 0, params.gx - 1);                                 \
  const int y0c = clamp(y0, 0, params.gy - 1);                                 \
  const int z0c = clamp(z0, 0, params.gz - 1);                                 \
  const int x1c = clamp(x1, 0, params.gx - 1);                                 \
  const int y1c = clamp(y1, 0, params.gy - 1);                                 \
  const int z1c = clamp(z1, 0, params.gz - 1);                                 \
  const size_t i000 = arrayIdx3d(x0c, y0c, z0c);                               \
  const size_t i001 = arrayIdx3d(x0c, y0c, z1c);                               \
  const size_t i010 = arrayIdx3d(x0c, y1c, z0c);                               \
  const size_t i011 = arrayIdx3d(x0c, y1c, z1c);                               \
  const size_t i100 = arrayIdx3d(x1c, y0c, z0c);                               \
  const size_t i101 = arrayIdx3d(x1c, y0c, z1c);                               \
  const size_t i110 = arrayIdx3d(x1c, y1c, z0c);                               \
  const size_t i111 = arrayIdx3d(x1c, y1c, z1c);                               \
                                                                               \
  const float dx = x - xf, dy = y - yf, dz = z - zf;                           \
  const float Dx = 1.f - dx, Dy = 1.f - dy, Dz = 1.f - dz;                     \
                                                                               \
  float w000 = grid->fluidity[i000] * Dx * Dy * Dz;                            \
  float w001 = grid->fluidity[i001] * Dx * Dy * dz;                            \
  float w010 = grid->fluidity[i010] * Dx * dy * Dz;                            \
  float w011 = grid->fluidity[i011] * Dx * dy * dz;                            \
  float w100 = grid->fluidity[i100] * dx * Dy * Dz;                            \
  float w101 = grid->fluidity[i101] * dx * Dy * dz;                            \
  float w110 = grid->fluidity[i110] * dx * dy * Dz;                            \
  float w111 = grid->fluidity[i111] * dx * dy * dz;                            \
  const float wAcc = w000 + w001 + w010 + w011 + w100 + w101 + w110 + w111;    \
  const float wInv = 1.f / wAcc;                                               \
                                                                               \
  w000 *= wInv;                                                                \
  w001 *= wInv;                                                                \
  w010 *= wInv;                                                                \
  w011 *= wInv;                                                                \
  w100 *= wInv;                                                                \
  w101 *= wInv;                                                                \
  w110 *= wInv;                                                                \
  w111 *= wInv;

inline __device__ float3 advectVelocity(UniformGrid *grid, const float3 &pos) {
  INIT_SAMPLE

  if (wAcc < 1e-6f)
    return make_float3(0.f);

  const float3 v000 = velocityBndCond(grid->velocity[i000], x0, y0, z0);
  const float3 v001 = velocityBndCond(grid->velocity[i001], x0, y0, z1);
  const float3 v010 = velocityBndCond(grid->velocity[i010], x0, y1, z0);
  const float3 v011 = velocityBndCond(grid->velocity[i011], x0, y1, z1);
  const float3 v100 = velocityBndCond(grid->velocity[i100], x1, y0, z0);
  const float3 v101 = velocityBndCond(grid->velocity[i101], x1, y0, z1);
  const float3 v110 = velocityBndCond(grid->velocity[i110], x1, y1, z0);
  const float3 v111 = velocityBndCond(grid->velocity[i111], x1, y1, z1);

  return v000 * w000 + v001 * w001 + v010 * w010 + v011 * w011 + v100 * w100 +
         v101 * w101 + v110 * w110 + v111 * w111;
}

inline __device__ float advectDensity(UniformGrid *grid, const float3 &pos) {
  INIT_SAMPLE

  if (wAcc < 1e-6f)
    return 0.f;

  const float q000 = densityBndCond(grid->density[i000], x0, y0, z0);
  const float q001 = densityBndCond(grid->density[i001], x0, y0, z1);
  const float q010 = densityBndCond(grid->density[i010], x0, y1, z0);
  const float q011 = densityBndCond(grid->density[i011], x0, y1, z1);
  const float q100 = densityBndCond(grid->density[i100], x1, y0, z0);
  const float q101 = densityBndCond(grid->density[i101], x1, y0, z1);
  const float q110 = densityBndCond(grid->density[i110], x1, y1, z0);
  const float q111 = densityBndCond(grid->density[i111], x1, y1, z1);

  return q000 * w000 + q001 * w001 + q010 * w010 + q011 * w011 + q100 * w100 +
         q101 * w101 + q110 * w110 + q111 * w111;
}

__global__ void k_uniform_advect_velocity(UniformGrid *grid) {
  INIT_XYZ
  INIT_IDX

  const float3 backtracedPos =
      make_float3(x, y, z) + .5f - grid->velocity[idx] * params.dt * params.rdx;
  grid->t_velocity[idx] = advectVelocity(grid, backtracedPos);
}

__global__ void k_uniform_advect_density(UniformGrid *grid) {
  INIT_XYZ
  INIT_IDX

  const float3 backtracedPos =
      make_float3(x, y, z) + .5f - grid->velocity[idx] * params.dt * params.rdx;

  grid->t_density[idx] = advectDensity(grid, backtracedPos);
}

__global__ void k_uniform_calc_divergence(UniformGrid *grid) {
  INIT_XYZ
  INIT_STENCIL_IDX

  grid->pressure[idx] = 0.f;
  grid->t_pressure[idx] = 0.f;
  grid->divergence[idx] = 0.f;

  const float &wl = grid->fluidity[idxl];
  const float &wr = grid->fluidity[idxr];
  const float &wd = grid->fluidity[idxd];
  const float &wu = grid->fluidity[idxu];
  const float &wb = grid->fluidity[idxb];
  const float &wf = grid->fluidity[idxf];

  const float3 vl = velocityBndCond(grid->velocity[idxl], x - 1, y, z);
  const float3 vr = velocityBndCond(grid->velocity[idxr], x + 1, y, z);
  const float3 vd = velocityBndCond(grid->velocity[idxd], x, y - 1, z);
  const float3 vu = velocityBndCond(grid->velocity[idxu], x, y + 1, z);
  const float3 vb = velocityBndCond(grid->velocity[idxb], x, y, z - 1);
  const float3 vf = velocityBndCond(grid->velocity[idxf], x, y, z + 1);

  grid->divergence[idx] =
      .5f * params.rdx *
      (wr * vr.x - wl * vl.x + wu * vu.y - wd * vd.y + wf * vf.z - wb * vb.z);
}

__global__ void k_uniform_restrict(UniformGrid *grid, int mipmapLevel) {
  INIT_XYZ_MIPMAP
  const int scale = 1 << mipmapLevel;
  const int childScale = scale / 2;
  const int w = params.gx / childScale;
  const int h = params.gy / childScale;

  const size_t idx = mipmapIdx(x, y, z, params.gx, params.gy, params.gz, scale);

  const size_t i000 = mipmapIdx(x * 2, y * 2, z * 2, params.gx, params.gy,
                                params.gz, childScale);
  const size_t i001 = i000 + 1;
  const size_t i010 = i000 + w;
  const size_t i011 = i010 + 1;
  const size_t i100 = i000 + w * h;
  const size_t i101 = i100 + 1;
  const size_t i110 = i100 + w;
  const size_t i111 = i110 + 1;

  grid->pressure[idx] = 0.f;
  grid->t_pressure[idx] = 0.f;
  grid->divergence[idx] =
      .125f * (grid->divergence[i000] + grid->divergence[i001] +
               grid->divergence[i010] + grid->divergence[i011] +
               grid->divergence[i100] + grid->divergence[i101] +
               grid->divergence[i110] + grid->divergence[i111]);
}

template <float *UniformGrid::*inputPtr, float *UniformGrid::*outputPtr>
inline __device__ void calcPressure(UniformGrid *grid, const int &x,
                                    const int &y, const int &z,
                                    const int &mipmapLevel) {
  const int scale = 1 << mipmapLevel;
  const size_t idx = mipmapIdx(x, y, z, params.gx, params.gy, params.gz, scale);
  const float *inputField = grid->*inputPtr;
  float &output = (grid->*outputPtr)[idx];

  const float alpha = params.dx * params.dx * scale * scale;

  const int w = params.gx / scale;
  const int h = params.gy / scale;
  const int d = params.gz / scale;

  const size_t idxl = x > 0 ? idx - 1 : idx;
  const size_t idxr = x < w - 1 ? idx + 1 : idx;
  const size_t idxd = y > 0 ? idx - w : idx;
  const size_t idxu = y < h - 1 ? idx + w : idx;
  const size_t idxb = z > 0 ? idx - w * h : idx;
  const size_t idxf = z < d - 1 ? idx + w * h : idx;

  const float &pl = inputField[idxl];
  const float &pr = inputField[idxr];
  const float &pd = inputField[idxd];
  const float &pu = inputField[idxu];
  const float &pb = inputField[idxb];
  const float &pf = inputField[idxf];

  output = (pl + pr + pd + pu + pb + pf - alpha * grid->divergence[idx]) / 6.f;
}

__global__ void k_uniform_jacobi(UniformGrid *grid, int mipmapLevel) {
  INIT_XYZ_MIPMAP
  calcPressure<&UniformGrid::pressure, &UniformGrid::t_pressure>(grid, x, y, z,
                                                                 mipmapLevel);
}

__global__ void k_uniform_jacobi_inv(UniformGrid *grid, int mipmapLevel) {
  INIT_XYZ_MIPMAP
  calcPressure<&UniformGrid::t_pressure, &UniformGrid::pressure>(grid, x, y, z,
                                                                 mipmapLevel);
}

__global__ void k_uniform_prolongate(UniformGrid *grid, int mipmapLevel) {
  INIT_XYZ_MIPMAP
  const int scale = 1 << mipmapLevel;
  const int w = params.gx / scale;
  const int h = params.gy / scale;
  const int d = params.gz / scale;

  const size_t idx = mipmapIdx(x, y, z, params.gx, params.gy, params.gz, scale);

  const size_t i000 = mipmapIdx(x / 2, y / 2, z / 2, params.gx, params.gy,
                                params.gz, 2 * scale);

  const int sx = (x == 0 || x == w - 1) ? 0 : 2 * (x % 2) - 1;
  const int sy = (y == 0 || y == h - 1) ? 0 : 2 * (y % 2) - 1;
  const int sz = (z == 0 || z == d - 1) ? 0 : 2 * (z % 2) - 1;

  const size_t i001 = i000 + sx;
  const size_t i010 = i000 + sy * w / 2;
  const size_t i011 = i010 + sx;
  const size_t i100 = i000 + sz * (w / 2) * (h / 2);
  const size_t i101 = i100 + sx;
  const size_t i110 = i100 + sy * (w / 2);
  const size_t i111 = i110 + sx;

  grid->pressure[idx] = (27.f * grid->pressure[i000] +
                         9.f * (grid->pressure[i001] + grid->pressure[i010] +
                                grid->pressure[i100]) +
                         3.f * (grid->pressure[i011] + grid->pressure[i101] +
                                grid->pressure[i110]) +
                         grid->pressure[i111]) /
                        64.f;
}

__global__ void k_uniform_apply_pressure(UniformGrid *grid) {
  INIT_XYZ
  INIT_STENCIL_IDX

  const float alpha = .5f * params.rdx;

  const float &p = grid->pressure[idx];
  const float &pl = grid->pressure[idxl];
  const float &pr = grid->pressure[idxr];
  const float &pd = grid->pressure[idxd];
  const float &pu = grid->pressure[idxu];
  const float &pb = grid->pressure[idxb];
  const float &pf = grid->pressure[idxf];

  float3 &velocity = grid->velocity[idx];
  velocity.x -= alpha * (grid->fluidity[idxr] * (pr - p) +
                         grid->fluidity[idxl] * (p - pl));
  velocity.y -= alpha * (grid->fluidity[idxu] * (pu - p) +
                         grid->fluidity[idxd] * (p - pd));
  velocity.z -= alpha * (grid->fluidity[idxf] * (pf - p) +
                         grid->fluidity[idxb] * (p - pb));
}
