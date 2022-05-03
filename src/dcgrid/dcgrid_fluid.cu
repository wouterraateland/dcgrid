#include "dcgrid.h"
#include "dcgrid_utils.cuh"
#include "utils/grid_math.cuh"
#include "utils/sim_utils.h"
#include <helper_math.h>

#define INIT_SAMPLE                                                            \
  int3 ipos = clampPos(make_int3(floorf(pos)));                                \
  uint8_t level = 0;                                                           \
  const size_t blockIndex = getBlockIndexDeep(grid, ipos, level);              \
                                                                               \
  const int scale = 1 << level;                                                \
  const float scaleInv = 1.f / scale;                                          \
  const float x = pos.x * scaleInv - .5f;                                      \
  const float y = pos.y * scaleInv - .5f;                                      \
  const float z = pos.z * scaleInv - .5f;                                      \
  const float xf = floorf(x);                                                  \
  const float yf = floorf(y);                                                  \
  const float zf = floorf(z);                                                  \
                                                                               \
  const float dx = x - xf;                                                     \
  const float dy = y - yf;                                                     \
  const float dz = z - zf;                                                     \
                                                                               \
  const int x0 = (int)xf;                                                      \
  const int y0 = (int)yf;                                                      \
  const int z0 = (int)zf;                                                      \
  const int x1 = x0 + 1;                                                       \
  const int y1 = y0 + 1;                                                       \
  const int z1 = z0 + 1;                                                       \
                                                                               \
  const int3 &blockPosition = grid->blockPositions[blockIndex];                \
  const int i = clamp(x1 - blockPosition.x, 0, DCGrid::apronW - 2);            \
  const int j = clamp(y1 - blockPosition.y, 0, DCGrid::apronW - 2);            \
  const int k = clamp(z1 - blockPosition.z, 0, DCGrid::apronW - 2);            \
                                                                               \
  const size_t *apronIndices = grid->cellIndices +                             \
                               DCGrid::apronV * blockIndex +                   \
                               DCGrid::apronA * i + DCGrid::apronW * j + k;    \
                                                                               \
  const size_t &i000 = apronIndices[0];                                        \
  const size_t &i001 = apronIndices[1];                                        \
  const size_t &i010 = apronIndices[DCGrid::apronW];                           \
  const size_t &i011 = apronIndices[DCGrid::apronW + 1];                       \
  const size_t &i100 = apronIndices[DCGrid::apronA];                           \
  const size_t &i101 = apronIndices[DCGrid::apronA + 1];                       \
  const size_t &i110 = apronIndices[DCGrid::apronA + DCGrid::apronW];          \
  const size_t &i111 = apronIndices[DCGrid::apronA + DCGrid::apronW + 1];      \
                                                                               \
  const float Dx = 1.f - dx;                                                   \
  const float Dy = 1.f - dy;                                                   \
  const float Dz = 1.f - dz;                                                   \
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

inline __device__ float3 advectVelocity(DCGrid *grid, const float3 &pos) {
  INIT_SAMPLE

  if (wAcc < 1e-6f)
    return make_float3(0.f);

  const float3 v000 = velocityBndCond(grid->velocity[i000], x0, y0, z0, scale);
  const float3 v001 = velocityBndCond(grid->velocity[i001], x0, y0, z1, scale);
  const float3 v010 = velocityBndCond(grid->velocity[i010], x0, y1, z0, scale);
  const float3 v011 = velocityBndCond(grid->velocity[i011], x0, y1, z1, scale);
  const float3 v100 = velocityBndCond(grid->velocity[i100], x1, y0, z0, scale);
  const float3 v101 = velocityBndCond(grid->velocity[i101], x1, y0, z1, scale);
  const float3 v110 = velocityBndCond(grid->velocity[i110], x1, y1, z0, scale);
  const float3 v111 = velocityBndCond(grid->velocity[i111], x1, y1, z1, scale);

  return v000 * w000 + v001 * w001 + v010 * w010 + v011 * w011 + v100 * w100 +
         v101 * w101 + v110 * w110 + v111 * w111;
}

inline __device__ float advectDensity(DCGrid *grid, const float3 &pos) {
  INIT_SAMPLE

  if (wAcc < 1e-6f)
    return 0.f;

  const float q000 = densityBndCond(grid->density[i000], x0, y0, z0, scale);
  const float q001 = densityBndCond(grid->density[i001], x0, y0, z1, scale);
  const float q010 = densityBndCond(grid->density[i010], x0, y1, z0, scale);
  const float q011 = densityBndCond(grid->density[i011], x0, y1, z1, scale);
  const float q100 = densityBndCond(grid->density[i100], x1, y0, z0, scale);
  const float q101 = densityBndCond(grid->density[i101], x1, y0, z1, scale);
  const float q110 = densityBndCond(grid->density[i110], x1, y1, z0, scale);
  const float q111 = densityBndCond(grid->density[i111], x1, y1, z1, scale);

  return q000 * w000 + q001 * w001 + q010 * w010 + q011 * w011 + q100 * w100 +
         q101 * w101 + q110 * w110 + q111 * w111;
}

__global__ void k_dcgrid_advect_velocity(DCGrid *grid) {
  USE_CELL_INDEX

  const int3 &p0 = grid->blockPositions[blockIndex];
  const float scale = 1 << grid->blockLevels[blockIndex];
  const float alpha = params.dt * params.rdx;

  if (grid->childIndices[subblockIndex] == DCGrid::notFound) {
    const float3 pos = make_float3(IDX2POS);
    const float3 posBacktraced =
        (pos + .5f) * scale - grid->velocity[cellIndex] * alpha;

    grid->t_velocity[cellIndex] = advectVelocity(grid, posBacktraced);
  } else
    grid->t_velocity[cellIndex] = make_float3(0.f, 0.f, 0.f);
}

__global__ void k_dcgrid_advect_density(DCGrid *grid) {
  USE_CELL_INDEX

  const int3 &p0 = grid->blockPositions[blockIndex];
  const float scale = 1 << grid->blockLevels[blockIndex];
  const float alpha = params.dt * params.rdx;

  if (grid->childIndices[subblockIndex] == DCGrid::notFound) {
    const float3 pos = make_float3(IDX2POS);
    const float3 posBacktraced =
        (pos + .5f) * scale - grid->velocity[cellIndex] * alpha;

    grid->t_density[cellIndex] = advectDensity(grid, posBacktraced);
  } else
    grid->t_density[cellIndex] = 0.f;
}

__global__ void k_dcgrid_calc_vorticity(DCGrid *grid) {
  USE_CELL_INDEX
  USE_APRON_INDEX

  const size_t *cellIndices = grid->cellIndices + DCGrid::apronV * blockIndex;
  const int scale = 1 << grid->blockLevels[blockIndex];
  const float alpha = .5f * params.rdx / scale;

  const float &wl = grid->fluidity[cellIndices[IDX_L]];
  const float &wr = grid->fluidity[cellIndices[IDX_R]];
  const float &wd = grid->fluidity[cellIndices[IDX_D]];
  const float &wu = grid->fluidity[cellIndices[IDX_U]];
  const float &wb = grid->fluidity[cellIndices[IDX_B]];
  const float &wf = grid->fluidity[cellIndices[IDX_F]];

  const float3 &vl = grid->velocity[cellIndices[IDX_L]];
  const float3 &vr = grid->velocity[cellIndices[IDX_R]];
  const float3 &vd = grid->velocity[cellIndices[IDX_D]];
  const float3 &vu = grid->velocity[cellIndices[IDX_U]];
  const float3 &vb = grid->velocity[cellIndices[IDX_B]];
  const float3 &vf = grid->velocity[cellIndices[IDX_F]];

  grid->vorticity[cellIndex] =
      alpha * make_float3((wu * vu.z - wd * vd.z) - (wf * vf.y - wb * vb.y),
                          (wf * vf.x - wb * vb.x) - (wr * vr.z - wl * vl.z),
                          (wr * vr.y - wl * vl.y) - (wu * vu.x - wd * vd.x));
}

__global__ void k_dcgrid_calc_divergence(DCGrid *grid) {
  __shared__ float3 velocity[DCGrid::apronV];

  USE_CELL_INDEX

  grid->pressure[cellIndex] = 0.f;
  grid->t_pressure[cellIndex] = 0.f;
  grid->divergence[cellIndex] = 0.f;

  const size_t *cellIndices = grid->cellIndices + DCGrid::apronV * blockIndex;
  const int scale = 1 << grid->blockLevels[blockIndex];
  USE_APRON_INDEX
  USE_CELL_POS

  velocity[apronIndex] = grid->velocity[cellIndex];
  if (apronX == 1)
    velocity[IDX_L] =
        velocityBndCond(grid->velocity[cellIndices[IDX_L]], x - 1, y, z, scale);
  if (apronY == 1)
    velocity[IDX_D] =
        velocityBndCond(grid->velocity[cellIndices[IDX_D]], x, y - 1, z, scale);
  if (apronZ == 1)
    velocity[IDX_B] =
        velocityBndCond(grid->velocity[cellIndices[IDX_B]], x, y, z - 1, scale);
  if (apronX == DCGrid::blockW)
    velocity[IDX_R] =
        velocityBndCond(grid->velocity[cellIndices[IDX_R]], x + 1, y, z, scale);
  if (apronY == DCGrid::blockW)
    velocity[IDX_U] =
        velocityBndCond(grid->velocity[cellIndices[IDX_U]], x, y + 1, z, scale);
  if (apronZ == DCGrid::blockW)
    velocity[IDX_F] =
        velocityBndCond(grid->velocity[cellIndices[IDX_F]], x, y, z + 1, scale);
  __syncthreads();

  if (grid->childIndices[subblockIndex] != DCGrid::notFound)
    return;

  const float alpha = .5f * params.rdx / scale;

  const float &wl = grid->fluidity[cellIndices[IDX_L]];
  const float &wr = grid->fluidity[cellIndices[IDX_R]];
  const float &wd = grid->fluidity[cellIndices[IDX_D]];
  const float &wu = grid->fluidity[cellIndices[IDX_U]];
  const float &wb = grid->fluidity[cellIndices[IDX_B]];
  const float &wf = grid->fluidity[cellIndices[IDX_F]];

  const float3 &vl = velocity[IDX_L];
  const float3 &vr = velocity[IDX_R];
  const float3 &vd = velocity[IDX_D];
  const float3 &vu = velocity[IDX_U];
  const float3 &vb = velocity[IDX_B];
  const float3 &vf = velocity[IDX_F];

  grid->divergence[cellIndex] = alpha * (wr * vr.x - wl * vl.x + wu * vu.y -
                                         wd * vd.y + wf * vf.z - wb * vb.z);
}

__global__ void k_dcgrid_apply_pressure(DCGrid *grid) {
  USE_CELL_INDEX
  if (grid->childIndices[subblockIndex] != DCGrid::notFound)
    return;

  const size_t *cellIndices = grid->cellIndices + DCGrid::apronV * blockIndex;
  USE_APRON_INDEX

  const int scale = 1 << grid->blockLevels[blockIndex];
  const float alpha = .5f * params.rdx / scale;
  const float &p = grid->pressure[cellIndex];

  grid->velocity[cellIndex].x -=
      alpha * (grid->fluidity[cellIndices[IDX_R]] *
                   (grid->pressure[cellIndices[IDX_R]] - p) +
               grid->fluidity[cellIndices[IDX_L]] *
                   (p - grid->pressure[cellIndices[IDX_L]]));
  grid->velocity[cellIndex].y -=
      alpha * (grid->fluidity[cellIndices[IDX_U]] *
                   (grid->pressure[cellIndices[IDX_U]] - p) +
               grid->fluidity[cellIndices[IDX_D]] *
                   (p - grid->pressure[cellIndices[IDX_D]]));
  grid->velocity[cellIndex].z -=
      alpha * (grid->fluidity[cellIndices[IDX_F]] *
                   (grid->pressure[cellIndices[IDX_F]] - p) +
               grid->fluidity[cellIndices[IDX_B]] *
                   (p - grid->pressure[cellIndices[IDX_B]]));
}
