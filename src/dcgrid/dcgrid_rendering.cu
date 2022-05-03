#include "dcgrid.h"
#include "dcgrid_utils.cuh"
#include "raymarching.cuh"
#include "utils/grid_math.cuh"

inline __device__ void sampleCoarse(DCGrid *grid, const float3 &pos,
                                    MarchAux &aux) {
  aux.ipos = clampPos(make_int3(floorf(pos)));
  uint8_t level = 0;
  const size_t blockIndex = getBlockIndexDeep(grid, aux.ipos, level);
  aux.cellIndices[0] = DCGrid::blockV * blockIndex +
                       SPREAD(aux.ipos.x % DCGrid::blockW, 2) +
                       SPREAD(aux.ipos.y % DCGrid::blockW, 1) +
                       SPREAD(aux.ipos.z % DCGrid::blockW, 0);
  aux.cellIndices[1] = aux.cellIndices[0];
  aux.cellIndices[2] = aux.cellIndices[0];
  aux.cellIndices[3] = aux.cellIndices[0];
  aux.cellIndices[4] = aux.cellIndices[0];
  aux.cellIndices[5] = aux.cellIndices[0];
  aux.cellIndices[6] = aux.cellIndices[0];
  aux.cellIndices[7] = aux.cellIndices[0];
  aux.scale = 1 << level;
  aux.density = grid->density[aux.cellIndices[0]];
}

inline __device__ void samplePrecise(DCGrid *grid, const float3 &pos,
                                     MarchAux &aux) {
  aux.ipos = clampPos(make_int3(floorf(pos)));
  uint8_t level = 0;
  const size_t blockIndex = getBlockIndexDeep(grid, aux.ipos, level);
  aux.scale = 1 << level;

  const float scaleInv = 1.f / aux.scale;
  const float x = pos.x * scaleInv - .5f;
  const float y = pos.y * scaleInv - .5f;
  const float z = pos.z * scaleInv - .5f;
  const float xf = floorf(x), yf = floorf(y), zf = floorf(z);

  aux.fraction.x = x - xf;
  aux.fraction.y = y - yf;
  aux.fraction.z = z - zf;

  const int3 &bpos = grid->blockPositions[blockIndex];
  const size_t d = DCGrid::apronA * (xf + 1 - bpos.x) +
                   DCGrid::apronW * (yf + 1 - bpos.y) + (zf + 1 - bpos.z);
  const size_t *apronIndices =
      grid->cellIndices + DCGrid::apronV * blockIndex + d;

  aux.cellIndices[0] = apronIndices[0];
  aux.cellIndices[1] = apronIndices[1];
  aux.cellIndices[2] = apronIndices[DCGrid::apronW];
  aux.cellIndices[3] = apronIndices[DCGrid::apronW + 1];
  aux.cellIndices[4] = apronIndices[DCGrid::apronA];
  aux.cellIndices[5] = apronIndices[DCGrid::apronA + 1];
  aux.cellIndices[6] = apronIndices[DCGrid::apronA + DCGrid::apronW];
  aux.cellIndices[7] = apronIndices[DCGrid::apronA + DCGrid::apronW + 1];
  aux.density = aux.interpolate(grid->density);
}

__global__ void k_dcgrid_render(DCGrid *grid, cudaSurfaceObject_t dest,
                                int2 resolution, float3 eyeOrigin,
                                float3 eyeDir, float fov, float3 sunDir) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= resolution.x || y >= resolution.y)
    return;

  const float cx = .5f * resolution.x;
  const float cy = .5f * resolution.y;
  const float s = fov / fmaxf(cx, cy);
  const float step = 1.f / params.aa_samples;

  const float3 rayOrigin = eyeOrigin * params.rdx;
  float4 sample = make_float4(0.f, 0.f, 0.f, 0.f);
  for (float dx = 0; dx < 1.f; dx += step)
    for (float dy = 0; dy < 1.f; dy += step) {
      const float3 viewDir =
          normalize(make_float3((x + dx - cx) * s, (y + dy - cy) * s, 1.f));
      const float3 rayDir =
          make_float3(viewDir.x, viewDir.z * eyeDir.y - viewDir.y * eyeDir.z,
                      viewDir.y * eyeDir.y + viewDir.z * eyeDir.z);

      sample +=
          params.render_precise
              ? raymarch<DCGrid, samplePrecise>(grid, rayOrigin, rayDir, sunDir)
              : raymarch<DCGrid, sampleCoarse>(grid, rayOrigin, rayDir, sunDir);
    }
  sample *= step * step;

  surf2Dwrite(sample, dest, sizeof(float4) * x, y);
}
