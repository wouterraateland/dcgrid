#include "raymarching.cuh"
#include "uniformgrid.h"
#include "utils/grid_math.cuh"

inline __device__ void sampleCoarse(UniformGrid *grid, const float3 &pos,
                                    MarchAux &aux) {
  aux.ipos = clampPos(make_int3(pos));
  aux.cellIndices[0] = arrayIdx3d(aux.ipos.x, aux.ipos.y, aux.ipos.z);
  aux.cellIndices[1] = aux.cellIndices[0];
  aux.cellIndices[2] = aux.cellIndices[0];
  aux.cellIndices[3] = aux.cellIndices[0];
  aux.cellIndices[4] = aux.cellIndices[0];
  aux.cellIndices[5] = aux.cellIndices[0];
  aux.cellIndices[6] = aux.cellIndices[0];
  aux.cellIndices[7] = aux.cellIndices[0];
  aux.density = grid->density[aux.cellIndices[0]];
}

inline __device__ void samplePrecise(UniformGrid *grid, const float3 &pos,
                                     MarchAux &aux) {
  const float x = pos.x - .5f, y = pos.y - .5f, z = pos.z - .5f;
  const float xf = floorf(x), yf = floorf(y), zf = floorf(z);

  aux.ipos.x = xf;
  aux.ipos.y = yf;
  aux.ipos.z = zf;

  aux.fraction.x = x - xf;
  aux.fraction.y = y - yf;
  aux.fraction.z = z - zf;

  const int x0c = clamp(aux.ipos.x, 0, params.gx - 1);
  const int y0c = clamp(aux.ipos.y, 0, params.gy - 1);
  const int z0c = clamp(aux.ipos.z, 0, params.gz - 1);
  const int xs = aux.ipos.x < params.gx - 1 ? 1 : 0;
  const int ys = aux.ipos.y < params.gy - 1 ? params.gx : 0;
  const int zs = aux.ipos.z < params.gz - 1 ? params.gx * params.gy : 0;

  aux.cellIndices[0] = arrayIdx3d(x0c, y0c, z0c);
  aux.cellIndices[1] = aux.cellIndices[0] + zs;
  aux.cellIndices[2] = aux.cellIndices[0] + ys;
  aux.cellIndices[3] = aux.cellIndices[1] + ys;
  aux.cellIndices[4] = aux.cellIndices[0] + xs;
  aux.cellIndices[5] = aux.cellIndices[4] + zs;
  aux.cellIndices[6] = aux.cellIndices[4] + ys;
  aux.cellIndices[7] = aux.cellIndices[6] + zs;
  aux.density = aux.interpolate(grid->density);
}

__global__ void k_uniform_render(UniformGrid *grid, cudaSurfaceObject_t dest,
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

      sample += params.render_precise
                    ? raymarch<UniformGrid, samplePrecise>(grid, rayOrigin,
                                                           rayDir, sunDir)
                    : raymarch<UniformGrid, sampleCoarse>(grid, rayOrigin,
                                                          rayDir, sunDir);
    }
  sample *= step * step;

  surf2Dwrite(sample, dest, sizeof(float4) * x, y);
}
