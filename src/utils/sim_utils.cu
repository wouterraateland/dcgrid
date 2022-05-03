#include "helper_cuda.h"
#include "helper_math.h"
#include "sdf.cuh"
#include "sim_utils.h"

__constant__ SimParams params;
void copySimParamsToDevice(SimParams &h_params) {
  cudaMemcpyToSymbol(params, &h_params, sizeof(SimParams));
}

__device__ float getCellFluidity(const int &x, const int &y, const int &z,
                                 const int &scale) {
  static const float sqrt3 = 1.73205f; // = sqrt(3)

  if (!params.enable_additional_solids)
    return 1.f;

  const float d = sceneSDF((make_float3(x, y, z) + .5f) * scale);
  const float sceneOverlap = clamp(.5f - d / (scale * sqrt3), 0.f, 1.f);

  return 1.f - sceneOverlap;
}

__device__ float3 velocityBndCond(const float3 &velocity, const int &x,
                                  const int &y, const int &z,
                                  const int &scale) {
  // Bottom center: inlet (v = (0, v_i, 0))
  if (y < 0 && length(make_float2(x * scale - .5f * params.gx,
                                  z * scale - .5f * params.gz)) <
                   params.emission_radius * params.rdx)
    return make_float3(0.f, params.velocity_emission_rate, 0.f);

  // Other boundaries: no-slip (v = 0)
  if (x < 0 || y < 0 || z < 0 || x * scale >= params.gx ||
      y * scale >= params.gy || z * scale >= params.gz)
    return make_float3(0.f);

  return velocity;
}

__device__ float densityBndCond(const float &density, const int &x,
                                const int &y, const int &z, const int &scale) {
  // Bottom center: inlet (v = (0, v_i, 0))
  if (y < 0 && length(make_float2(x * scale - .5f * params.gx,
                                  z * scale - .5f * params.gz)) <
                   params.emission_radius * params.rdx)
    return params.density_emission_rate;

  // Other boundaries: no-slip (v = 0)
  if (x < 0 || y < 0 || z < 0 || x * scale >= params.gx ||
      y * scale >= params.gy || z * scale >= params.gz)
    return 0.f;

  return density;
}
