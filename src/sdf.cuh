#pragma once
#include "utils/sim_utils.h"
#include <cuda_runtime.h>
#include <helper_math.h>

#define EPSILON 1e-4f

inline __device__ float sphereSDF(const float3 &pos) {
  const float cx = params.gx * .5f;
  const float cy = params.gy * .45f;
  const float cz = params.gz * .5f;
  const float r = 2000.f * params.rdx;

  const float dx = pos.x - cx;
  const float dy = pos.y - cy;
  const float dz = pos.z - cz;
  return sqrtf(dx * dx + dy * dy + dz * dz) - r;
}

inline __device__ float sceneSDF(const float3 &pos) { return sphereSDF(pos); }

inline __device__ float3 estimateNormal(const float3 &p) {
  return normalize(
      make_float3(sceneSDF(make_float3(p.x + EPSILON, p.y, p.z)) -
                      sceneSDF(make_float3(p.x - EPSILON, p.y, p.z)),
                  sceneSDF(make_float3(p.x, p.y + EPSILON, p.z)) -
                      sceneSDF(make_float3(p.x, p.y - EPSILON, p.z)),
                  sceneSDF(make_float3(p.x, p.y, p.z + EPSILON)) -
                      sceneSDF(make_float3(p.x, p.y, p.z - EPSILON))));
}

#undef EPSILON