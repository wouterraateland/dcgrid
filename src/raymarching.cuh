#pragma once
#include "sdf.cuh"
#include "utils/colors.cuh"
#include "utils/grid_math.cuh"
#include "utils/interpolation.cuh"
#include "utils/sim_utils.h"
#include <cuda_runtime.h>
#include <float.h>
#include <helper_math.h>
#include <stdint.h>

#define Z_NEAR 10.f
#define STEP_MULTIPLIER 1.00f
#define SUNMARCH_SAMPLES 100

enum class Renderable { None, Scene, Floor, Debug };

struct MarchAux {
  int3 ipos = make_int3(0);
  float3 fraction = make_float3(0.f);
  size_t cellIndices[8];
  int scale = 1;

  float density;

  template <typename T> inline __device__ T interpolate(const T *values) {
    const float &dx = fraction.x;
    const float dxInv = 1.f - dx;
    const T c00 = values[cellIndices[0]] * dxInv + values[cellIndices[4]] * dx;
    const T c01 = values[cellIndices[1]] * dxInv + values[cellIndices[5]] * dx;
    const T c10 = values[cellIndices[2]] * dxInv + values[cellIndices[6]] * dx;
    const T c11 = values[cellIndices[3]] * dxInv + values[cellIndices[7]] * dx;

    const float &dy = fraction.y;
    const float dyInv = 1.f - dy;
    const T c0 = c00 * dyInv + c10 * dy;
    const T c1 = c01 * dyInv + c11 * dy;

    return c0 * (1.f - fraction.z) + c1 * fraction.z;
  }
};

inline __device__ float densityToAlpha(const float &distance,
                                       const float &density) {
  return 1.f - __expf(-distance * params.dx * density);
}

// Returns (dstToBox, dstInsideBox). If ray misses box, dstInsideBox will be
// zero
inline __device__ bool rayDomainIntersections(float &dNear, float &dFar,
                                              const float3 &rayOrigin,
                                              const float3 &rayDirInv) {
  const float cx = params.gx * .5f;
  const float cy = params.gy * .5f;
  const float cz = params.gz * .5f;

  const float nx = rayDirInv.x * (rayOrigin.x - cx);
  const float ny = rayDirInv.y * (rayOrigin.y - cy);
  const float nz = rayDirInv.z * (rayOrigin.z - cz);

  const float kx = fabsf(rayDirInv.x) * cx;
  const float ky = fabsf(rayDirInv.y) * cy;
  const float kz = fabsf(rayDirInv.z) * cz;

  const float t1x = -nx - kx;
  const float t1y = -ny - ky;
  const float t1z = -nz - kz;
  const float t2x = -nx + kx;
  const float t2y = -ny + ky;
  const float t2z = -nz + kz;

  dNear = fmaxf(fmaxf(t1x, t1y), t1z);
  dFar = fminf(fminf(t2x, t2y), t2z);

  return dNear < dFar && dFar > 0.f;
}

template <typename T, void (*sample)(T *data, const float3 &pos, MarchAux &aux)>
__device__ float lightmarch(T *data, const float3 &origin, MarchAux &aux,
                            const float3 &sunDir, const float3 &sunDirInv) {
  float lightTransport = 1.f;

  if (params.render_solids && params.enable_additional_solids) {
    float sdfScene = FLT_MAX;
    float dScene = 0.f;
    for (int i = 0; i < 100; i++) {
      sdfScene = sceneSDF(origin + sunDir * dScene);
      lightTransport = fminf(lightTransport, (sdfScene / dScene) / .06f);

      if (lightTransport <= 1e-2f)
        return 0.f;

      dScene += sdfScene;
    }
  }

  if (params.render_channel == RenderChannel::Fluid) {
    float dNear, dFar;
    if (rayDomainIntersections(dNear, dFar, origin, sunDirInv)) {
      float3 pos = origin;
      if (dNear > 0.f) {
        pos += sunDir * dNear;
        dFar -= dNear;
      }
      const float stepSize = fmaxf(dFar / SUNMARCH_SAMPLES, 1e-1f);
      float density = 0.f;

      for (float dCurrent = stepSize; dCurrent < dFar && density < 5.f;
           dCurrent += stepSize) {
        pos += sunDir * stepSize;

        sample(data, pos, aux);
        density += aux.density * params.dx * stepSize;
      }
      lightTransport *= __expf(-density);
    }
  }

  return lightTransport;
}

template <typename T, void (*sample)(T *data, const float3 &pos, MarchAux &aux)>
__device__ float4 raymarch(T *data, const float3 &origin, const float3 &dir,
                           const float3 &sunDir) {
  const float ambientInv = 1.f - params.ambient;
  const float floorIllumination = fmaxf(0.f, sunDir.y);
  const float sliceZ = params.gz * .5f;

  const float3 dirInv = 1.f / dir;
  const float3 sunDirInv = 1.f / sunDir;

  float dCurrent = 0.f;
  float3 pos = origin;

  float dSolid;
  float illumination;
  float3 color = make_float3(0.f);
  float alpha = 0.f;

  MarchAux aux;

  // Start ray marching
  // while (alpha < .99f) {
  dSolid = FLT_MAX;
  Renderable renderable = Renderable::None;

  // Debug
  if (params.render_channel != RenderChannel::Fluid &&
      (sliceZ - pos.z) * dir.z > 0.f) {
    const float dDebug = (sliceZ - origin.z) / dir.z;
    const float dx = origin.x + dDebug * dir.x;
    const float dy = origin.y + dDebug * dir.y;
    if (dx >= 0.f && dy >= 0.f && dx < params.gx && dy < params.gy &&
        dDebug < dSolid) {
      dSolid = dDebug;
      renderable = Renderable::Debug;
    }
  }

  // Floor
  if (params.render_solids && (0.f - pos.y) * dir.y > 0.f) {
    const float dFloor = (0.f - origin.y) / dir.y;
    if (dFloor < dSolid) {
      dSolid = dFloor;
      renderable = Renderable::Floor;
    }
  }

  // Scene
  if (params.render_solids && params.enable_additional_solids) {
    float sdfScene = FLT_MAX;
    float dScene = dCurrent;
    for (int i = 0; i < 100; i++) {
      sdfScene = sceneSDF(origin + dir * dScene);

      if (sdfScene < 1e-1f || dScene >= dSolid)
        break;

      dScene += sdfScene;
    }

    if (sdfScene < 1e-1f && dScene < dSolid) {
      dSolid = dScene;
      renderable = Renderable::Scene;
    }
  }

  if (params.render_channel == RenderChannel::Fluid) {
    float dNear, dFar;
    if (rayDomainIntersections(dNear, dFar, origin, dirInv)) {
      dCurrent = fmaxf(dCurrent, fmaxf(dNear, Z_NEAR));
      float3 pos = origin + dir * dCurrent;
      float3 prevPos = pos;
      const float dMax = fminf(dFar, dSolid);
      while (dCurrent < dMax && alpha <= .99f) {
        prevPos = pos;
        dCurrent =
            fminf(fmaxf(dCurrent * STEP_MULTIPLIER, dCurrent + .5f), dMax);
        pos = origin + dir * dCurrent;

        sample(data, (prevPos + pos) * .5f, aux);

        if (aux.density > 1e-6f) {
          const float sampleAlpha =
              densityToAlpha(aux.density, length(pos - prevPos));
          illumination = 1.f;
          if (params.render_shadows)
            illumination *= lightmarch<T, sample>(data, (prevPos + pos) * .5f,
                                                  aux, sunDir, sunDirInv);
          illumination = params.ambient + illumination * ambientInv;
          mix(illumination * params.smoke_color, sampleAlpha, color, alpha);
        }
      }
    }
  }

  dCurrent = dSolid;
  pos = origin + dir * dSolid;

  const float backgroundAlpha = densityToAlpha(1e-6f, dSolid);
  // mix(params.background_color, backgroundAlpha, color, alpha);

  if (renderable == Renderable::Scene) {
    illumination = fmaxf(0.f, dot(estimateNormal(pos), sunDir));
    if (params.render_shadows)
      illumination *= lightmarch<T, sample>(data, pos, aux, sunDir, sunDirInv);
    illumination = params.ambient + illumination * ambientInv;
    mix(illumination * params.scene_color, 1.f - backgroundAlpha, color, alpha);
  }

  if (renderable == Renderable::Floor) {
    pos.y = 0.f;
    illumination = floorIllumination;
    if (params.render_shadows)
      illumination *= lightmarch<T, sample>(data, pos, aux, sunDir, sunDirInv);
    illumination = params.ambient + illumination * ambientInv;

    // Checkerboard pattern
    illumination *= 1.f + .1f * fmodf(fabsf(floorf(pos.x / params.gx) +
                                            floorf(pos.z / params.gx)),
                                      2.f);
    mix(illumination * params.floor_color, 1.f - backgroundAlpha, color, alpha);
  }

  if (renderable == Renderable::Debug) {
    pos.z = sliceZ;
    sample(data, pos, aux);

    if (params.render_channel == RenderChannel::Density) {
      const float v = aux.density * 1e2f;
      mix(make_float3(v, 0.f, 0.f), 1.f, color, alpha);
    }

    if (params.render_channel == RenderChannel::Velocity) {
      const float3 velocity = aux.interpolate(data->velocity);
      const float v = length(velocity) * 1e-1f;
      mix(make_float3(v - 2.f, v - 1.f, v), 1.f, color, alpha);
    }

    if (params.render_channel == RenderChannel::Resolution) {
      const int c = (aux.ipos.x / 4 + aux.ipos.y / 4) % 2;
      mix(make_float3(c), 1.f, color, alpha);
    }

    if (params.render_channel == RenderChannel::Fluidity) {
      const float f = aux.interpolate(data->fluidity);
      mix(make_float3(f), 1.f, color, alpha);
    }
  }
  // }

  // mix(params.background_color, 1.f, color, alpha);

  return make_float4(color, alpha);
}

#undef Z_NEAR
#undef STEP_MULTIPLIER
#undef SUNMARCH_SAMPLES
