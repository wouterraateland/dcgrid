#pragma once
#include <cuda_runtime.h>

enum class RenderChannel {
  Fluid,

  Density,
  Velocity,
  Fluidity,

  Resolution,
};

struct SimParams {
  // Timestep
  float dt;

  // Grid parameters
  int gx;
  int gy;
  int gz;
  float dx;
  float rdx;

  float velocity_emission_rate;
  float density_emission_rate;
  float emission_radius;

  // Controls
  bool enable_additional_solids;

  // Rendering
  bool render_solids;
  bool render_shadows;
  bool render_precise;
  RenderChannel render_channel;
  float aa_samples;

  // Colors
  float ambient;
  float3 background_color;
  float3 floor_color;
  float3 smoke_color;
  float3 scene_color;

  static SimParams defaultParams();
};