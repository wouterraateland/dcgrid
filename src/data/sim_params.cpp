#include "sim_params.h"
#include <helper_math.h>

SimParams SimParams::defaultParams() {
  SimParams params;
  params.gx = 128;
  params.gy = 128;
  params.gz = 128;
  params.dx = 10000.f / 128;
  params.dt = 3.f;

  params.velocity_emission_rate = 150.f;
  params.density_emission_rate = 0.002f;
  params.emission_radius = 750.f;

  // Debug controls
  params.enable_additional_solids = false;

  // Rendering
  params.render_solids = true;
  params.render_shadows = true;
  params.render_precise = true;
  params.render_channel = RenderChannel::Resolution;
  params.aa_samples = 1.f;

  // Colors
  params.ambient = .3f;
  params.background_color = make_float3(0.f, 0.f, 0.f);
  params.floor_color = make_float3(178.f, 158.f, 135.f) / 255.f;
  params.smoke_color = make_float3(.9f, .9f, .9f);
  params.scene_color = make_float3(107.f, 163.f, 204.f) / 255.f;

  return params;
}
