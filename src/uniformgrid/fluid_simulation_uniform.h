#pragma once
#include "fluid_simulation.h"
#include "uniformgrid.h"

class FluidSimulationUniform : public FluidSimulation {
public:
  FluidSimulationUniform(const int3 &size);
  ~FluidSimulationUniform();

  void init();
  void reset();
  void adaptTopology();

  void advectVelocity();
  void project();
  void projectLocal();
  void advectDensity();

  void render(cudaSurfaceObject_t dest, const int2 &resolution,
              const float3 &eyePos, const float3 &eyeDir, const float &fov,
              const float3 &sunDir);

  void debugStats();

private:
  size_t mipmapLevels;

  dim3 blockSize = dim3(8, 8, 4);
  dim3 renderBlockSize = dim3(16, 16);
  dim3 *blockSizeLevel;

  dim3 gridSize;
  dim3 *gridSizeLevel;

  float *h_stats;
  float *d_stats;

  UniformGrid *h_grid;
  UniformGrid *d_grid;
};