#pragma once
#include <cuda_runtime.h>

class FluidSimulation {
public:
  FluidSimulation(const int3 &size);
  virtual ~FluidSimulation();

  virtual void init() = 0;
  virtual void reset() = 0;
  virtual void adaptTopology() = 0;

  virtual void advectVelocity() = 0;
  virtual void project() = 0;
  virtual void projectLocal() = 0;
  virtual void advectDensity() = 0;

  virtual void render(cudaSurfaceObject_t dest, const int2 &resolution,
                      const float3 &eyePos, const float3 &eyeDir,
                      const float &fov, const float3 &sunDir) = 0;

  virtual void debugStats() = 0;

protected:
  int3 size;
  size_t numCells;
};