#pragma once
#include "data/sim_params.h"
#include <cuda_runtime.h>

extern __constant__ SimParams params;
void copySimParamsToDevice(SimParams &h_params);

__device__ float getCellFluidity(const int &x, const int &y, const int &z,
                                 const int &scale = 1);

__device__ float3 velocityBndCond(const float3 &velocity, const int &x,
                                  const int &y, const int &z,
                                  const int &scale = 1);

__device__ float densityBndCond(const float &density, const int &x,
                                const int &y, const int &z,
                                const int &scale = 1);
