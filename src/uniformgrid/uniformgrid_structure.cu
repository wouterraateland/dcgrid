#include "uniformgrid.h"
#include "utils/grid_math.cuh"
#include "utils/sim_utils.h"
#include <helper_cuda.h>
#include <helper_math.h>

__global__ void k_uniform_init_aliasses(UniformGrid *grid) {
  grid->t_velocity = (float3 *)grid->temporary;
  grid->t_pressure = (float *)grid->temporary;
  grid->divergence =
      ((float *)grid->temporary) + mipmapCells(params.gx, params.gy, params.gz);
  grid->t_density = (float *)grid->temporary;
}

__global__ void k_uniform_init(UniformGrid *grid) {
  INIT_XYZ
  INIT_IDX

  grid->density[idx] = 0.f;
  grid->velocity[idx] = make_float3(0.f);
}

__global__ void k_uniform_set_solidity_ratio(UniformGrid *grid,
                                             int mipmapLevel) {
  INIT_XYZ_MIPMAP
  const int scale = 1 << mipmapLevel;

  const size_t idx = mipmapIdx(x, y, z, params.gx, params.gy, params.gz, scale);

  grid->fluidity[idx] = getCellFluidity(x, y, z, scale);
}

__global__ void k_uniform_debug_stats(UniformGrid *grid, float *stats,
                                      size_t numBins) {
  const size_t binIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (binIndex >= numBins)
    return;
  const size_t cellsPerBin = 256;

  size_t idx = binIndex * cellsPerBin;
  for (int i = 0; i < cellsPerBin; i++, idx++)
    stats[binIndex] += grid->density[idx] * grid->fluidity[idx];
}