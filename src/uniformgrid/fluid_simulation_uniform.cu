#include "fluid_simulation_uniform.h"
#include "helper_cuda.h"
#include "utils/grid_math.cuh"
#include <cuda_runtime.h>

FluidSimulationUniform::FluidSimulationUniform(const int3 &size)
    : FluidSimulation(size) {
  size_t minDim = size.x < size.y && size.x < size.z ? size.x
                  : size.y < size.z                  ? size.y
                                                     : size.z;
  size_t cellSize = 2;
  mipmapLevels = 1;
  while (size.x % cellSize == 0 && size.y % cellSize == 0 &&
         size.z % cellSize == 0 && cellSize * 4 <= minDim) {
    mipmapLevels++;
    cellSize *= 2;
  }

  gridSize = dim3(iDivUp(size.x, blockSize.x), iDivUp(size.y, blockSize.y),
                  iDivUp(size.z, blockSize.z));
  blockSizeLevel = (dim3 *)malloc(mipmapLevels * sizeof(dim3));
  gridSizeLevel = (dim3 *)malloc(mipmapLevels * sizeof(dim3));
  dim3 mapSize = dim3(size.x, size.y, size.z);
  for (int level = 0; level < mipmapLevels; level++) {
    blockSizeLevel[level] = min(blockSize, mapSize);
    gridSizeLevel[level] = dim3(iDivUp(mapSize.x, blockSizeLevel[level].x),
                                iDivUp(mapSize.y, blockSizeLevel[level].y),
                                iDivUp(mapSize.z, blockSizeLevel[level].z));
    mapSize.x /= 2;
    mapSize.y /= 2;
    mapSize.z /= 2;
  }

  h_grid = new UniformGrid;

  cudaMalloc(&h_grid->density, numCells * sizeof(float));
  cudaMalloc(&h_grid->velocity, numCells * sizeof(float3));
  cudaMalloc(&h_grid->pressure,
             mipmapCells(size.x, size.y, size.z) * sizeof(float));
  cudaMalloc(&h_grid->fluidity,
             mipmapCells(size.x, size.y, size.z) * sizeof(float));

  cudaMalloc(&h_grid->temporary, numCells * max(sizeof(float3), sizeof(float)));

  cudaMalloc(&d_grid, sizeof(UniformGrid));
  cudaMemcpy(d_grid, h_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice);
  getLastCudaError("Init grid");

  k_uniform_init_aliasses<<<1, 1>>>(d_grid);
  cudaMemcpy(h_grid, d_grid, sizeof(UniformGrid), cudaMemcpyDeviceToHost);

  const size_t numBins = numCells / 256;
  h_stats = (float *)malloc(numBins * sizeof(float));
  cudaMalloc(&d_stats, numBins * sizeof(float));

  reset();
}

FluidSimulationUniform::~FluidSimulationUniform() {
  cudaFree(h_grid->density);
  cudaFree(h_grid->velocity);
  cudaFree(h_grid->pressure);
  cudaFree(h_grid->fluidity);
  cudaFree(h_grid->temporary);
  cudaFree(d_grid);

  free(gridSizeLevel);
  free(blockSizeLevel);

  free(h_stats);
  cudaFree(d_stats);

  delete h_grid;
}

void FluidSimulationUniform::init() {
  adaptTopology();
  k_uniform_init<<<gridSize, blockSize>>>(d_grid);
}

void FluidSimulationUniform::reset() {
  const size_t numMipmapCells = mipmapCells(size.x, size.y, size.z);
  cudaMemset(h_grid->density, 0, numCells * sizeof(float));
  cudaMemset(h_grid->velocity, 0, numCells * sizeof(float3));
  cudaMemset(h_grid->pressure, 0, numMipmapCells * sizeof(float));
  cudaMemset(h_grid->fluidity, 0, numMipmapCells * sizeof(float));
  init();
}

void FluidSimulationUniform::advectVelocity() {
  k_uniform_advect_velocity<<<gridSize, blockSize>>>(d_grid);
  cudaMemcpy(h_grid->velocity, h_grid->t_velocity, numCells * sizeof(float3),
             cudaMemcpyDeviceToDevice);
}

void FluidSimulationUniform::project() {
  k_uniform_calc_divergence<<<gridSize, blockSize>>>(d_grid);

  for (int level = 1; level < mipmapLevels; level++)
    k_uniform_restrict<<<gridSizeLevel[level], blockSizeLevel[level]>>>(d_grid,
                                                                        level);

  for (int i = 0; i < 2; i++) {
    k_uniform_jacobi<<<gridSizeLevel[mipmapLevels - 1],
                       blockSizeLevel[mipmapLevels - 1]>>>(d_grid,
                                                           mipmapLevels - 1);
    k_uniform_jacobi_inv<<<gridSizeLevel[mipmapLevels - 1],
                           blockSizeLevel[mipmapLevels - 1]>>>(
        d_grid, mipmapLevels - 1);
  }

  for (int level = mipmapLevels - 2; level >= 0; level--) {
    k_uniform_prolongate<<<gridSizeLevel[level], blockSizeLevel[level]>>>(
        d_grid, level);
    for (int i = 0; i < 1; i++) {
      k_uniform_jacobi<<<gridSizeLevel[level], blockSizeLevel[level]>>>(d_grid,
                                                                        level);
      k_uniform_jacobi_inv<<<gridSizeLevel[level], blockSizeLevel[level]>>>(
          d_grid, level);
    }
  }

  k_uniform_apply_pressure<<<gridSize, blockSize>>>(d_grid);
}

void FluidSimulationUniform::projectLocal() {
  k_uniform_calc_divergence<<<gridSize, blockSize>>>(d_grid);

  for (int i = 0; i < 5; i++) {
    k_uniform_jacobi<<<gridSizeLevel[0], blockSizeLevel[0]>>>(d_grid, 0);
    k_uniform_jacobi_inv<<<gridSizeLevel[0], blockSizeLevel[0]>>>(d_grid, 0);
  }

  k_uniform_apply_pressure<<<gridSize, blockSize>>>(d_grid);
}

void FluidSimulationUniform::advectDensity() {
  k_uniform_advect_density<<<gridSize, blockSize>>>(d_grid);
  cudaMemcpy(h_grid->density, h_grid->t_density, numCells * sizeof(float),
             cudaMemcpyDeviceToDevice);
}

void FluidSimulationUniform::adaptTopology() {
  for (int level = 0; level < mipmapLevels; level++)
    k_uniform_set_solidity_ratio<<<gridSizeLevel[level],
                                   blockSizeLevel[level]>>>(d_grid, level);
}

void FluidSimulationUniform::render(cudaSurfaceObject_t dest,
                                    const int2 &resolution,
                                    const float3 &eyePos, const float3 &eyeDir,
                                    const float &fov, const float3 &sunDir) {
  const dim3 renderGridSize = dim3(iDivUp(resolution.x, renderBlockSize.x),
                                   iDivUp(resolution.y, renderBlockSize.y));

  k_uniform_render<<<renderGridSize, renderBlockSize>>>(
      d_grid, dest, resolution, eyePos, eyeDir, fov, sunDir);
}

void FluidSimulationUniform::debugStats() {
  const size_t numBins = numCells / 256;
  const size_t debugBlockSize = 256;
  const size_t debugGridSize = iDivUp(numBins, debugBlockSize);

  cudaMemset(d_stats, 0, numBins * sizeof(float));
  k_uniform_debug_stats<<<debugGridSize, debugBlockSize>>>(d_grid, d_stats,
                                                           numBins);

  cudaMemcpy(h_stats, d_stats, numBins * sizeof(float), cudaMemcpyDeviceToHost);

  float sum = 0.f;
  for (size_t i = 0; i < numBins; i++)
    sum += h_stats[i];

  printf("Smoke: %g\n", sum);
}