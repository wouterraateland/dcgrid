#pragma once
#include "dcgrid/dcgrid.h"
#include "fluid_simulation.h"

class FluidSimulationDCGrid : public FluidSimulation {
public:
  FluidSimulationDCGrid(const int3 &size, const size_t &maxNumBlocks);
  ~FluidSimulationDCGrid();

  void init();
  void reset();

  void advectVelocity();
  void project();
  void projectLocal();
  void advectDensity();

  void adaptTopology();
  void moveBlocks(size_t &numRefinedBlocks);
  void refineSubblocks(size_t &numRefinedBlocks);

  void accumulateVelocity();
  void accumulateDensity();
  void accumulateDivergence();

  void render(cudaSurfaceObject_t dest, const int2 &resolution,
              const float3 &eyePos, const float3 &eyeDir, const float &fov,
              const float3 &sunDir);

  void debugStats();

private:
  size_t *h_fullBlocksLevel = nullptr;
  size_t *h_maxNumBlocksLevel = nullptr;
  size_t *h_blockLoads = nullptr;
  size_t *h_levelOffsets = nullptr;

  float *h_stats = nullptr;
  float *d_stats = nullptr;

  // Adaptation
  float *h_blockScores = nullptr;
  float *d_blockScores = nullptr;
  float *h_subblockScores = nullptr;
  float *d_subblockScores = nullptr;

  size_t *moveLimit = nullptr;
  size_t *h_blockIndicesToMove = nullptr;
  size_t *d_blockIndicesToMove = nullptr;
  size_t *h_destSubblockIndices = nullptr;
  size_t *d_destSubblockIndices = nullptr;
  size_t *d_touchedBlockIndices = nullptr;

  size_t hashTableSize = 0;
  size_t blockSize = 256;
  size_t gridSize = 0;
  dim3 renderBlockSize = dim3(16, 16);

  DCGrid *h_grid = nullptr;
  DCGrid *d_grid = nullptr;
};
