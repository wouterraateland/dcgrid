#include "fluid_simulation_dcgrid.h"
#include "helper_cuda.h"
#include "utils/grid_math.cuh"
#include "utils/timer.h"
#include <algorithm>
#include <float.h>
#include <numeric>

FluidSimulationDCGrid::FluidSimulationDCGrid(const int3 &size,
                                             const size_t &maxNumBlocks)
    : FluidSimulation(size) {
  int minDim = size.x < size.y && size.x < size.z ? size.x
               : size.y < size.z                  ? size.y
                                                  : size.z;

  int cellSize = 2;
  int levels = 1;
  while (size.x % cellSize == 0 && size.y % cellSize == 0 &&
         size.z % cellSize == 0 && cellSize * DCGrid::blockW <= minDim) {
    levels++;
    cellSize *= 2;
  }

  h_fullBlocksLevel = (size_t *)malloc(levels * sizeof(size_t));
  h_maxNumBlocksLevel = (size_t *)malloc(levels * sizeof(size_t));
  h_blockLoads = (size_t *)malloc(levels * sizeof(size_t));
  h_levelOffsets = (size_t *)malloc(levels * sizeof(size_t));

  for (int level = 0, cellSize = 1; level < levels; level++, cellSize *= 2)
    h_fullBlocksLevel[level] = iDivUp(size.x, cellSize * DCGrid::blockW) *
                               iDivUp(size.y, cellSize * DCGrid::blockW) *
                               iDivUp(size.z, cellSize * DCGrid::blockW);

  // Lowest resolution block should be completely filled
  if (maxNumBlocks < h_fullBlocksLevel[levels - 1]) {
    printf("Error: Too few blocks to fill lowest resolution (%zu / %zu).\n",
           maxNumBlocks, h_fullBlocksLevel[levels - 1]);
    exit(1);
  }

  h_maxNumBlocksLevel[levels - 1] = h_fullBlocksLevel[levels - 1];
  size_t blocksLeft = maxNumBlocks - h_maxNumBlocksLevel[levels - 1];

  for (int level = levels - 2; level >= 0; level--) {
    h_maxNumBlocksLevel[level] =
        min(blocksLeft / (level + 1), h_fullBlocksLevel[level]);
    blocksLeft -= h_maxNumBlocksLevel[level];
  }

  if (h_maxNumBlocksLevel[0] == 0) {
    printf("Error: Too few blocks to reach highest resolution.\n");
    exit(1);
  }

  h_levelOffsets[0] = 0;
  for (int level = 1; level < levels; level++)
    h_levelOffsets[level] =
        h_levelOffsets[level - 1] + h_maxNumBlocksLevel[level - 1];

  gridSize = iDivUp(maxNumBlocks, blockSize);

  h_grid = new DCGrid;
  h_grid->maxNumBlocks = maxNumBlocks;
  hashTableSize = 4 * maxNumBlocks;
  h_grid->levels = levels;
  h_grid->sparseLevels = 0;
  for (int level = 0; level < levels; level++)
    if (h_maxNumBlocksLevel[level] < h_fullBlocksLevel[level])
      h_grid->sparseLevels = level + 1;

  numCells = maxNumBlocks * DCGrid::blockV;
  const size_t numApronCells = maxNumBlocks * DCGrid::apronV;
  const size_t numSubblocks = 8 * maxNumBlocks;
  const size_t tempSize = max(sizeof(float), sizeof(float3));

  cudaMalloc(&h_grid->density, numCells * sizeof(float));
  cudaMalloc(&h_grid->velocity, numCells * sizeof(float3));
  cudaMalloc(&h_grid->fluidity, numCells * sizeof(float));
  cudaMalloc(&h_grid->temporary, numCells * tempSize);

  cudaMalloc(&h_grid->hashTableSize, levels * sizeof(size_t));
  cudaMalloc(&h_grid->hashTableOffset, levels * sizeof(size_t));
  cudaMalloc(&h_grid->hashKey, hashTableSize * sizeof(uint32_t));
  cudaMalloc(&h_grid->hashVal, hashTableSize * sizeof(size_t));
  cudaMalloc(&h_grid->blockPositions, maxNumBlocks * sizeof(int3));
  cudaMalloc(&h_grid->blockLevels, maxNumBlocks * sizeof(uint8_t));
  cudaMalloc(&h_grid->blockFlags, maxNumBlocks * sizeof(uint8_t));
  cudaMalloc(&h_grid->parentIndices, maxNumBlocks * sizeof(size_t));
  cudaMalloc(&h_grid->cellIndices, numApronCells * sizeof(size_t));
  cudaMalloc(&h_grid->childIndices, numSubblocks * sizeof(size_t));
  cudaMalloc(&h_grid->maxNumBlocksLevel, levels * sizeof(size_t));
  cudaMalloc(&h_grid->fullBlocksLevel, levels * sizeof(size_t));
  cudaMalloc(&h_grid->levelOffsets, levels * sizeof(size_t));
  cudaMalloc(&h_grid->freeBlockIndices, maxNumBlocks * sizeof(size_t));
  cudaMalloc(&h_grid->blockLoads, levels * sizeof(size_t));
  cudaMalloc(&d_grid, sizeof(DCGrid));
  getLastCudaError("Alloc grid data");

  size_t *h_hashTableSize = (size_t *)malloc(levels * sizeof(size_t));
  size_t *h_hashTableOffset = (size_t *)malloc(levels * sizeof(size_t));
  for (int level = 0; level < levels; level++)
    h_hashTableSize[level] = 4 * h_maxNumBlocksLevel[level];
  h_hashTableOffset[0] = 0;
  for (int level = 1; level < levels; level++)
    h_hashTableOffset[level] =
        h_hashTableOffset[level - 1] + h_hashTableSize[level - 1];
  cudaMemcpy(h_grid->hashTableSize, h_hashTableSize, levels * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(h_grid->hashTableOffset, h_hashTableOffset,
             levels * sizeof(size_t), cudaMemcpyHostToDevice);
  free(h_hashTableSize);
  free(h_hashTableOffset);

  cudaMemcpy(h_grid->maxNumBlocksLevel, h_maxNumBlocksLevel,
             levels * sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(h_grid->fullBlocksLevel, h_fullBlocksLevel,
             levels * sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(h_grid->levelOffsets, h_levelOffsets, levels * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_grid, h_grid, sizeof(DCGrid), cudaMemcpyHostToDevice);
  getLastCudaError("Copy grid data");

  h_blockScores = (float *)malloc(maxNumBlocks * sizeof(float));
  h_subblockScores = (float *)malloc(maxNumBlocks * 8 * sizeof(float));
  h_stats = (float *)malloc(maxNumBlocks * sizeof(float));

  cudaMalloc(&d_stats, maxNumBlocks * sizeof(float));

  cudaMalloc(&d_blockScores, maxNumBlocks * sizeof(float));
  cudaMalloc(&d_subblockScores, maxNumBlocks * 8 * sizeof(float));

  moveLimit = (size_t *)malloc(levels * sizeof(size_t));
  h_blockIndicesToMove = (size_t *)malloc(maxNumBlocks * sizeof(size_t));
  h_destSubblockIndices = (size_t *)malloc(maxNumBlocks * 8 * sizeof(size_t));
  cudaMalloc(&d_blockIndicesToMove, maxNumBlocks * sizeof(size_t));
  cudaMalloc(&d_destSubblockIndices, maxNumBlocks * 8 * sizeof(size_t));
  cudaMalloc(&d_touchedBlockIndices, maxNumBlocks * sizeof(size_t));

  reset();
}

FluidSimulationDCGrid::~FluidSimulationDCGrid() {
  reset();

  cudaFree(h_grid->maxNumBlocksLevel);
  cudaFree(h_grid->blockLoads);

  cudaFree(d_blockScores);
  cudaFree(d_subblockScores);
  cudaFree(d_stats);

  free(h_subblockScores);
  free(h_blockScores);
  free(h_stats);

  cudaFree(h_grid->hashKey);
  cudaFree(h_grid->hashVal);

  cudaFree(h_grid->blockPositions);
  cudaFree(h_grid->blockLevels);
  cudaFree(h_grid->blockFlags);
  cudaFree(h_grid->childIndices);
  cudaFree(h_grid->cellIndices);
  cudaFree(h_grid->freeBlockIndices);
  cudaFree(h_grid->parentIndices);

  cudaFree(h_grid->density);
  cudaFree(h_grid->velocity);
  cudaFree(h_grid->fluidity);
  cudaFree(h_grid->temporary);
  cudaFree(d_grid);

  free(h_fullBlocksLevel);
  free(h_maxNumBlocksLevel);
  free(h_blockLoads);
  free(h_levelOffsets);

  free(moveLimit);
  free(h_blockIndicesToMove);
  free(h_destSubblockIndices);
  cudaFree(d_blockIndicesToMove);
  cudaFree(d_destSubblockIndices);
  cudaFree(d_touchedBlockIndices);

  delete h_grid;

  getLastCudaErrorSync("Destruct DCGrid");
}

void FluidSimulationDCGrid::init() {
  k_dcgrid_init_apron_indices<<<gridSize, blockSize>>>(d_grid);
  getLastCudaErrorSync("Init apron indices");

  // Allocate fully allocatable levels in ordered fashion
  for (int level = h_grid->sparseLevels; level < h_grid->levels; level++) {
    const int extent = DCGrid::blockW << level;
    const size_t rx = iDivUp(size.x, extent);
    const size_t ry = iDivUp(size.y, extent);
    const size_t rz = iDivUp(size.z, extent);

    const dim3 levelBlockSize(4, 4, 4);
    const dim3 levelGridSize(iDivUp(rx, 4), iDivUp(ry, 4), iDivUp(rz, 4));
    k_dcgrid_activate_level<<<levelGridSize, levelBlockSize>>>(d_grid, level);
  }
  getLastCudaErrorSync("Activate levels");

  for (int i = 0; i < 5; i++) {
    adaptTopology();
  }
}

void FluidSimulationDCGrid::reset() {
  cudaMemset(h_grid->hashKey, 0xff, hashTableSize * sizeof(uint32_t));
  cudaMemset(h_grid->hashVal, 0xff, hashTableSize * sizeof(size_t));
  cudaMemset(h_grid->blockPositions, 0, h_grid->maxNumBlocks * sizeof(int3));
  cudaMemset(h_grid->blockLevels, 0xff, h_grid->maxNumBlocks * sizeof(uint8_t));
  cudaMemset(h_grid->blockFlags, 0, h_grid->maxNumBlocks * sizeof(uint8_t));
  cudaMemset(h_grid->parentIndices, 0xff,
             h_grid->maxNumBlocks * sizeof(size_t));
  cudaMemset(h_grid->childIndices, 0xff,
             8 * h_grid->maxNumBlocks * sizeof(size_t));
  cudaMemset(h_grid->cellIndices, 0,
             h_grid->maxNumBlocks * DCGrid::apronV * sizeof(size_t));

  cudaMemset(h_grid->density, 0, numCells * sizeof(float));
  cudaMemset(h_grid->velocity, 0, numCells * sizeof(float3));
  cudaMemset(h_grid->fluidity, 0, numCells * sizeof(float));
  cudaMemset(h_grid->temporary, 0,
             numCells * max(sizeof(float), sizeof(float3)));
  getLastCudaError("reset flags");

  for (int level = 0; level < h_grid->levels; level++) {
    h_blockLoads[level] =
        (h_maxNumBlocksLevel[level] == h_fullBlocksLevel[level])
            ? h_maxNumBlocksLevel[level]
            : 0;
    moveLimit[level] = 0;
  }

  cudaMemcpy(h_grid->blockLoads, h_blockLoads, h_grid->levels * sizeof(size_t),
             cudaMemcpyHostToDevice);

  size_t *h_freeBlockIndices =
      (size_t *)malloc(h_grid->maxNumBlocks * sizeof(size_t));
  for (size_t i = 0; i < h_grid->maxNumBlocks; i++)
    h_freeBlockIndices[i] = i;

  cudaMemcpy(h_grid->freeBlockIndices, h_freeBlockIndices,
             h_grid->maxNumBlocks * sizeof(size_t), cudaMemcpyHostToDevice);
  free(h_freeBlockIndices);
  getLastCudaError("memset blockloads");

  cudaMemcpy(d_grid, h_grid, sizeof(DCGrid), cudaMemcpyHostToDevice);
  getLastCudaError("memcpy data");

  k_dcgrid_init_aliasses<<<1, 1>>>(d_grid);
  cudaMemcpy(h_grid, d_grid, sizeof(DCGrid), cudaMemcpyDeviceToHost);
  getLastCudaErrorSync("Init aliasses");

  init();
}

void FluidSimulationDCGrid::advectVelocity() {
  k_dcgrid_advect_velocity<<<h_grid->maxNumBlocks, DCGrid::blockV>>>(d_grid);
  cudaMemcpy(h_grid->velocity, h_grid->t_velocity, numCells * sizeof(float3),
             cudaMemcpyDeviceToDevice);
  accumulateVelocity();
}

void FluidSimulationDCGrid::project() {
  k_dcgrid_calc_divergence<<<h_grid->maxNumBlocks, DCGrid::blockV>>>(d_grid);
  accumulateDivergence();

  for (int i = 0; i < 5; i++) {
    k_dcgrid_jacobi<<<h_maxNumBlocksLevel[h_grid->levels - 1],
                      DCGrid::blockV>>>(d_grid, h_grid->levels - 1);
    k_dcgrid_jacobi_inv<<<h_maxNumBlocksLevel[h_grid->levels - 1],
                          DCGrid::blockV>>>(d_grid, h_grid->levels - 1);
  }

  for (int level = h_grid->levels - 2; level >= 0; level--) {
    k_dcgrid_prolongate<<<h_maxNumBlocksLevel[level], DCGrid::blockV>>>(d_grid,
                                                                        level);
    for (int i = 0; i < 5; i++) {
      k_dcgrid_jacobi<<<h_maxNumBlocksLevel[level], DCGrid::blockV>>>(d_grid,
                                                                      level);
      k_dcgrid_jacobi_inv<<<h_maxNumBlocksLevel[level], DCGrid::blockV>>>(
          d_grid, level);
    }
  }

  k_dcgrid_apply_pressure<<<h_grid->maxNumBlocks, DCGrid::blockV>>>(d_grid);
  accumulateVelocity();
}

void FluidSimulationDCGrid::projectLocal() {
  k_dcgrid_calc_divergence<<<h_grid->maxNumBlocks, DCGrid::blockV>>>(d_grid);
  accumulateDivergence();

  for (int level = h_grid->levels - 1; level >= 0; level--) {
    for (int i = 0; i < 10; i++) {
      k_dcgrid_jacobi<<<h_maxNumBlocksLevel[level], DCGrid::blockV>>>(d_grid,
                                                                      level);
      k_dcgrid_jacobi_inv<<<h_maxNumBlocksLevel[level], DCGrid::blockV>>>(
          d_grid, level);
    }
  }

  k_dcgrid_apply_pressure<<<h_grid->maxNumBlocks, DCGrid::blockV>>>(d_grid);
  accumulateVelocity();
}

void FluidSimulationDCGrid::advectDensity() {
  k_dcgrid_advect_density<<<h_grid->maxNumBlocks, DCGrid::blockV>>>(d_grid);
  cudaMemcpy(h_grid->density, h_grid->t_density, numCells * sizeof(float),
             cudaMemcpyDeviceToDevice);
  accumulateDensity();
}

void FluidSimulationDCGrid::adaptTopology() {
  k_dcgrid_calc_vorticity<<<h_grid->maxNumBlocks, DCGrid::blockV>>>(d_grid);

  size_t numBlocksTouched = 0;
  moveBlocks(numBlocksTouched);

  if (numBlocksTouched > 0) {
    cudaMemset(h_grid->hashKey, 0xff, hashTableSize * sizeof(uint32_t));
    cudaMemset(h_grid->hashVal, 0xff, hashTableSize * sizeof(size_t));
    getLastCudaError("Reset hash table");
    k_dcgrid_refill_hash_table<<<gridSize, blockSize>>>(d_grid);
    getLastCudaErrorSync("Refill hash table");
  }

  refineSubblocks(numBlocksTouched);

  if (numBlocksTouched > 0) {
    k_dcgrid_refresh_apron_indices<<<h_grid->maxNumBlocks, DCGrid::apronV>>>(
        d_grid);
    getLastCudaErrorSync("Refresh apron indices");

    for (int level = h_grid->levels - 2; level >= 0; level--)
      k_dcgrid_propagate_values<<<numBlocksTouched, DCGrid::blockV>>>(
          d_grid, d_touchedBlockIndices, level);
    getLastCudaErrorSync("Propagate values");
  }
}

void FluidSimulationDCGrid::moveBlocks(size_t &numBlocksTouched) {
  // Smallest first, negative last
  auto blockOrder = [this](size_t i1, size_t i2) {
    return h_blockScores[i1] < 0.f ? false
                                   : h_blockScores[i1] < h_blockScores[i2];
  };

  // Largest first
  auto subblockOrder = [this](size_t i1, size_t i2) {
    return h_subblockScores[i1] > h_subblockScores[i2];
  };

  k_dcgrid_calc_subblock_scores<<<iDivUp(h_grid->maxNumBlocks * 8, 64), 64>>>(
      d_grid, d_subblockScores);
  k_dcgrid_accumulate_subblock_scores<<<iDivUp(h_grid->maxNumBlocks, 64), 64>>>(
      d_grid, d_blockScores, d_subblockScores);
  getLastCudaError("Calc scores");
  cudaMemcpy(h_subblockScores, d_subblockScores,
             h_grid->maxNumBlocks * 8 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_blockScores, d_blockScores, h_grid->maxNumBlocks * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemset(h_grid->blockFlags, 0, h_grid->maxNumBlocks * sizeof(uint8_t));
  cudaMemcpy(h_blockLoads, h_grid->blockLoads, h_grid->levels * sizeof(size_t),
             cudaMemcpyDeviceToHost);
  getLastCudaError("Copy scores");

  size_t blocksToMove = 0;
  for (int level = 0; level < h_grid->levels - 1; level++) {
    // Determine max blocks to move
    const size_t d0 = h_maxNumBlocksLevel[level];
    const size_t d1 = 8 * h_maxNumBlocksLevel[level + 1];
    size_t l = std::min({d0, d1, h_blockLoads[level],
                         h_fullBlocksLevel[level] - h_blockLoads[level]});
    if (moveLimit[level] > 0)
      l = std::min(l, moveLimit[level]);

    if (l == 0)
      continue;

    // Find l lowest scoring blocks
    size_t *moveCandidates = h_blockIndicesToMove + blocksToMove;
    std::iota(moveCandidates, moveCandidates + d0, h_levelOffsets[level]);
    if (d0 <= l)
      std::sort(moveCandidates, moveCandidates + d0, blockOrder);
    else {
      std::nth_element(moveCandidates, moveCandidates + l, moveCandidates + d0,
                       blockOrder);
      std::sort(moveCandidates, moveCandidates + l, blockOrder);
    }

    // Find l highest scoring parent subblocks
    size_t *destCandidates = h_destSubblockIndices + blocksToMove;
    std::iota(destCandidates, destCandidates + d1,
              8 * h_levelOffsets[level + 1]);
    if (d1 <= l)
      std::sort(destCandidates, destCandidates + d1, subblockOrder);
    else {
      std::nth_element(destCandidates, destCandidates + l, destCandidates + d1,
                       subblockOrder);
      std::sort(destCandidates, destCandidates + l, subblockOrder);
    }

    size_t matches = 0;
    while (matches < l && h_blockScores[moveCandidates[matches]] >= 0.f &&
           h_subblockScores[destCandidates[matches]] >= 0.f &&
           h_blockScores[moveCandidates[matches]] <
               h_subblockScores[destCandidates[matches]]) {
      matches++;
      // Prevent block containing subblock from being moved
      h_blockScores[destCandidates[matches] / 8] = -FLT_MAX;
    }
    // Adjust move limit, keep some slack
    moveLimit[level] = matches * 1.2f;
    blocksToMove += matches;
  }

  if (blocksToMove > 0) {
    cudaMemcpy(d_blockIndicesToMove, h_blockIndicesToMove,
               blocksToMove * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_destSubblockIndices, h_destSubblockIndices,
               blocksToMove * sizeof(size_t), cudaMemcpyHostToDevice);
    k_dcgrid_move_blocks<<<gridSize, blockSize>>>(
        d_grid, d_blockIndicesToMove, d_destSubblockIndices, blocksToMove);
    getLastCudaErrorSync("Move blocks");

    cudaMemcpy(d_touchedBlockIndices, h_blockIndicesToMove,
               blocksToMove * sizeof(size_t), cudaMemcpyHostToDevice);
    numBlocksTouched += blocksToMove;
  }
}

void FluidSimulationDCGrid::refineSubblocks(size_t &numBlocksTouched) {
  const size_t gridSize = iDivUp(h_grid->maxNumBlocks * 8, 64);
  k_dcgrid_calc_subblock_scores<<<gridSize, 64>>>(d_grid, d_subblockScores);
  cudaMemcpy(h_subblockScores, d_subblockScores,
             h_grid->maxNumBlocks * 8 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_blockLoads, h_grid->blockLoads, h_grid->levels * sizeof(size_t),
             cudaMemcpyDeviceToHost);
  getLastCudaError("Copy block flags and subblock scores");

  size_t numSubblocksToRefine = 0;
  for (int level = 1; level < h_grid->levels; level++) {
    const size_t refineLimit =
        std::min(h_maxNumBlocksLevel[level - 1] - h_blockLoads[level - 1],
                 8 * h_blockLoads[level] - h_blockLoads[level - 1]);
    if (refineLimit == 0)
      continue;

    const size_t start = 8 * h_levelOffsets[level];
    const size_t end = start + 8 * h_maxNumBlocksLevel[level];
    size_t *destIndices = h_destSubblockIndices + numSubblocksToRefine;
    size_t n = 0;
    for (size_t i = start; i < end; i++)
      if (h_subblockScores[i] > 1e-4f)
        destIndices[n++] = i;

    if (n > refineLimit)
      std::nth_element(destIndices, destIndices + refineLimit, destIndices + n,
                       std::greater{});

    numSubblocksToRefine += std::min(n, refineLimit);
  }

  if (numSubblocksToRefine > 0) {
    cudaMemcpy(d_destSubblockIndices, h_destSubblockIndices,
               numSubblocksToRefine * sizeof(size_t), cudaMemcpyHostToDevice);

    const size_t refineGridSize = iDivUp(numSubblocksToRefine, blockSize);
    k_dcgrid_refine_subblocks<<<refineGridSize, blockSize>>>(
        d_grid, d_destSubblockIndices, numSubblocksToRefine, numBlocksTouched,
        d_touchedBlockIndices);
    getLastCudaErrorSync("Upsample blocks");

    numBlocksTouched += numSubblocksToRefine;
  }
}

void FluidSimulationDCGrid::render(cudaSurfaceObject_t dest,
                                   const int2 &resolution, const float3 &eyePos,
                                   const float3 &eyeDir, const float &fov,
                                   const float3 &sunDir) {
  const dim3 renderGridSize = dim3(iDivUp(resolution.x, renderBlockSize.x),
                                   iDivUp(resolution.y, renderBlockSize.y));

  k_dcgrid_render<<<renderGridSize, renderBlockSize>>>(
      d_grid, dest, resolution, eyePos, eyeDir, fov, sunDir);
}

void FluidSimulationDCGrid::accumulateVelocity() {
  for (int level = 0; level < h_grid->levels - 1; level++) {
    const size_t gridSize = iDivUp(h_maxNumBlocksLevel[level] * 8, 64);
    k_dcgrid_accumulate_velocity<<<gridSize, 64>>>(d_grid, level);
  }
}

void FluidSimulationDCGrid::accumulateDensity() {
  for (int level = 0; level < h_grid->levels - 1; level++) {
    const size_t gridSize = iDivUp(h_maxNumBlocksLevel[level] * 8, 64);
    k_dcgrid_accumulate_density<<<gridSize, 64>>>(d_grid, level);
  }
}

void FluidSimulationDCGrid::accumulateDivergence() {
  for (int level = 0; level < h_grid->levels - 1; level++) {
    const size_t gridSize = iDivUp(h_maxNumBlocksLevel[level] * 8, 64);
    k_dcgrid_accumulate_divergence<<<gridSize, 64>>>(d_grid, level);
  }
}

void FluidSimulationDCGrid::debugStats() {
  cudaMemset(d_stats, 0, h_grid->maxNumBlocks * sizeof(float));
  k_dcgrid_debug_stats<<<gridSize, blockSize>>>(d_grid, d_stats);

  cudaMemcpy(h_stats, d_stats, h_grid->maxNumBlocks * sizeof(float),
             cudaMemcpyDeviceToHost);

  float sum = 0.f;
  for (size_t i = 0; i < h_grid->maxNumBlocks; i++)
    sum += h_stats[i];

  printf("Residual: %g\n", sum);
}