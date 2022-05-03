#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

struct DCGrid {
  size_t maxNumBlocks = 0;
  int levels = 0;
  int sparseLevels = 0;

  size_t *maxNumBlocksLevel = nullptr;
  size_t *fullBlocksLevel = nullptr;
  size_t *blockLoads = nullptr;
  size_t *levelOffsets = nullptr;

  size_t *hashTableSize = nullptr;
  size_t *hashTableOffset = nullptr;
  uint32_t *hashKey = nullptr;
  size_t *hashVal = nullptr;

  int3 *blockPositions = nullptr;
  uint8_t *blockLevels = nullptr;
  size_t *cellIndices = nullptr;
  size_t *freeBlockIndices = nullptr;
  uint8_t *blockFlags = nullptr;

  size_t *parentIndices = nullptr;
  size_t *childIndices = nullptr;

  // Permanent
  float *density = nullptr;
  float3 *velocity = nullptr;
  float *fluidity = nullptr;

  // Temporary
  void *temporary = nullptr;

  // Aliasses of temporary
  float3 *t_velocity = nullptr;
  float3 *vorticity = nullptr;
  float *divergence = nullptr;
  float *pressure = nullptr;
  float *t_pressure = nullptr;
  float *t_density = nullptr;

  enum : uint8_t {
    blockClean = 0,
    blockMoved = 1,
    blockRefined = 2,
  };

  enum : int {
    blockW = 4,
    blockA = blockW * blockW,
    blockV = blockA * blockW,
    blockBits = 6,

    superblockW = 2 * blockW,

    apronW = blockW + 2,
    apronA = apronW * apronW,
    apronV = apronA * apronW,

    subblockW = blockW / 2,
    subblockA = subblockW * subblockW,
    subblockV = subblockA * subblockW,
    subblockBits = 3,
  };

  enum : size_t { notFound = SIZE_MAX };
  enum : uint32_t { hashEmpty = UINT32_MAX };
};

typedef size_t(index_grid_t)[DCGrid::apronW][DCGrid::apronW][DCGrid::apronW];

// Fluid dynamics
__global__ void k_dcgrid_advect_velocity(DCGrid *grid);
__global__ void k_dcgrid_calc_vorticity(DCGrid *grid);
__global__ void k_dcgrid_calc_divergence(DCGrid *grid);
__global__ void k_dcgrid_apply_pressure(DCGrid *grid);
__global__ void k_dcgrid_advect_density(DCGrid *grid);

// Multigrid
__global__ void k_dcgrid_jacobi(DCGrid *grid, uint8_t targetLevel);
__global__ void k_dcgrid_jacobi_inv(DCGrid *grid, uint8_t targetLevel);
__global__ void k_dcgrid_prolongate(DCGrid *grid, uint8_t targetLevel);

// Structure
__global__ void k_dcgrid_init_aliasses(DCGrid *grid);
__global__ void k_dcgrid_init_apron_indices(DCGrid *grid);
__global__ void k_dcgrid_activate_level(DCGrid *grid, uint8_t level);
__global__ void k_dcgrid_set_fluidity(DCGrid *grid);
__global__ void k_dcgrid_refresh_apron_indices(DCGrid *grid);
__global__ void k_dcgrid_accumulate_velocity(DCGrid *grid, uint8_t targetLevel);
__global__ void k_dcgrid_accumulate_density(DCGrid *grid, uint8_t targetLevel);
__global__ void k_dcgrid_accumulate_divergence(DCGrid *grid,
                                               uint8_t targetLevel);

// Rendering
__global__ void k_dcgrid_render(DCGrid *grid, cudaSurfaceObject_t dest,
                                int2 resolution, float3 eyeOrigin,
                                float3 eyeDir, float fov, float3 sunDir);

// Adaptation
__global__ void k_dcgrid_calc_scores(DCGrid *grid, float *blockScores,
                                     float *subblockScores);
__global__ void k_dcgrid_calc_subblock_scores(DCGrid *grid,
                                              float *subblockScores);
__global__ void k_dcgrid_accumulate_subblock_scores(DCGrid *grid,
                                                    float *blockScores,
                                                    float *subblockScores);
__global__ void k_dcgrid_move_blocks(DCGrid *grid, size_t *blockIndices,
                                     size_t *destSubblockIndices, size_t n);
__global__ void k_dcgrid_propagate_values(DCGrid *grid, size_t *blockIndices,
                                          int targetLevel);
__global__ void k_dcgrid_refill_hash_table(DCGrid *grid);
__global__ void k_dcgrid_refine_subblocks(DCGrid *grid, size_t *subblockIndices,
                                          size_t n, size_t numTouchedBlocks,
                                          size_t *touchedBlockIndices);

// Debug
__global__ void k_dcgrid_debug_stats(DCGrid *grid, float *stats);
