#include "dcgrid.h"
#include "dcgrid_utils.cuh"
#include "utils/sim_utils.h"
#include <float.h>

inline __device__ float calcCellScore(DCGrid *grid, const size_t &cellIndex) {
  return length(grid->vorticity[cellIndex]);
}

__global__ void k_dcgrid_calc_subblock_scores(DCGrid *grid,
                                              float *subblockScores) {
  const size_t subblockIndex = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t blockIndex = subblockIndex / 8;
  if (blockIndex >= grid->maxNumBlocks)
    return;

  const uint8_t &level = grid->blockLevels[blockIndex];
  if (grid->childIndices[subblockIndex] != DCGrid::notFound || level == 0xFF ||
      (level == 0 && grid->blockLoads[0] == grid->fullBlocksLevel[0]) ||
      (level > 0 &&
       grid->blockLoads[level - 1] == grid->fullBlocksLevel[level - 1])) {
    subblockScores[subblockIndex] = -FLT_MAX;
    return;
  }

  const float3 p =
      (1 << level) *
      (make_float3(grid->blockPositions[blockIndex]) +
       2.f * make_float3((subblockIndex >> 2) & 1, (subblockIndex >> 1) & 1,
                         (subblockIndex >> 0) & 1) +
       1.f);
  const float d = length(
      p - make_float3(.5f * params.gx, .45f * params.gy, .5f * params.gz));
  subblockScores[subblockIndex] = d < .2f * params.gx ? 0.f : params.gx / d;

  // subblockScores[subblockIndex] = 0.f;
  // size_t cellIndex = DCGrid::subblockV * subblockIndex;
  // for (int i = 0; i < DCGrid::subblockV; i++, cellIndex++)
  //   subblockScores[subblockIndex] += calcCellScore(grid, cellIndex);
}

__global__ void k_dcgrid_accumulate_subblock_scores(DCGrid *grid,
                                                    float *blockScores,
                                                    float *subblockScores) {
  const size_t blockIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if (blockIndex >= grid->maxNumBlocks)
    return;

  blockScores[blockIndex] = -FLT_MAX;

  const uint8_t &level = grid->blockLevels[blockIndex];
  if (level == 0xFF ||
      (level == 0 && grid->blockLoads[0] == grid->fullBlocksLevel[0]) ||
      (level > 0 &&
       grid->blockLoads[level - 1] == grid->fullBlocksLevel[level - 1]))
    return;

  const float *s = subblockScores + 8 * blockIndex;
  if (s[0] > 0.f && s[1] > 0.f && s[2] > 0.f && s[3] > 0.f && s[4] > 0.f &&
      s[5] > 0.f && s[6] > 0.f && s[7] > 0.f)
    blockScores[blockIndex] =
        .125f * (s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7] + s[8]);
}

__global__ void k_dcgrid_move_blocks(DCGrid *grid, size_t *blockIndices,
                                     size_t *nextSubblockIndices, size_t n) {
  const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n)
    return;
  const size_t &blockIndex = blockIndices[idx];

  // Activate previous parent subblock
  const size_t &prevParentSubblockIndex = grid->parentIndices[blockIndex];
  grid->childIndices[prevParentSubblockIndex] = DCGrid::notFound;

  // Move block
  const size_t &nextParentSubblockIndex = nextSubblockIndices[idx];
  grid->childIndices[nextParentSubblockIndex] = blockIndex;
  grid->parentIndices[blockIndex] = nextParentSubblockIndex;

  const size_t nextParentBlockIndex = nextParentSubblockIndex / 8;
  const int3 d = make_int3((nextParentSubblockIndex / 4) % 2,
                           (nextParentSubblockIndex / 2) % 2,
                           (nextParentSubblockIndex / 1) % 2);

  grid->blockPositions[blockIndex] =
      2 * grid->blockPositions[nextParentBlockIndex] + DCGrid::blockW * d;
  grid->blockFlags[blockIndex] |= DCGrid::blockMoved;
  grid->blockFlags[nextParentBlockIndex] |= DCGrid::blockRefined;
}

__global__ void k_dcgrid_propagate_values(DCGrid *grid, size_t *blockIndices,
                                          int targetLevel) {
  const size_t &blockIndex = blockIndices[blockIdx.x];
  if (grid->blockLevels[blockIndex] != targetLevel)
    return;

  const size_t cellIndex = blockIndex * DCGrid::blockV + threadIdx.x;
  const size_t subblockIndex = cellIndex >> DCGrid::subblockBits;

  const size_t &parentSubblockIndex = grid->parentIndices[blockIndex];
  if (parentSubblockIndex == DCGrid::notFound)
    return;
  size_t *parentCellIndices =
      grid->cellIndices + (parentSubblockIndex / 8) * DCGrid::apronV;
  const int scale = 1 << targetLevel;

  USE_CELL_POS

  const int px = (x / 2) % DCGrid::blockW;
  const int py = (y / 2) % DCGrid::blockW;
  const int pz = (z / 2) % DCGrid::blockW;
  const size_t idx =
      DCGrid::apronA * (1 + px) + DCGrid::apronW * (1 + py) + (1 + pz);
  const int i = x % 2 ? DCGrid::apronA : -DCGrid::apronA;
  const int j = y % 2 ? DCGrid::apronW : -DCGrid::apronW;
  const int k = z % 2 ? 1 : -1;

  const size_t &i000 = parentCellIndices[idx];
  const size_t &i001 = parentCellIndices[idx + k];
  const size_t &i010 = parentCellIndices[idx + j];
  const size_t &i100 = parentCellIndices[idx + i];
  const size_t &i011 = parentCellIndices[idx + j + k];
  const size_t &i101 = parentCellIndices[idx + i + k];
  const size_t &i110 = parentCellIndices[idx + i + j];
  const size_t &i111 = parentCellIndices[idx + i + j + k];

  grid->density[cellIndex] =
      ((27.f / 64.f) * grid->density[i000] +
       (9.f / 64.f) *
           (grid->density[i001] + grid->density[i010] + grid->density[i100]) +
       (3.f / 64.f) *
           (grid->density[i011] + grid->density[i101] + grid->density[i110]) +
       (1.f / 64.f) * grid->density[i111]);
  grid->velocity[cellIndex] =
      ((27.f / 64.f) * grid->velocity[i000] +
       (9.f / 64.f) * (grid->velocity[i001] + grid->velocity[i010] +
                       grid->velocity[i100]) +
       (3.f / 64.f) * (grid->velocity[i011] + grid->velocity[i101] +
                       grid->velocity[i110]) +
       (1.f / 64.f) * grid->velocity[i111]);
  grid->fluidity[cellIndex] = getCellFluidity(x, y, z, scale);
}

__global__ void k_dcgrid_refine_subblocks(DCGrid *grid, size_t *subblockIndices,
                                          size_t n, size_t numTouchedBlocks,
                                          size_t *touchedBlockIndices) {
  const size_t rank = blockDim.x * blockIdx.x + threadIdx.x;
  if (rank >= n)
    return;
  const size_t subblockIndex = subblockIndices[rank];

  if (subblockIndex == DCGrid::notFound)
    return;

  const size_t parentBlockIndex = subblockIndex / 8;
  const uint8_t &level = grid->blockLevels[parentBlockIndex];
  const int3 parentBlockPosition = grid->blockPositions[parentBlockIndex];
  const int3 d = make_int3((subblockIndex / 4) % 2, (subblockIndex / 2) % 2,
                           subblockIndex % 2);
  const int3 childBlockPosition = parentBlockPosition * 2 + d * DCGrid::blockW;
  const size_t childBlockIndex =
      insertBlock(grid, childBlockPosition, level - 1);

  if (childBlockIndex == DCGrid::notFound) {
    printf("No space left to refine to %d %d %d @%d\n", childBlockPosition.x,
           childBlockPosition.y, childBlockPosition.z, level - 1);
    return;
  }

  grid->parentIndices[childBlockIndex] = subblockIndex;
  grid->childIndices[subblockIndex] = childBlockIndex;

  grid->blockFlags[parentBlockIndex] |= DCGrid::blockRefined;
  grid->blockFlags[childBlockIndex] |= DCGrid::blockMoved;

  touchedBlockIndices[numTouchedBlocks + rank] = childBlockIndex;
}

__global__ void k_dcgrid_refill_hash_table(DCGrid *grid) {
  USE_BLOCK_INDEX

  const uint8_t &level = grid->blockLevels[blockIndex];
  const uint32_t key = gridHash(grid->blockPositions[blockIndex]);
  const size_t &o = grid->hashTableOffset[level];
  size_t slot = key % grid->hashTableSize[level];
  const size_t slot0 = slot;

  do {
    const uint32_t prev =
        atomicCAS(&grid->hashKey[o + slot], DCGrid::hashEmpty, key);
    if (prev == DCGrid::hashEmpty || prev == key) {
      grid->hashVal[o + slot] = blockIndex;
      return;
    }

    slot = (slot + 127) % grid->hashTableSize[level];
  } while (slot != slot0);

  printf("Hashtable fully loaded. Shouldn't occur\n");
  return;
}