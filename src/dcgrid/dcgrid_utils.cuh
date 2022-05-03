#pragma once
#include "dcgrid.h"
#include "utils/cudamath.cuh"
#include "utils/grid_math.cuh"
#include "utils/sim_utils.h"
#include <stdio.h>

#define USE_BLOCK_INDEX                                                        \
  const size_t blockIndex = blockDim.x * blockIdx.x + threadIdx.x;             \
  if (blockIndex >= grid->maxNumBlocks ||                                      \
      grid->blockLevels[blockIndex] == 0xFF)                                   \
    return;

#define USE_BLOCK_INDEX_LEVEL                                                  \
  if (blockDim.x * blockIdx.x + threadIdx.x >                                  \
      grid->maxNumBlocksLevel[targetLevel])                                    \
    return;                                                                    \
  const size_t blockIndex =                                                    \
      grid->levelOffsets[targetLevel] + blockDim.x * blockIdx.x + threadIdx.x;

#define USE_SUBBLOCK_INDEX_LEVEL                                               \
  size_t subblockIndex = blockDim.x * blockIdx.x + threadIdx.x;                \
  size_t blockIndex = subblockIndex / 8;                                       \
  if (blockIndex > grid->maxNumBlocksLevel[targetLevel])                       \
    return;                                                                    \
  blockIndex += grid->levelOffsets[targetLevel];                               \
  if (grid->blockLevels[blockIndex] != targetLevel)                            \
    return;                                                                    \
  subblockIndex += 8 * grid->levelOffsets[targetLevel];

#define USE_CELL_INDEX                                                         \
  size_t cellIndex = blockDim.x * blockIdx.x + threadIdx.x;                    \
  size_t blockIndex = cellIndex >> DCGrid::blockBits;                          \
  if (blockIndex >= grid->maxNumBlocks ||                                      \
      grid->blockLevels[blockIndex] == 0xFF)                                   \
    return;                                                                    \
  size_t subblockIndex = cellIndex >> DCGrid::subblockBits;

#define USE_CELL_INDEX_LEVEL                                                   \
  size_t cellIndex = blockDim.x * blockIdx.x + threadIdx.x;                    \
  size_t blockIndex = cellIndex >> DCGrid::blockBits;                          \
  if (blockIndex > grid->maxNumBlocksLevel[targetLevel])                       \
    return;                                                                    \
  blockIndex += grid->levelOffsets[targetLevel];                               \
  if (grid->blockLevels[blockIndex] != targetLevel)                            \
    return;                                                                    \
  cellIndex += DCGrid::blockV * grid->levelOffsets[targetLevel];               \
  size_t subblockIndex = cellIndex >> DCGrid::subblockBits;

#define USE_APRON_INDEX                                                        \
  const int apronX =                                                           \
      1 + ((((subblockIndex >> 2) << 1) & 2) | ((cellIndex >> 2) & 1));        \
  const int apronY =                                                           \
      1 + ((((subblockIndex >> 1) << 1) & 2) | ((cellIndex >> 1) & 1));        \
  const int apronZ =                                                           \
      1 + ((((subblockIndex >> 0) << 1) & 2) | ((cellIndex >> 0) & 1));        \
  const int apronIndex =                                                       \
      DCGrid::apronA * apronX + DCGrid::apronW * apronY + apronZ;

#define IDX_L apronIndex - DCGrid::apronA
#define IDX_R apronIndex + DCGrid::apronA
#define IDX_D apronIndex - DCGrid::apronW
#define IDX_U apronIndex + DCGrid::apronW
#define IDX_B apronIndex - 1
#define IDX_F apronIndex + 1

#define SPREAD(data, offset) ((((data)&1) | (((data) << 2) & 8)) << (offset))

#define IDX2POS                                                                \
  p0.x | (((subblockIndex >> 2) << 1) & 2) | ((cellIndex >> 2) & 1),           \
      p0.y | (((subblockIndex >> 1) << 1) & 2) | ((cellIndex >> 1) & 1),       \
      p0.z | (((subblockIndex >> 0) << 1) & 2) | ((cellIndex >> 0) & 1)

#define USE_CELL_POS                                                           \
  const int3 &p0 = grid->blockPositions[blockIndex];                           \
  const int x =                                                                \
      p0.x | (((subblockIndex >> 2) << 1) & 2) | ((cellIndex >> 2) & 1);       \
  const int y =                                                                \
      p0.y | (((subblockIndex >> 1) << 1) & 2) | ((cellIndex >> 1) & 1);       \
  const int z =                                                                \
      p0.z | (((subblockIndex >> 0) << 1) & 2) | ((cellIndex >> 0) & 1);

inline __device__ uint32_t bitSpread3(uint32_t data) {
  uint32_t res = 0;

  for (uint32_t mask = 0x00000001Ui32; mask; mask <<= 3, data <<= 2)
    res |= data & mask;

  return res;
}

inline __device__ uint32_t gridHash(const int3 &blockPosition) {
  return (bitSpread3(blockPosition.x / DCGrid::blockW) << 2) |
         (bitSpread3(blockPosition.y / DCGrid::blockW) << 1) |
         bitSpread3(blockPosition.z / DCGrid::blockW);
}

// Find the block offset in hash table, create a new block if not found
inline __device__ size_t insertBlock(DCGrid *grid, const int3 &position,
                                     const int &level) {
  const uint32_t key = gridHash(position);
  const size_t &o = grid->hashTableOffset[level];
  size_t slot = key % grid->hashTableSize[level];
  const size_t slot0 = slot;

  do {
    const uint32_t prev =
        atomicCAS(&grid->hashKey[o + slot], DCGrid::hashEmpty, key);
    // Return existing block
    if (prev == key) {
      printf("Block %d %d %d @%d already exists\n", position.x, position.y,
             position.z, level);
      return grid->hashVal[o + slot];
    }

    if (prev == DCGrid::hashEmpty) {
      // Create new block
      const size_t allocatedBlocks = atomicAdd(&grid->blockLoads[level], 1);
      // No space left
      if (allocatedBlocks >= grid->maxNumBlocksLevel[level]) {
        atomicSubSize_t(&grid->blockLoads[level], 1);
        printf("All blocks allocated on level %d\n", level);
        return DCGrid::notFound;
      }

      const size_t blockIndex =
          grid->freeBlockIndices[grid->levelOffsets[level] + allocatedBlocks];
      grid->hashVal[o + slot] = blockIndex;
      grid->blockPositions[blockIndex] = position;
      grid->blockLevels[blockIndex] = level;
      return blockIndex;
    }

    // Search next entry
    slot = (slot + 127) % grid->hashTableSize[level];
  } while (slot != slot0);

  printf("Hashtable fully loaded. Shouldn't occur\n");
  return DCGrid::notFound;
}

inline __device__ void deleteBlock(DCGrid *grid, const size_t &blockIndex) {
  const uint8_t &level = grid->blockLevels[blockIndex];
  const uint32_t key = gridHash(grid->blockPositions[blockIndex]);
  const size_t &o = grid->hashTableOffset[level];
  size_t slot = key % grid->hashTableSize[level];
  const size_t slot0 = slot;

  do {
    if (grid->hashKey[o + slot] == key) {
      grid->hashVal[o + slot] = DCGrid::notFound;

      const size_t allocatedBlocks =
          atomicSubSize_t(&grid->blockLoads[level], 1);
      grid->freeBlockIndices[grid->levelOffsets[level] + allocatedBlocks - 1] =
          blockIndex;
      return;
    }
    if (grid->hashKey[o + slot] == DCGrid::hashEmpty) {
      printf("No block to delete\n");
      return;
    }

    slot = (slot + 127) % grid->hashTableSize[level];
  } while (slot != slot0);
}

// Find the block offset in hash table
inline __device__ size_t getBlockIndex(DCGrid *grid, const int3 &blockPosition,
                                       const uint8_t &level) {
  // Special case, blocks are ordered
  if (level >= grid->sparseLevels) {
    const int extent = DCGrid::blockW << level;
    const size_t rx = iDivUp(params.gx, extent);
    const size_t ry = iDivUp(params.gy, extent);
    const size_t rz = iDivUp(params.gz, extent);
    const int px = blockPosition.x / DCGrid::blockW;
    const int py = blockPosition.y / DCGrid::blockW;
    const int pz = blockPosition.z / DCGrid::blockW;
    return px >= rx || py >= ry || pz >= rz
               ? DCGrid::notFound
               : grid->levelOffsets[level] + (px * ry + py) * rz + pz;
  }

  const uint32_t key = gridHash(blockPosition);
  const size_t &o = grid->hashTableOffset[level];
  size_t slot = key % grid->hashTableSize[level];
  const size_t slot0 = slot;
  do {
    if (grid->hashKey[o + slot] == key)
      return grid->hashVal[o + slot];
    if (grid->hashKey[o + slot] == DCGrid::hashEmpty)
      return DCGrid::notFound;

    slot = (slot + 127) % grid->hashTableSize[level];
  } while (slot != slot0);

  return DCGrid::notFound;
}

inline __device__ size_t getBlockIndexDeep(DCGrid *grid, int3 &position,
                                           uint8_t &level) {
  if (level < grid->sparseLevels)
    for (uint32_t key = gridHash(position); level < grid->sparseLevels;
         level++, key >>= 3, position /= 2) {
      const size_t &o = grid->hashTableOffset[level];
      size_t slot = key % grid->hashTableSize[level];
      const size_t slot0 = slot;

      do {
        if (grid->hashKey[o + slot] == key)
          return grid->hashVal[o + slot];
        if (grid->hashKey[o + slot] == DCGrid::hashEmpty)
          break;

        slot = (slot + 127) % grid->hashTableSize[level];
      } while (slot != slot0);
    }

  position >>= grid->sparseLevels - level;
  level = grid->sparseLevels;

  const int extent = DCGrid::blockW << level;
  const size_t rx = iDivUp(params.gx, extent);
  const size_t ry = iDivUp(params.gy, extent);
  const size_t rz = iDivUp(params.gz, extent);
  const int px = position.x / DCGrid::blockW;
  const int py = position.y / DCGrid::blockW;
  const int pz = position.z / DCGrid::blockW;
  return px >= rx || py >= ry || pz >= rz
             ? DCGrid::notFound
             : grid->levelOffsets[level] + (px * ry + py) * rz + pz;
}