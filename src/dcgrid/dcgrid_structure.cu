#include "dcgrid.h"
#include "dcgrid_utils.cuh"
#include "utils/grid_math.cuh"
#include "utils/sim_utils.h"

__global__ void k_dcgrid_init_apron_indices(DCGrid *grid) {
  const size_t blockIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if (blockIndex >= grid->maxNumBlocks)
    return;

  index_grid_t &cellIndices = reinterpret_cast<index_grid_t &>(
      grid->cellIndices[blockIndex * DCGrid::apronV]);
  for (int i = 0; i < DCGrid::apronW; i++)
    for (int j = 0; j < DCGrid::apronW; j++)
      for (int k = 0; k < DCGrid::apronW; k++)
        cellIndices[i][j][k] = DCGrid::notFound;

  size_t cellIndex = DCGrid::blockV * blockIndex;
  for (int bi = 0; bi < 2; bi++)
    for (int bj = 0; bj < 2; bj++)
      for (int bk = 0; bk < 2; bk++)
        for (int i = 0; i < DCGrid::subblockW; i++)
          for (int j = 0; j < DCGrid::subblockW; j++)
            for (int k = 0; k < DCGrid::subblockW; k++, cellIndex++)
              cellIndices[1 + DCGrid::subblockW * bi + i]
                         [1 + DCGrid::subblockW * bj + j]
                         [1 + DCGrid::subblockW * bk + k] = cellIndex;
}

__global__ void k_dcgrid_refresh_apron_indices(DCGrid *grid) {
  const size_t blockIndex = blockIdx.x;
  const size_t apronIndex = threadIdx.x;
  if (blockIndex > grid->maxNumBlocks || grid->blockLevels[blockIndex] == 0xFF)
    return;
  // Skip inner cells
  const int i = (apronIndex / DCGrid::apronA) % DCGrid::apronW;
  const int j = (apronIndex / DCGrid::apronW) % DCGrid::apronW;
  const int k = apronIndex % DCGrid::apronW;
  if (i % (DCGrid::apronW - 1) != 0 && j % (DCGrid::apronW - 1) != 0 &&
      k % (DCGrid::apronW - 1) != 0)
    return;

  size_t &cellIndex =
      grid->cellIndices[blockIndex * DCGrid::apronV + apronIndex];
  const int3 &p0 = grid->blockPositions[blockIndex];
  const uint8_t &level = grid->blockLevels[blockIndex];
  size_t newNeighborBlockIndex = DCGrid::notFound;

  if (grid->blockFlags[blockIndex] & DCGrid::blockMoved) {
    // Find new cell
    int3 nPos = make_int3(p0.x + i - 1, p0.y + j - 1, p0.z + k - 1);
    const int scale = 1 << level;

    if (nPos.x < 0 || nPos.y < 0 || nPos.z < 0 || nPos.x * scale >= params.gx ||
        nPos.y * scale >= params.gy || nPos.z * scale >= params.gz) {
      cellIndex = blockIndex * DCGrid::blockV +
                  SPREAD(clamp(i, 1, DCGrid::blockW), 2) +
                  SPREAD(clamp(j, 1, DCGrid::blockW), 1) +
                  SPREAD(clamp(k, 1, DCGrid::blockW), 0);
      return;
    }

    uint8_t nLevel = level;
    newNeighborBlockIndex = getBlockIndexDeep(grid, nPos, nLevel);
  } else {
    // Search moved or refined neighbors
    const size_t prevNeighborBlockIndex = cellIndex / DCGrid::blockV;
    const uint8_t &cmpFlags = grid->blockFlags[prevNeighborBlockIndex];

    if (cmpFlags & DCGrid::blockMoved) {
      // Now we need to find a new cell
      int3 nPos = make_int3(p0.x + i - 1, p0.y + j - 1, p0.z + k - 1);
      uint8_t nLevel = level;
      newNeighborBlockIndex = getBlockIndexDeep(grid, nPos, nLevel);
    } else if (cmpFlags & DCGrid::blockRefined &&
               grid->blockLevels[prevNeighborBlockIndex] > level) {
      // Check if subblock is actually refined
      newNeighborBlockIndex = grid->childIndices[cellIndex / DCGrid::subblockV];
    }
  }

  if (newNeighborBlockIndex == DCGrid::notFound)
    return;

  const int3 &np0 = grid->blockPositions[newNeighborBlockIndex];
  const int s = 1 << (grid->blockLevels[newNeighborBlockIndex] - level);
  const int x = (p0.x + i - 1) / s - np0.x;
  const int y = (p0.y + j - 1) / s - np0.y;
  const int z = (p0.z + k - 1) / s - np0.z;
  cellIndex = newNeighborBlockIndex * DCGrid::blockV + SPREAD(x, 2) +
              SPREAD(y, 1) + SPREAD(z, 0);
}

__global__ void k_dcgrid_init_aliasses(DCGrid *grid) {
  const size_t numCells = grid->maxNumBlocks * DCGrid::blockV;
  grid->t_velocity = (float3 *)grid->temporary;
  grid->vorticity = (float3 *)grid->temporary;
  grid->divergence = (float *)grid->temporary;
  grid->pressure = ((float *)grid->temporary) + numCells;
  grid->t_pressure = ((float *)grid->temporary) + 2 * numCells;
  grid->t_density = (float *)grid->temporary;
}

__global__ void k_dcgrid_activate_level(DCGrid *grid, uint8_t level) {
  const int3 pos = DCGrid::blockW * make_int3(blockDim * blockIdx + threadIdx);
  const int scale = 1 << level;

  if (pos.x * scale >= params.gx || pos.y * scale >= params.gy ||
      pos.z * scale >= params.gz)
    return;

  const size_t blockIndex = getBlockIndex(grid, pos, level);
  if (blockIndex >= grid->levelOffsets[level] + grid->maxNumBlocksLevel[level])
    return;

  grid->blockPositions[blockIndex] = pos;
  grid->blockLevels[blockIndex] = level;

  if (level < grid->levels - 1)
    grid->parentIndices[blockIndex] =
        8 * getBlockIndex(grid, pos / 2, level + 1) +
        (pos.x % (2 * DCGrid::blockW)) + (pos.y % (2 * DCGrid::blockW)) / 2 +
        (pos.z % (2 * DCGrid::blockW)) / 4;

  if (level - 1 >= grid->sparseLevels)
    for (int b = 0; b < 8; b++)
      grid->childIndices[8 * blockIndex + b] = getBlockIndex(
          grid,
          2 * pos + DCGrid::blockW * make_int3((b / 4) % 2, (b / 2) % 2, b % 2),
          level - 1);

  index_grid_t &cellIndices = reinterpret_cast<index_grid_t &>(
      grid->cellIndices[blockIndex * DCGrid::apronV]);
  const int i0 = pos.x > 0 ? 0 : 1;
  const int i1 =
      DCGrid::apronW - ((pos.x + DCGrid::blockW) * scale >= params.gx ? 1 : 0);
  const int j0 = pos.y > 0 ? 0 : 1;
  const int j1 =
      DCGrid::apronW - ((pos.y + DCGrid::blockW) * scale >= params.gy ? 1 : 0);
  const int k0 = pos.z > 0 ? 0 : 1;
  const int k1 =
      DCGrid::apronW - ((pos.z + DCGrid::blockW) * scale >= params.gz ? 1 : 0);

  for (int i = i0; i < i1; i++)
    for (int j = j0; j < j1; j++)
      for (int k = k0; k < k1; k++) {
        const size_t nBlockIndex =
            getBlockIndex(grid, pos + make_int3(i - 1, j - 1, k - 1), level);
        index_grid_t &nCellIndices = reinterpret_cast<index_grid_t &>(
            grid->cellIndices[nBlockIndex * DCGrid::apronV]);
        cellIndices[i][j][k] =
            nCellIndices[1 + ((i - 1 + DCGrid::blockW) % DCGrid::blockW)]
                        [1 + ((j - 1 + DCGrid::blockW) % DCGrid::blockW)]
                        [1 + ((k - 1 + DCGrid::blockW) % DCGrid::blockW)];
      }

  for (int i = 0; i < DCGrid::apronW; i++) {
    const int ic = clamp(i, 1, DCGrid::blockW);
    for (int j = 0; j < DCGrid::apronW; j++) {
      const int jc = clamp(j, 1, DCGrid::blockW);
      for (int k = 0; k < DCGrid::apronW; k++) {
        const int kc = clamp(k, 1, DCGrid::blockW);
        if (cellIndices[i][j][k] == DCGrid::notFound)
          cellIndices[i][j][k] = cellIndices[ic][jc][kc];
      }
    }
  }

  for (int i = 1; i <= DCGrid::blockW; i++)
    for (int j = 1; j <= DCGrid::blockW; j++)
      for (int k = 1; k <= DCGrid::blockW; k++) {
        const int3 p = pos + make_int3(i - 1, j - 1, k - 1);
        const size_t cellIndex = cellIndices[i][j][k];
        grid->density[cellIndex] = 0.f;
        grid->velocity[cellIndex] = make_float3(0.f);
        grid->fluidity[cellIndex] = getCellFluidity(p.x, p.y, p.z, scale);
      }
}

__global__ void k_dcgrid_set_fluidity(DCGrid *grid) {
  USE_CELL_INDEX
  USE_CELL_POS
  const int scale = 1 << grid->blockLevels[blockIndex];

  grid->fluidity[cellIndex] = getCellFluidity(x, y, z, scale);
}

template <typename T>
inline __device__ void accumulate(DCGrid *grid, T *channel,
                                  const size_t &subblockIndex,
                                  const T &defaultValue) {
  const size_t &parentSubblockIndex = grid->parentIndices[subblockIndex / 8];

  if (parentSubblockIndex == DCGrid::notFound)
    return;

  const size_t parentCellIndex =
      DCGrid::subblockV * parentSubblockIndex + (subblockIndex % 8);
  channel[parentCellIndex] = defaultValue;

  size_t cellIndex = DCGrid::subblockV * subblockIndex;
  for (int i = 0; i < DCGrid::subblockV; i++, cellIndex++)
    channel[parentCellIndex] += channel[cellIndex];
  channel[parentCellIndex] *= .125f;
}

__global__ void k_dcgrid_accumulate_velocity(DCGrid *grid,
                                             uint8_t targetLevel) {
  USE_SUBBLOCK_INDEX_LEVEL
  accumulate(grid, grid->velocity, subblockIndex, make_float3(0.f, 0.f, 0.f));
}

__global__ void k_dcgrid_accumulate_density(DCGrid *grid, uint8_t targetLevel) {
  USE_SUBBLOCK_INDEX_LEVEL
  accumulate(grid, grid->density, subblockIndex, 0.f);
}

__global__ void k_dcgrid_accumulate_divergence(DCGrid *grid,
                                               uint8_t targetLevel) {
  USE_SUBBLOCK_INDEX_LEVEL
  accumulate(grid, grid->divergence, subblockIndex, 0.f);
}

__global__ void k_dcgrid_debug_stats(DCGrid *grid, float *stats) {
  USE_BLOCK_INDEX
  index_grid_t &cellIndices = reinterpret_cast<index_grid_t &>(
      grid->cellIndices[DCGrid::apronV * blockIndex]);
  const int scale = 1 << grid->blockLevels[blockIndex];

  const float alpha = params.rdx * params.rdx / (scale * scale);

  for (int i = 1; i <= DCGrid::blockW; i++)
    for (int j = 1; j <= DCGrid::blockW; j++)
      for (int k = 1; k <= DCGrid::blockW; k++) {
        const size_t &idx = cellIndices[i][j][k];
        if (grid->childIndices[idx >> DCGrid::subblockBits] != DCGrid::notFound)
          continue;

        const float &pc = grid->pressure[idx];
        const float &pl = grid->pressure[cellIndices[i - 1][j][k]];
        const float &pr = grid->pressure[cellIndices[i + 1][j][k]];
        const float &pd = grid->pressure[cellIndices[i][j - 1][k]];
        const float &pu = grid->pressure[cellIndices[i][j + 1][k]];
        const float &pb = grid->pressure[cellIndices[i][j][k - 1]];
        const float &pf = grid->pressure[cellIndices[i][j][k + 1]];

        const float residual = grid->divergence[idx] -
                               (pl + pr + pd + pu + pb + pf - 6.f * pc) * alpha;
        stats[blockIndex] += scale * fabsf(residual);
      }
}
