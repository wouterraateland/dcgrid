#include "dcgrid.h"
#include "dcgrid_utils.cuh"
#include "utils/sim_utils.h"

__global__ void k_dcgrid_jacobi(DCGrid *grid, uint8_t targetLevel) {
  USE_CELL_INDEX_LEVEL
  USE_APRON_INDEX
  const size_t *cellIndices = grid->cellIndices + DCGrid::apronV * blockIndex;

  const int blockScale = 1 << targetLevel;
  const float alpha = blockScale * blockScale * params.dx * params.dx;

  const float &pl = grid->pressure[cellIndices[IDX_L]];
  const float &pr = grid->pressure[cellIndices[IDX_R]];
  const float &pd = grid->pressure[cellIndices[IDX_D]];
  const float &pu = grid->pressure[cellIndices[IDX_U]];
  const float &pb = grid->pressure[cellIndices[IDX_B]];
  const float &pf = grid->pressure[cellIndices[IDX_F]];

  grid->t_pressure[cellIndex] =
      (pl + pr + pd + pu + pb + pf - alpha * grid->divergence[cellIndex]) / 6.f;
}

__global__ void k_dcgrid_jacobi_inv(DCGrid *grid, uint8_t targetLevel) {
  USE_CELL_INDEX_LEVEL
  USE_APRON_INDEX
  const size_t *cellIndices = grid->cellIndices + DCGrid::apronV * blockIndex;

  const int blockScale = 1 << targetLevel;
  const float alpha = blockScale * blockScale * params.dx * params.dx;

  const float &pl = grid->t_pressure[cellIndices[IDX_L]];
  const float &pr = grid->t_pressure[cellIndices[IDX_R]];
  const float &pd = grid->t_pressure[cellIndices[IDX_D]];
  const float &pu = grid->t_pressure[cellIndices[IDX_U]];
  const float &pb = grid->t_pressure[cellIndices[IDX_B]];
  const float &pf = grid->t_pressure[cellIndices[IDX_F]];

  grid->pressure[cellIndex] =
      (pl + pr + pd + pu + pb + pf - alpha * grid->divergence[cellIndex]) / 6.f;
}

__global__ void k_dcgrid_prolongate(DCGrid *grid, uint8_t targetLevel) {
  USE_CELL_INDEX_LEVEL

  const size_t &parentSubblockIndex = grid->parentIndices[blockIndex];

  if (parentSubblockIndex == DCGrid::notFound)
    return;
  size_t *parentCellIndices =
      grid->cellIndices + (parentSubblockIndex / 8) * DCGrid::apronV;

  USE_CELL_POS

  const int px = (x / 2) % DCGrid::blockW;
  const int py = (y / 2) % DCGrid::blockW;
  const int pz = (z / 2) % DCGrid::blockW;
  const int idx =
      DCGrid::apronA * (1 + px) + DCGrid::apronW * (1 + py) + (1 + pz);
  const int i = x % 2 ? DCGrid::apronA : -DCGrid::apronA;
  const int j = y % 2 ? DCGrid::apronW : -DCGrid::apronW;
  const int k = z % 2 ? 1 : -1;

  const float &p000 = grid->pressure[parentCellIndices[idx]];
  const float &p001 = grid->pressure[parentCellIndices[idx + k]];
  const float &p010 = grid->pressure[parentCellIndices[idx + j]];
  const float &p100 = grid->pressure[parentCellIndices[idx + i]];
  const float &p011 = grid->pressure[parentCellIndices[idx + j + k]];
  const float &p101 = grid->pressure[parentCellIndices[idx + i + k]];
  const float &p110 = grid->pressure[parentCellIndices[idx + i + j]];
  const float &p111 = grid->pressure[parentCellIndices[idx + i + j + k]];

  grid->pressure[cellIndex] = (27.f * p000 + 9.f * (p001 + p010 + p100) +
                               3.f * (p011 + p101 + p110) + p111) /
                              64.f;
}
