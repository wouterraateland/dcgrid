function calcMipmapCells(w, h) {
  let numCells = 0;
  for (s = 1; w % s === 0 && h % s === 0; s *= 2) numCells += (w * h) / (s * s);
  return numCells;
}

const args = process.argv.slice(2);
const w = parseInt(args[0]);
const h = parseInt(args[1]);

const numCells = calcMipmapCells(w, h);

const memoryPerSliceCell = 4 * (3 + 5);
const memoryPerGroundCell = 4 * 10;
const memoryPerSurfaceCell = 4 * 12;
const memoryPerCell =
  memoryPerGroundCell + memoryPerSurfaceCell + memoryPerSliceCell;

console.log(
  `Terrain: ${
    (numCells * memoryPerCell) / (1024 * 1024)
  }MB for ${numCells} cells (${memoryPerCell}B / cell)`
);
