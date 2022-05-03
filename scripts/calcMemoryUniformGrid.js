function calcMipmapCells(w, h, d) {
  let numCells = 0;
  for (s = 1; w % s === 0 && h % s === 0 && d % s === 0; s *= 2)
    numCells += (w * h * d) / (s * s * s);
  return numCells;
}

const args = process.argv.slice(2);
const w = parseInt(args[0]);
const h = parseInt(args[1]);
const d = parseInt(args[2]);
const numChannels = args[3] ? parseInt(args[3]) : 18;

const memoryPerCell = 4 * numChannels + 1;
const numCells = w * h * d;
const mipmapCells = calcMipmapCells(w, h, d);
const memory = numCells * memoryPerCell + (mipmapCells - numCells) * 4;

console.log(
  `Uniform grid: ${memory / (1024 * 1024)}MB for ${numCells} cells (${Math.ceil(
    memory / numCells
  )}B / cell)`
);
