const args = process.argv.slice(2);
const numBlocks = parseInt(args[0]);
const numChannels = args[1] ? parseInt(args[1]) : 12;

const indexSize = 8;
const hashTableEntriesPerBlock = 4;
const cellsPerBlock = 4 * 4 * 4;
const cellsPerApron = 6 * 6 * 6;
const subblocksPerBlock = 2 * 2 * 2;

const memoryPerLevel = 12 * indexSize;

const memoryPerBlock =
  1 * (14 + 5 * indexSize) +
  subblocksPerBlock * (12 + indexSize) +
  hashTableEntriesPerBlock * (8 + indexSize) +
  cellsPerBlock * (4 * numChannels + 1) +
  cellsPerApron * indexSize;

console.log(
  `DCGrid: ${
    (memoryPerLevel + numBlocks * memoryPerBlock) / (1024 * 1024)
  }MB for ${numBlocks * cellsPerBlock} cells (${Math.ceil(
    memoryPerBlock / cellsPerBlock
  )}B / cell)`
);
