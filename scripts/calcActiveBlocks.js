const args = process.argv.slice(2);
const totalBlocks = parseInt(args[0]);
const totalSubblocks = 8 * totalBlocks;
const totalCells = 8 * totalSubblocks;

// "root" block gives 8 active subblock, each other block gives 8 and takes 1
//   8 + (8 - 1)(n - 1)
// = 8 + 7n - 7
// = 7n + 1
const activeSubblocks = 7 * totalBlocks + 1;
const activeCells = 8 * activeSubblocks;

console.log(`Total blocks: ${totalBlocks}`);
console.log(`Active subblocks: ${activeSubblocks} / ${totalSubblocks}`);
console.log(`Active cells: ${activeCells} / ${totalCells}`);
