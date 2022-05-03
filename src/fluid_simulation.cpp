#include "fluid_simulation.h"

FluidSimulation::FluidSimulation(const int3 &size)
    : size(size), numCells(size.x * size.y * size.z) {}

FluidSimulation::~FluidSimulation() {}