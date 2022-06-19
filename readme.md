# DCGrid

> **Please note:** The project won't run out of the box. For managing the window, UI, user input, etc. A proprietary library was used, which I can't include here. Only `main.cpp` and `simulation.*` are affected. I'd be happy to accept a PR that allows this project to run without the library. If you're interested in helping out, feel free to reach out for any questions.

## Prerequisites

- Visual Studio 2019
- CUDA v11.1 or later
- VCPKG
- C++ boost

## Installation

1. Clone the repository
2. Open the repository in Visual Studio
3. Run the project in Visual Studio (Preferably in "Release" or "Release with debug symbols" mode)

## Controls

| Key          | Description                                    |
| ------------ | ---------------------------------------------- |
| <kbd>N       | Switch between uniform and adaptive simulation |
| <kbd>H       | Reset simulation                               |
| <kbd>Space   | Toggle Pause / Running                         |
| <kbd>T       | Toggle terrain debugging                       |
| <kbd>1       | Set render mode: normal                        |
| <kbd>2       | Set render mode: pressure                      |
| <kbd>3       | Set render mode: temperature                   |
| <kbd>4       | Set render mode: resolution                    |
| <kbd>5       | Set render mode: vapor                         |
| <kbd>6       | Set render mode: velocity                      |
| <kbd>W S A D | Move camera                                    |
| Mousewheel   | Move camera back and forth                     |

Explore the UI for further interactions.

## Related paper

- [DCGrid: An Adaptive Grid Structure for
  Memory-Constrained Fluid Simulation on the GPU](https://graphics.tudelft.nl/~klaus/papers/DCGrid.pdf)
