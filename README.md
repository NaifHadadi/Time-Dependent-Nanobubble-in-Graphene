#Electronic Transport in Graphene with Time-dependent Strain-Induced Nanobubbles

![image](https://github.com/user-attachments/assets/1b32d16b-ccb7-4d47-ac8a-4db694ac881d)
![image](https://github.com/user-attachments/assets/7c93c1c8-946f-4cf1-b5f7-37912d25b529)

## Description
This project simulates quantum transport in strained graphene, focusing on time-dependent deformations from nanobubbles that generate pseudo-electric fields and pump valley currents. The code reproduces the results of the paper published in Physical Review B, "Pseudo Electric Field and Pumping Valley Current in Graphene Nano-bubbles" Phys. Rev. B 109, 195405 (2024), arXiv:2310.11904.

It uses a tight-binding model with dynamic strain fields and transport calculations implemented via the Kwant and tkwant package.



The system models:
- **Strained honeycomb lattice** with Gaussian bump potential
- **Zigzag boundary conditions** for nanoribbon geometry
- **Valley-resolved currents** (K and K' points)
- **Time-dependent perturbation** (oscillating nanobubble height)

Key features:
- MPI-parallelized computation using `mpi4py` and Tkwant
- Charge and valley current calculations
- Strain-modulated hopping parameters

## Authors:
- Naif Hadadi
- Adel Abbout
