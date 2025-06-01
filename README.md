# Time-Dependent Quantum Transport in Graphene with Strain-Induced Nanobubbles

![image](https://github.com/user-attachments/assets/1b32d16b-ccb7-4d47-ac8a-4db694ac881d)
![image](https://github.com/user-attachments/assets/0c7ddecd-762a-4b66-9c1e-cd6086276d39)

## Description
This project simulates quantum transport in graphene with time-dependent strain induced by a nanobubble deformation. The code reproduces the published simulations in the paper Pseudo Electric Field and Pumping Valley Current in Graphene Nano-bubbles:
https://arxiv.org/pdf/2310.11904

The system models:
- **Strained honeycomb lattice** with Gaussian bump potential
- **Zigzag boundary conditions** for nanoribbon geometry
- **Valley-resolved currents** (K and K' points)
- **Time-dependent perturbation** (oscillating nanobubble height)

Key features:
- MPI-parallelized computation using `mpi4py` and T-Kwant
- Charge and valley current calculations
- Strain-modulated hopping parameters

