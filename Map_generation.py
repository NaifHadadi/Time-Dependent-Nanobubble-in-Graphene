#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel Wavefunction Evolution for Quantum Transport
-----------------------------------------------------
This script implements parallel computation of time-dependent wavefunctions
for strained graphene systems using MPI-distributed processing.

Key Features:
- MPI-parallelized wavefunction evolution
- Strain-engineered graphene Hamiltonian
- Automatic boundary conditions for time evolution
- Distributed storage of wavefunction snapshots
"""

import kwant
import tkwant
import numpy as np
from math import sqrt
import tinyarray
from numpy import pi, sqrt, arccos
from kwant.wraparound import wraparound
import scipy
import warnings
from tkwant import onebody, leads
from mpi4py import MPI

# Suppress warnings
warnings.filterwarnings('ignore')

# System parameters
Lx, Ly = 20, 25
params = {
    'Ep': 0, 'omega': 0.01, 'dh': 0.35, 'hop': -2.7,
    'lat_constant': 0.24595, 'beta': 3.37, 'height': 3.5,
    'sigma': 5, 'a': 0.2, 'b': 0.5,
    'center': (0, 0), 'E': 1, 'vf': 1, 'hbar': 1
}

# Lattice definition
lat = kwant.lattice.honeycomb(a=params['lat_constant'], norbs=1, name=["a", "b"])
a, b = lat.sublattices

# System building functions
def onsite(site, Ep):
    return Ep

def hopping(site_1, site_2, hop):
    return hop

def gaussian_bump_transform(pos, height, sigma, center):
    """Calculate height deformation due to Gaussian bump potential"""
    x, y = pos
    cx, cy = center[0], center[1]
    r = sqrt((x-cx)**2 + (y-cy)**2)
    return height * np.exp(-r**2/(2*sigma**2))

def strained_hopping(site1, site2, beta, lat_constant, height, sigma, center, hop, time):
    """Calculate strain-modified hopping energy"""
    dz1 = gaussian_bump_transform(site1.pos, height + params['dh']*np.sin(params['omega']*time), sigma, center)
    dz2 = gaussian_bump_transform(site2.pos, height + params['dh']*np.sin(params['omega']*time), sigma, center)
    
    d_strained = np.sqrt((site1.pos[0]-site2.pos[0])**2 + 
                        (site1.pos[1]-site2.pos[1])**2 + 
                        (dz1-dz2)**2)
    d_nearest = lat_constant/sqrt(3)
    return hop * np.exp(-beta * ((d_strained/d_nearest) - 1))

def make_system(length, width, boundary):
    """Construct the quantum system with specified boundary conditions"""
    def rectangle(pos):
        x, y = pos
        return abs(x) < length and abs(y) < width
    
    system = kwant.Builder()
    system[lat.shape(rectangle, (0, 0))] = onsite
    system[lat.neighbors(1)] = strained_hopping

    if boundary == 'zigzag':
        symmetry = kwant.TranslationalSymmetry(lat.vec((-1, 0)))
        symmetry.add_site_family(lat.sublattices[0], other_vectors=[(-1, 2)])
        symmetry.add_site_family(lat.sublattices[1], other_vectors=[(-1, 2)])

        def lead_shape(pos):
            x, y = pos
            return abs(y) < width

        lead = kwant.Builder(symmetry)
        lead[lat.shape(lead_shape, (0, 0))] = onsite
        lead[lat.neighbors(1)] = hopping

        system.eradicate_dangling()
        lead.eradicate_dangling()

        system.attach_lead(lead)
        system.attach_lead(lead.reversed())

    return {'system': system, 'lead': lead}

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocessor = comm.size

def is_master():
    """Check if current process is the master rank"""
    return tkwant.mpi.get_communicator().rank == 0

# Pauli matrices for calculations
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Main execution
if __name__ == "__main__":
    # Time parameters
    times = np.linspace(0, 4000, 800)
    
    # System construction
    system_data = make_system(Lx, Ly, 'zigzag')
    system = system_data['system']
    finalized_system = system.finalized()
    
    # Initialize wavefunction evolution
    boundaries = leads.automatic_boundary(
        finalized_system.leads,
        tmax=max(times),
        params=params
    )
    
    # Get initial scattering states
    scattering_states = kwant.wave_function(
        finalized_system,
        energy=0.07,
        params=dict({'time': 0}, **params)
    )
    
    # Combine states from both leads
    Psi = np.concatenate((scattering_states(0), scattering_states(1)))
    np.save('Psi', Psi)
    
    # Distribute wavefunction evolution across MPI ranks
    Psi = Psi[:4]  # Use first 4 states
    if rank < Psi.shape[0]:
        wave_func = onebody.WaveFunction.from_kwant(
            finalized_system,
            Psi[rank],
            boundaries=boundaries,
            energy=0.07,
            params=params
        )
        
        # Evolve and save wavefunctions at each time step
        for time in times:
            wave_func.evolve(time)
            current_psi = wave_func.psi()
            np.save(f'Psi{rank}_{time}', current_psi)