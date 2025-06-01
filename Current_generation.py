#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum Transport Simulation in Strained Graphene Nanoribbons

This script implements a time-dependent quantum transport simulation for graphene nanoribbons
with strain-induced deformation using the kwant and tkwant packages. The system models:

1. A honeycomb lattice with zigzag boundary conditions
2. Time-dependent Gaussian bump deformation (strain)
3. Valley-resolved current calculations

Key features:
- Strain is modeled as a time-dependent Gaussian bump potential
- Valley currents (K and K' points) are calculated separately
- Uses MPI for parallel computation across multiple processors


The simulation:
1. Builds a graphene nanoribbon system with leads
2. Calculates time-dependent currents through the system
3. Stores results for K, K', charge, and valley currents

Output files:
- currentK0.npy: K-point current data
- currentK_prime0.npy: K'-point current data
- currentC0.npy: Charge current (sum)
- currentV0.npy: Valley current (difference)

"""

import kwant
import tkwant 
import numpy as np
from math import sqrt
import tinyarray
from numpy import pi, sqrt, arccos
from kwant.wraparound import wraparound, plot_2d_bands
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
    
    # Interface hopping identification
    interface_hoppings = [
        (s1, s2) if s1.pos[0] > s2.pos[0] else (s2, s1)
        for s1, s2 in system.hoppings()
        if ((s1 in system.leads[0].interface or s2 in system.leads[0].interface)
        and (s1.pos[0] != s2.pos[0]))
    
    # Current operator setup
    current_operator = kwant.operator.Current(
        finalized_system, 
        where=interface_hoppings,
        sum=True
    )
    
    # Initialize current storage
    currents = {
        'K': [],
        'K_prime': [],
        'charge': [],
        'valley': []
    }

    # Time evolution loop
    for time in times:
        # Load wavefunctions
        wavefunctions = {
            'K_left': np.load(f'Psi1_{time}.npy'),
            'K_prime_left': np.load(f'Psi0_{time}.npy'),
            'K_right': np.load(f'Psi4_{time}.npy'),
            'K_prime_right': np.load(f'Psi3_{time}.npy')
        }

        # Calculate currents
        params['time'] = time
        K_prime_current = (current_operator(wavefunctions['K_prime_left'], params) +
                          current_operator(wavefunctions['K_prime_right'], params))
        
        K_current = (current_operator(wavefunctions['K_left'], params) +
                    current_operator(wavefunctions['K_right'], params))
        
        # Store results
        currents['K'].append(K_current)
        currents['K_prime'].append(K_prime_current)
        currents['charge'].append(K_prime_current + K_current)
        currents['valley'].append(K_prime_current - K_current)

    # Save results
    if is_master():
        np.save('currentK0', currents['K'])
        np.save('currentK_prime0', currents['K_prime'])
        np.save('currentC0', currents['charge'])
        np.save('currentV0', currents['valley'])