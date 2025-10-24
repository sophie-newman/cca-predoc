""" Run Synthesizer pipeline with different AGN model
e.g. mpirun -np 5 python agn_pipeline.py --nthreads 4 --subvol "0_0_0" --snap 67

Working well with 200 GB, 20 core job, 5 ranks, 4 threads.
"""

import argparse
import os
import time
import numpy as np
import h5py
from unyt import angstrom, Msun, yr, deg
from astropy.cosmology import Planck15 as cosmo
from mpi4py import MPI
import multiprocessing as mp
import random
import math
import pandas as pd

# Synthesizer imports
from synthesizer import check_openmp
print('OpenMP enabled:', check_openmp() )

from synthesizer.grid import Grid
from synthesizer.emission_models import (
    BlackHoleEmissionModel,
    GalaxyEmissionModel,
    StellarEmissionModel,
)
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy
from synthesizer.particle.stars import Stars as ParticleStars
from synthesizer.particle import BlackHoles
from synthesizer.pipeline import Pipeline #, combine_files_virtual

# Define subvolumes
subvolumes = [
    "0_0_0", #
    "0_0_1", #
    "0_1_0", #
    "0_1_1", #
    "1_0_0", #
    "1_0_1", #
    "1_1_0", #
    "1_1_1"
]

grid_dir = '/mnt/ceph/users/snewman/grids'
grid_name = 'qsosed.hdf5'
grid_sps_name = 'bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35.hdf5'

# Define a single, desired output wavelength array (e.g., 10000 points from 10 A to 1,000,000 A)
desired_lams = np.logspace(0.1, 6, 10000) * angstrom

# Load grids
grid_agn = Grid(grid_dir=grid_dir, grid_name=grid_name, ignore_lines=True, new_lam=desired_lams)
grid_sps = Grid(grid_dir=grid_dir, grid_name=grid_sps_name, ignore_lines=True, new_lam=desired_lams)

def get_sfh_data(filename):
    """
    Load an SCSAM sfhist_*.dat file containing one or more galaxies.

    File structure example:
        0.3 0.6711              <-- cosmology (Omega_m, h)
        1405                    <-- number of age bins
        <1405 age bin values>   <-- in Gyr
        # 54 ... 0.227446       <-- start of galaxy block (redshift = 0.227446)
        <1405 lines of "SFH  Z"> per galaxy
         
    Uses 2D NumPy arrays for fast access.

    Returns:
        age_lst   : np.ndarray of age bins (same for all galaxies)
        sfh_arr   : np.ndarray, shape (n_gal, n_age_bins)
        Z_arr     : np.ndarray, shape (n_gal, n_age_bins)
        redshifts : np.ndarray, shape (n_gal,)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Cosmology & number of age bins
    n_age_bins = int(lines[1].strip())

    # Read age bins
    age_values = []
    line_idx = 2
    while len(age_values) < n_age_bins:
        tokens = lines[line_idx].strip().split()
        age_values.extend(map(float, tokens))
        line_idx += 1
    age_values = np.array(age_values[:n_age_bins])

    # Count galaxies
    n_gal = sum(1 for line in lines if line.startswith("#"))

    # Preallocate arrays
    sfh_arr = np.zeros((n_gal, n_age_bins), dtype=float)
    Z_arr = np.zeros((n_gal, n_age_bins), dtype=float)
    redshifts = np.zeros(n_gal, dtype=float)

    # Read galaxy blocks
    gal_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if line.startswith("#"):
            redshifts[gal_idx] = float(line.split()[-1])
            for j in range(n_age_bins):
                vals = lines[line_idx + 1 + j].strip().split()
                if len(vals) >= 2:
                    sfh_arr[gal_idx, j] = float(vals[0])
                    Z_arr[gal_idx, j] = float(vals[1])
            line_idx += n_age_bins + 1
            gal_idx += 1
        else:
            line_idx += 1

    return age_values, sfh_arr, Z_arr, redshifts[0]

def get_single_galaxy(SFH, age_lst, Z_hist, z, bh_mass, bh_mdot, to_galprop_idx):
    """Create a particle-based galaxy using Synthesizer using the SC-SAM data
    loaded in get_galaxies."""

    stars = ParticleStars(
        initial_masses=SFH * 1e9 * Msun,
        ages=age_lst * 1e9 * yr,
        metallicities=Z_hist
    )

    black_holes = BlackHoles(
        masses = np.array([bh_mass]) * Msun,
        accretion_rates = np.array([bh_mdot]) * Msun/yr
    )

    # Pick a random cosine inclination between 0.1 and 0.98
    cosine_inc = random.uniform(0.1, 0.98)

    # Convert to inclination in degrees
    inc_deg = math.acos(cosine_inc) * (180 / math.pi)

    black_holes.inclination = inc_deg * deg 

    gal = ParticleGalaxy(
        redshift = z,
        stars = stars, 
        black_holes = black_holes,
        to_galprop_idx = to_galprop_idx)

    return gal

def get_galaxies(sfh_file, galprop_file, N=None, comm=None):
    """
    Load galaxies using SFH .dat and corresponding galprop .dat for non-SFH properties,
    distributing the galaxies across MPI ranks.
    Returns a list of ParticleGalaxy objects.
    """

    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    # Load SFH data
    sfh_t_bins, sfh, z_hist, redshift = get_sfh_data(sfh_file)
    n_gal = sfh.shape[0]

    # Load galprop data
    galprop_cols = [
        "halo_index", "birthhaloid", "roothaloid", "redshift", "sat_type",
        "mhalo", "m_strip", "rhalo", "mstar", "mbulge", "mstar_merge",
        "v_disk", "sigma_bulge", "r_disk", "r_bulge", "mcold", "mHI", "mH2",
        "mHII", "Metal_star", "Metal_cold", "sfr", "sfrave20myr", "sfrave100myr",
        "sfrave1gyr", "mass_outflow_rate", "metal_outflow_rate", "mBH",
        "maccdot", "maccdot_radio", "tmerge", "tmajmerge", "mu_merge",
        "t_sat", "r_fric", "x_position", "y_position", "z_position", "vx", "vy", "vz"
    ]
    df_galprop = pd.read_csv(galprop_file, comment='#', delim_whitespace=True, names=galprop_cols)

    # n galprop rows > n sfh rows
    # For each array in sfh_arr we want the corresponding data in galprop
    # Get indices in galprop sfh_to_galprop where redshift_sfh == redshift_prop
    # Use these indices to get bBH, maccdot

    # Find galprop rows with redshifts matching SFH redshift
    # (using a small tolerance for float comparisons)
    tol = 1e-4
    to_galprop = np.where(
        np.isclose(df_galprop["redshift"].to_numpy(), redshift, atol=tol)
    )[0]
    
    # Safety check
    if len(to_galprop) == 0:
        raise ValueError(f"No matching redshifts found between {sfh_file} and {galprop_file}")
    
    bh_mass = df_galprop["mBH"][to_galprop]
    bh_mdot = df_galprop["maccdot"][to_galprop]

    all_indices = np.arange(n_gal)

    # Case 1: Request a total of N galaxies
    if N:
        if rank == 0:
            sampled_indices = np.random.choice(all_indices, size=min(N, len(all_indices)), replace=False)
        else:
            sampled_indices = None
        indices = comm.bcast(sampled_indices, root=0)

    # Case 2: Default (no N) → take all galaxies
    else:
        indices = all_indices

    # Split indices evenly across MPI ranks
    my_indices = np.array_split(indices, size)[rank]

    # Debug print
    print(f"Rank {rank}/{size}: processing {len(my_indices)} galaxies "
          f"(total {len(indices)})")

    galaxies = [
        get_single_galaxy(SFH=sfh[i],
                          age_lst=sfh_t_bins,
                          Z_hist=z_hist[i],
                          bh_mass=bh_mass[i],
                          bh_mdot=bh_mdot[i],
                          z=redshift,
                          to_galprop_idx=to_galprop[i])
        for i in my_indices
    ]

    return galaxies

def emission_model():
    """Define combined emission model using loaded grids."""

    stellar_incident = StellarEmissionModel(
        "stellar_incident", grid=grid_sps, extract="incident", fesc=1.0
    )
    agn_incident = BlackHoleEmissionModel(
        "agn_incident", grid=grid_agn, extract="incident", fesc=1.0
    )
    combined_emission = GalaxyEmissionModel(
        "total", combine=(stellar_incident, agn_incident)
    )
    return combined_emission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive synthetic observations for SC-SAM.")
    parser.add_argument("--nthreads", type=int, default=1)
    # Total N of galaxies to process, randomly chosen
    parser.add_argument("--N", type=int, default=None) 
    parser.add_argument("--subvol", type=str, default="0_0_0")
    parser.add_argument("--snap", type=str, default=67)
    args = parser.parse_args()

    nthreads = args.nthreads
    N = args.N
    subvolume = args.subvol
    snap = args.snap

    sam_dir = '/mnt/ceph/users/lperez/AGNmodelingSCSAM/sam_newAGNcode_multizs_Sophie'
    sfh_file = f'{sam_dir}/{subvolume}/sfhist_{snap}-{snap}.dat'
    galprop_file = f'{sam_dir}/{subvolume}/galprop_{snap}-{snap}.dat'

    # MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load galaxies
    read_start = time.time()
    galaxies = get_galaxies(sfh_file=sfh_file, galprop_file=galprop_file, N=N, comm=comm)
    read_end = time.time()
    print(f"Rank {rank}: Creating {len(galaxies)} galaxies took {read_end - read_start:.2f} s")

    # Set emission model
    model = emission_model()

    # Start Pipeline
    pipeline = Pipeline(emission_model=model, nthreads=nthreads, verbose=1, comm=comm)
    pipeline.add_galaxies(galaxies)
    pipeline.get_spectra()
    #pipeline.get_observed_spectra(cosmo=cosmo)

    # Add galaxy info
    pipeline.add_analysis_func(lambda gal: gal.redshift, "Redshift")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.mass, "BlackHoleMass")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.accretion_rate, "AccretionRate")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.inclination, "InclinationDeg")
    pipeline.add_analysis_func(lambda gal: gal.to_galprop_idx, "GalPropIndex")

    pipeline.run()
    pipeline.write(f"/mnt/home/snewman/ceph/pipeline_results/pipeline_agn_snap{snap}.hdf5", verbose=0)