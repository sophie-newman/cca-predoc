""" Run Synthesizer pipeline with different AGN model
e.g. mpirun -np 5 python agn_pipeline.py --nthreads 4 --subvol "0_0_0"

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

sam_dir = '/mnt/ceph/users/lperez/AGNmodelingSCSAM/sam_newAGNcode_forestmgmt_fidSAM'
grid_dir = '/mnt/ceph/users/snewman/grids'
grid_name = 'qsosed.hdf5'
grid_sps_name = 'bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35.hdf5'

# Load grids
grid_agn = Grid(grid_dir=grid_dir, grid_name=grid_name, ignore_lines=True)
grid_sps = Grid(grid_dir=grid_dir, grid_name=grid_sps_name, ignore_lines=True)

# Interpolate grids onto new, shared wavelength array
new_lams = grid_agn.lam
grid_sps.interp_spectra(new_lam=new_lams)

def get_single_galaxy(SFH, age_lst, Z_hist, bh_mass, bh_mdot, z, to_gal_prop_idx):
    """Create a particle-based galaxy from SC-SAM data."""

    # Initial mass, age, metallicity arrays for stars
    p_imass, p_age, p_Z = [], [], []
    for age_ind in range(len(age_lst)):
        if (SFH[age_ind] == 0).all():
            continue
        p_imass.append(SFH[age_ind])
        p_age.append(age_lst[age_ind])
        p_Z.append(Z_hist[age_ind])
    p_imass = np.array(p_imass) * 1e9  # Msun
    p_age = np.array(p_age) * 1e9      # yr
    p_Z = np.array(p_Z)

    stars = ParticleStars(initial_masses=p_imass * Msun,
                          ages=p_age * yr,
                          metallicities=p_Z)

    black_holes = BlackHoles(
        masses = np.array([bh_mass]) * 1e9 * Msun,
        accretion_rates = np.array([bh_mdot]) * Msun/yr
    )

    # Pick a random cosine inclination between 0.1 and 0.98
    cosine_inc = random.uniform(0.1, 0.98)

    # Convert to inclination in degrees
    inc_deg = math.acos(cosine_inc) * (180 / math.pi)

    black_holes.inclination = inc_deg * deg 

    gal = ParticleGalaxy(
        redshift = z,
        to_gal_prop_idx = to_gal_prop_idx,
        stars = stars, 
        black_holes = black_holes)

    return gal

def get_galaxies(subvol="0_0_0", N=None, comm=None):
    """Load galaxies for a given subvolume (or all subvolumes),
    distributed across MPI ranks."""

    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    sv = subvol

    with h5py.File(f'{sam_dir}/volume.hdf5', 'r') as file:

        # Load properties
        bh_mass_sub = file[f'{sv}/Galprop/GalpropMBH'][:] * 1e9
        bh_mdot_sub = file[f'{sv}/Galprop/GalpropMaccdot_bright'][:]

        sfh = file[f'{sv}/Histprop/HistpropSFH'][:]
        z_hist = file[f'{sv}/Histprop/HistpropZt'][:]

        to_gal_prop = file[f'{sv}/Linkprop/LinkproptoGalprop'][:]
        redshift = file[f'{sv}/Linkprop/LinkpropRedshift'][:]

        sfh_t_bins = file[f'{sv}/Header/SFH_tbins'][:]

        # Align galaxy properties using mapping
        bh_mass = bh_mass_sub[to_gal_prop]
        bh_mdot = bh_mdot_sub[to_gal_prop]

    # Apply BH mass cut (M_BH > 1e6)
    mass_cut = bh_mass > 1e6

    bh_mass = bh_mass[mass_cut]
    bh_mdot = bh_mdot[mass_cut]
    redshift = redshift[mass_cut]
    sfh = sfh[mass_cut]
    z_hist = z_hist[mass_cut]

    all_indices = np.arange(len(bh_mass))

    # Case 1: User requested a total of N galaxies
    if N:
        if rank == 0:
            sampled_indices = np.random.choice(all_indices, size=N, replace=False)
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
                          z=redshift[i],
                          to_gal_prop_idx=to_gal_prop[i])
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
    args = parser.parse_args()

    nthreads = args.nthreads
    N = args.N
    subvolume = args.subvol

    # MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load galaxies
    read_start = time.time()
    galaxies = get_galaxies(subvol=subvolume, N=N, comm=comm)
    read_end = time.time()
    print(f"Rank {rank}: Creating {len(galaxies)} galaxies took {read_end - read_start:.2f} s")

    # Set emission model
    model = emission_model()

    # Start Pipeline
    pipeline = Pipeline(emission_model=model, nthreads=nthreads, verbose=1, comm=comm)
    pipeline.add_galaxies(galaxies)
    pipeline.get_spectra()
    pipeline.get_observed_spectra(cosmo=cosmo)

    # Add galaxy info
    pipeline.add_analysis_func(lambda gal: gal.redshift, "Redshift")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.mass, "BlackHoleMass")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.accretion_rate, "AccretionRate")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.inclination, "InclinationDeg")
    pipeline.add_analysis_func(lambda gal: gal.to_gal_prop_idx, "GalPropIndex")

    pipeline.run()
    pipeline.write(f"/mnt/home/snewman/ceph/pipeline_results/pipeline_no_dust_{subvolume}_bhmasscut.hdf5", verbose=0)