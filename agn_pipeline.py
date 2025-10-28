#!/usr/bin/env python3
"""
Run Synthesizer pipeline with different AGN model over many subvolumes efficiently.

Example:
  mpirun -np 20 python agn_pipeline.py --nthreads 4 --snap 67 --all

Behavior:
  - By default runs a single subvolume (old behavior).
  - With --all it will distribute work across MPI ranks in groups so multiple
    subvolumes can be processed in parallel while preserving intra-subvolume MPI.
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
from synthesizer.pipeline import Pipeline

# Define subvolumes (master list)
SUBVOLUMES = [
    "0_0_0",
    "0_0_1",
    "0_1_0",
    "0_1_1",
    "1_0_0",
    "1_0_1",
    "1_1_0",
    "1_1_1"
]

grid_dir = '/mnt/ceph/users/snewman/grids'
grid_name = 'qsosed.hdf5'
grid_sps_name = 'bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35.hdf5'

# Define a single, desired output wavelength array (e.g., 10000 points from 10 A to 1,000,000 A)
desired_lams = np.logspace(0.1, 6, 10000) * angstrom

# Load grids once (shared by all groups/ranks)
grid_agn = Grid(grid_dir=grid_dir, grid_name=grid_name, ignore_lines=True, new_lam=desired_lams)
grid_sps = Grid(grid_dir=grid_dir, grid_name=grid_sps_name, ignore_lines=True, new_lam=desired_lams)

def get_sfh_data(filename):
    """
    Load an SCSAM sfhist_*.dat file containing one or more galaxies.
    See docstring in original script for format details.
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

    # Return arrays and the first redshift value (original code used redshifts[0])
    return age_values, sfh_arr, Z_arr, redshifts[0]

def get_single_galaxy(SFH, age_lst, Z_hist, z, bh_mass, bh_mdot, to_galprop_idx):
    """Create a particle-based galaxy using Synthesizer using the SC-SAM data"""
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
    distributing the galaxies across MPI ranks (using comm).
    Returns a list of ParticleGalaxy objects local to this rank/comm.
    """
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    # Load SFH data
    sfh_t_bins, sfh, z_hist, redshift = get_sfh_data(sfh_file)

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

    # Find galprop rows with redshifts matching SFH redshift
    z_sfh = float(redshift)  # ensure scalar, not array
    to_galprop = np.where(np.isclose(df_galprop["redshift"].to_numpy(), z_sfh, atol=1e-4))[0]
    
    if len(to_galprop) == 0:
        raise ValueError(f"No matching redshifts found between {sfh_file} and {galprop_file}")
    
    print('N SFH gals:', len(sfh))
    print('N galprop gals:', len(df_galprop["mBH"].to_numpy()))
    
    bh_mass = df_galprop["mBH"].to_numpy()[to_galprop]
    bh_mdot = df_galprop["maccdot"].to_numpy()[to_galprop]

    print('N aligned gals:', len(bh_mdot))

    # Apply mask: keep only galaxies with bh_mdot > 0
    mask = bh_mdot > 0
    bh_mdot = bh_mdot[mask]
    bh_mass = bh_mass[mask]
    sfh = sfh[mask]
    z_hist = z_hist[mask]
    to_galprop = to_galprop[mask]

    all_indices = np.arange(len(bh_mass))

    # Case 1: Request a total of N galaxies (random sample)
    if N:
        if rank == 0:
            sampled_indices = np.random.choice(all_indices, size=min(N, len(all_indices)), replace=False)
        else:
            sampled_indices = None
        indices = comm.bcast(sampled_indices, root=0)

    # Case 2: Default (no N) → take all galaxies
    else:
        indices = all_indices

    # Split indices evenly across MPI ranks of this communicator
    my_indices = np.array_split(indices, size)[rank]

    print(f"[comm {comm.rank if comm else 0}/{comm.size if comm else 1}] processing {len(my_indices)} galaxies (total {len(indices)})")

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

def process_subvolume(subvol, snap, nthreads, N, comm_sub, sam_dir, out_dir):
    """
    Process a single subvolume using communicator comm_sub (which may be MPI.COMM_NULL
    if single-rank/no-parallelism).
    """
    rank = comm_sub.Get_rank() if comm_sub else 0
    size = comm_sub.Get_size() if comm_sub else 1

    sfh_file = f'{sam_dir}/{subvol}/sfhist_{snap}-{snap}.dat'
    galprop_file = f'{sam_dir}/{subvol}/galprop_{snap}-{snap}.dat'

    # Basic file checks on rank 0 of the communicator (if communicators exist)
    if (comm_sub is None) or (rank == 0):
        if not os.path.exists(sfh_file):
            raise FileNotFoundError(f"SFH file not found: {sfh_file}")
        if not os.path.exists(galprop_file):
            raise FileNotFoundError(f"Galprop file not found: {galprop_file}")

    comm_sub.Barrier() if comm_sub else None

    # Load galaxies (this function uses the comm_sub to split work)
    read_start = time.time()
    galaxies = get_galaxies(sfh_file=sfh_file, galprop_file=galprop_file, N=N, comm=comm_sub)
    read_end = time.time()
    if comm_sub:
        print(f"[subvol {subvol}] comm {comm_sub.Get_rank()}/{comm_sub.Get_size()}: Creating {len(galaxies)} galaxies took {read_end - read_start:.2f} s")
    else:
        print(f"[subvol {subvol}] single-rank: Created {len(galaxies)} galaxies in {read_end - read_start:.2f} s")

    # Build emission model and pipeline (one per subvolume)
    model = emission_model()
    pipeline = Pipeline(emission_model=model, nthreads=nthreads, verbose=1, comm=comm_sub)
    pipeline.add_galaxies(galaxies)
    pipeline.get_spectra()

    pipeline.add_analysis_func(lambda gal: gal.redshift, "Redshift")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.mass, "BlackHoleMass")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.accretion_rate, "AccretionRate")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.inclination, "InclinationDeg")
    pipeline.add_analysis_func(lambda gal: gal.to_galprop_idx, "GalPropIndex")

    pipeline.run()

    out_fname = os.path.join(out_dir, f"pipeline_agn_snap{snap}_subvol_{subvol}.hdf5")

    if comm_sub:
        pipeline.write(out_fname, verbose=0)
        print(f"[subvol {subvol}] rank finished writing")
    else:
        pipeline.write(out_fname, verbose=0)
        print(f"[subvol {subvol}] finished writing")


def main():
    parser = argparse.ArgumentParser(description="Derive synthetic observations for SC-SAM across subvolumes.")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--N", type=int, default=None, help="Total N of galaxies to process (random sample).")
    parser.add_argument("--subvol", type=str, default=None, help="Single subvolume to run (e.g. '0_0_0').")
    parser.add_argument("--snap", type=str, default="67")
    parser.add_argument("--all", action="store_true", help="Process all subvolumes from the SUBVOLUMES list.")
    parser.add_argument("--outdir", type=str, default="/mnt/home/snewman/ceph/pipeline_results", help="Output directory for pipeline results.")
    parser.add_argument("--samdir", type=str, default='/mnt/ceph/users/lperez/AGNmodelingSCSAM/sam_newAGNcode_multizs_Sophie', help="Base SAM directory containing subvolume folders.")
    args = parser.parse_args()

    nthreads = args.nthreads
    N = args.N
    snap = args.snap
    out_dir = args.outdir
    sam_dir = args.samdir

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # choose subvolumes to process
    if args.subvol and not args.all:
        target_subvols = [args.subvol]
    elif args.all:
        target_subvols = SUBVOLUMES.copy()
    else:
        # default fallback: if none provided, process first subvol (preserves backward compatibility)
        target_subvols = [SUBVOLUMES[0]]

    n_sub = len(target_subvols)

    # Determine number of groups (one group per concurrent subvolume up to available ranks)
    groups = min(n_sub, size)
    if groups < 1:
        groups = 1

    # Split ranks into contiguous groups
    rank_groups = np.array_split(np.arange(size), groups)
    my_group_id = None
    for gid, ranks in enumerate(rank_groups):
        if rank in ranks:
            my_group_id = gid
            break

    # Split subvolumes across groups (each group gets one or more subvolumes)
    subvol_groups = np.array_split(target_subvols, groups)
    # Determine if this group actually has any subvolumes
    my_subvols = list(subvol_groups[my_group_id]) if my_group_id is not None else []

    # If my_subvols is empty then this group should not participate (set color undefined)
    if len(my_subvols) == 0:
        # create no communicator for work
        comm_sub = None
        print(f"Rank {rank}: assigned to group {my_group_id} but no subvolumes -> idle for pipeline work.")
    else:
        # Create an intra-communicator for this group
        color = my_group_id
        comm_sub = comm.Split(color=color, key=rank)
        print(f"Rank {rank}: joined comm_sub (group {my_group_id}) with size {comm_sub.Get_size()} and will process subvolumes {my_subvols}")

    # Ensure output directory exists (only do on MPI rank 0 of WORLD)
    if rank == 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()

    # Each group processes its assigned subvolumes sequentially using comm_sub
    if comm_sub is None:
        # This rank has nothing to do (no subvolumes in its group)
        pass
    else:
        for subvol in my_subvols:
            try:
                process_subvolume(subvol=subvol, snap=snap, nthreads=nthreads, N=N, comm_sub=comm_sub, sam_dir=sam_dir, out_dir=out_dir)
            except Exception as e:
                # Report error to world rank 0 and continue (avoid silent termination on one rank)
                print(f"[group {my_group_id} rank {comm_sub.Get_rank()}] Error processing subvol {subvol}: {e}")
            # small barrier between subvolumes to synchronize group ranks
            comm_sub.Barrier()

        # Free the communicator
        comm_sub.Free()

    # Final barrier and exit
    comm.Barrier()
    if rank == 0:
        print("All subvolume processing complete.")

if __name__ == "__main__":
    main()
