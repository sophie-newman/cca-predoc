""" Run Synthesizer pipeline with different AGN model
e.g. mpirun -np 5 python pipeline_with_li08.py --nthreads 4 --subvol "0_0_0" --sample_index 0

Working well with 200 GB, 20 core job, 5 ranks, 4 threads.
"""

import argparse
import os
import time
import numpy as np
import h5py
from unyt import angstrom, Msun, yr, deg, kelvin
from astropy.cosmology import Planck15 as cosmo
from mpi4py import MPI
import multiprocessing as mp
import random
import math
from scipy.stats import qmc

# Synthesizer imports
from synthesizer import check_openmp
print('OpenMP enabled:', check_openmp())

from synthesizer.grid import Grid
from synthesizer.emission_models import (
    BlackHoleEmissionModel,
    GalaxyEmissionModel,
    StellarEmissionModel,
    AttenuatedEmission,
)

from synthesizer.emission_models.attenuation import ParametricLi08
from synthesizer.emission_models.dust.emission import Blackbody

from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy
from synthesizer.particle.stars import Stars as ParticleStars
from synthesizer.particle import BlackHoles
from synthesizer.pipeline import Pipeline  # , combine_files_virtual

# Define subvolumes
subvolumes = [
    "0_0_0",  #
    "0_0_1",  #
    "0_1_0",  #
    "0_1_1",  #
    "1_0_0",  #
    "1_0_1",  #
    "1_1_0",  #
    "1_1_1"
]

sam_dir = '/mnt/ceph/users/lperez/AGNmodelingSCSAM/sam_newAGNcode_forestmgmt_fidSAM'
grid_dir = '/mnt/ceph/users/snewman/grids'
grid_name = 'qsosed.hdf5'
grid_sps_name = 'bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35.hdf5'

# Define a single, desired output wavelength array (e.g., 10000 points from 10 A to 1,000,000 A)
desired_lams = np.logspace(0.1, 6, 100000) * angstrom

# Load grids
grid_agn = Grid(grid_dir=grid_dir, grid_name=grid_name, ignore_lines=True, new_lam=desired_lams)
grid_sps = Grid(grid_dir=grid_dir, grid_name=grid_sps_name, ignore_lines=True, new_lam=desired_lams)


def load_single_lhc_sample(filepath, index):
    """
    Load one Latin Hypercube sample (single row) by index from a .txt file.
    The dust parameter names are inferred from the header line.
    """
    # Read header (first line)
    with open(filepath, 'r') as f:
        header_line = f.readline().strip()
        # Split header by whitespace or tabs
        param_names = header_line.split()

    # Load numeric data, skipping the header line
    data = np.loadtxt(filepath, skiprows=1)

    # Validate index
    if index < 0 or index >= len(data):
        raise IndexError(f"Index {index} is out of range (file has {len(data)} samples).")

    # Build dictionary of parameter: value
    sample = {param_names[i]: float(data[index, i]) for i in range(len(param_names))}

    print(f"✅ Loaded sample #{index} from {os.path.basename(filepath)}")
    for k, v in sample.items():
        print(f"  {k}: {v:.4f}")

    return sample


def get_single_galaxy(SFH, age_lst, Z_hist, bh_mass, bh_mdot, z, to_gal_prop_idx, **dust_params):
    """Create a particle-based galaxy using Synthesizer using the SC-SAM data
    loaded in get_galaxies. We also need the LHS dust parameters."""

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
        masses=np.array([bh_mass]) * Msun,
        accretion_rates=np.array([bh_mdot]) * Msun / yr
    )

    # Pick a random cosine inclination between 0.1 and 0.98
    cosine_inc = random.uniform(0.1, 0.98)
    inc_deg = math.acos(cosine_inc) * (180 / math.pi)
    black_holes.inclination = inc_deg * deg

    gal = ParticleGalaxy(
        redshift=z,
        to_gal_prop_idx=to_gal_prop_idx,
        stars=stars,
        black_holes=black_holes
    )

    # Assign LHC parameters as attributes (e.g., gal.UV_slope)
    for k, v in dust_params.items():
        setattr(gal.stars, k, v)

    return gal


def get_galaxies(subvol="0_0_0", N=None, comm=None):
    """Load galaxies for a given subvolume (or all subvolumes),
    distributed across MPI ranks."""

    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1
    sv = subvol

    with h5py.File(f'{sam_dir}/volume.hdf5', 'r') as file:
        bh_mass_sub = file[f'{sv}/Galprop/GalpropMBH'][:] * 1e9
        bh_mdot_sub1 = file[f'{sv}/Galprop/GalpropMaccdot_bright'][:]
        bh_mdot_sub2 = file[f'{sv}/Galprop/GalpropMaccdot_radio'][:]
        bh_mdot_sub = bh_mdot_sub1

        sfh = file[f'{sv}/Histprop/HistpropSFH'][:]
        z_hist = file[f'{sv}/Histprop/HistpropZt'][:]
        to_gal_prop = file[f'{sv}/Linkprop/LinkproptoGalprop'][:]
        redshift = file[f'{sv}/Linkprop/LinkpropRedshift'][:]
        sfh_t_bins = file[f'{sv}/Header/SFH_tbins'][:]

        bh_mass = bh_mass_sub[to_gal_prop]
        bh_mdot = bh_mdot_sub[to_gal_prop]

    # Apply mask: keep only galaxies with bh_mdot > 0
    mask = bh_mdot > 0
    bh_mass = bh_mass[mask]
    bh_mdot = bh_mdot[mask]
    sfh = sfh[mask]
    z_hist = z_hist[mask]
    redshift = redshift[mask]
    to_gal_prop = to_gal_prop[mask]

    all_indices = np.arange(len(bh_mass))

    # Case 1: User requested a total of N galaxies
    if N:
        if rank == 0:
            sampled_indices = np.random.choice(all_indices, size=min(N, len(all_indices)), replace=False)
        else:
            sampled_indices = None
        sampled_indices = comm.bcast(sampled_indices, root=0)
    else:
        sampled_indices = all_indices

    # Retrieve the LHC dust parameter sample (only on rank 0)
    if rank == 0:
        lhc_file = args.lhc_file  
        sample_index = args.sample_index  
        dust_params = load_single_lhc_sample(lhc_file, sample_index)
    else:
        dust_params = None
    dust_params = comm.bcast(dust_params, root=0)

    # Split work across MPI ranks evenly
    my_indices = np.array_split(sampled_indices, size)[rank]
    print(f"Rank {rank}/{size}: processing {len(my_indices)} galaxies (total {len(sampled_indices)})")

    galaxies = []
    for gal_idx in my_indices:

        galaxy = get_single_galaxy(
            SFH=sfh[gal_idx],
            age_lst=sfh_t_bins,
            Z_hist=z_hist[gal_idx],
            bh_mass=bh_mass[gal_idx],
            bh_mdot=bh_mdot[gal_idx],
            z=redshift[gal_idx],
            to_gal_prop_idx=to_gal_prop[gal_idx],
            **dust_params
        )
        galaxies.append(galaxy)

    return galaxies


def emission_model():
    """Define combined emission model using loaded grids."""
    stellar_incident = StellarEmissionModel(
        "stellar_incident", grid=grid_sps, extract="incident", fesc=1.0
    )
    agn_incident = BlackHoleEmissionModel(
        "agn_incident", grid=grid_agn, extract="incident", fesc=1.0
    )

    # Get LHC dust parameters from galaxy attributes, loaded from LHC file
    li08_curve = ParametricLi08(
        UV_slope='UV_slope',
        OPT_NIR_slope=1.87,
        FUV_slope='FUV_slope',
        bump='bump',
        model='Custom'
    )

    stellar_attenuated = AttenuatedEmission(
        emitter="stellar",
        apply_to=stellar_incident,
        dust_curve=li08_curve,
        tau_v=0.5,
        label="stellar_attenuated",
    )

    combined_emission = GalaxyEmissionModel(
        "total", combine=(stellar_attenuated, agn_incident)
    )
    return combined_emission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive synthetic observations for SC-SAM.")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--subvol", type=str, default="0_0_0")
    parser.add_argument("--lhc_file", type=str, default="/mnt/home/snewman/ceph/lhc_samples/samples_5.txt")
    parser.add_argument("--sample_index", type=int, default=0)
    args = parser.parse_args()

    nthreads = args.nthreads
    N = args.N
    subvolume = args.subvol
    lhc_index = args.sample_index

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    read_start = time.time()
    galaxies = get_galaxies(subvol=subvolume, N=N, comm=comm)
    read_end = time.time()
    print(f"Rank {rank}: Creating {len(galaxies)} galaxies took {read_end - read_start:.2f} s")

    model = emission_model()

    # Load header names from the LHC file 
    if rank == 0:
        with open(args.lhc_file, 'r') as f:
            header_line = f.readline().strip()
            dust_param_names = header_line.split()
    else:
        dust_param_names = None

    # Broadcast to all ranks
    dust_param_names = comm.bcast(dust_param_names, root=0)

    pipeline = Pipeline(emission_model=model, nthreads=nthreads, verbose=1, comm=comm)
    pipeline.add_galaxies(galaxies)
    pipeline.get_spectra()
    pipeline.get_observed_spectra(cosmo=cosmo)

    pipeline.add_analysis_func(lambda gal: gal.redshift, "Redshift")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.mass, "BlackHoleMass")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.accretion_rate, "AccretionRate")
    pipeline.add_analysis_func(lambda gal: gal.black_holes.inclination, "InclinationDeg")
    pipeline.add_analysis_func(lambda gal: gal.to_gal_prop_idx, "GalPropIndex")

    for param in dust_param_names:
        pipeline.add_analysis_func(lambda gal, p=param: getattr(gal.stars, p), f'LHC_{param}')

    pipeline.run()
    pipeline.write(f"/mnt/home/snewman/ceph/pipeline_results/pipeline_li08_{subvolume}_lhcindex{lhc_index}.hdf5", verbose=0)