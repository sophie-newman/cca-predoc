import argparse
import os
import time
import numpy as np
import h5py
from unyt import angstrom, Msun, yr
from astropy.cosmology import Planck15 as cosmo
from mpi4py import MPI
import multiprocessing as mp

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

# Define subvolumes
subvolumes = [
    "0_0_0",
    "0_0_1",
    "0_1_0",
    "0_1_1",
    "1_0_0",
    "1_0_1",
    "1_1_0",
    "1_1_1"
]

sam_dir = '/mnt/ceph/users/lperez/AGNmodelingSCSAM/sam_newAGNcode_forestmgmt_fidSAM'
grid_dir = '/mnt/ceph/users/snewman/grids'
grid_name = 'qsosed-test.hdf5'
grid_sps_name = 'bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35.hdf5'

# Load grids
grid = Grid(grid_dir=grid_dir, grid_name=grid_name, ignore_lines=True)
grid_sps = Grid(grid_dir=grid_dir, grid_name=grid_sps_name, ignore_lines=True)

# Interpolate grids onto new, shared wavelength array
new_lams = np.logspace(-2, 5, 10000) * angstrom
grid.interp_spectra(new_lam=new_lams)
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
        masses=np.array([bh_mass]) * 1e9 * Msun,
        accretion_rates=np.array([bh_mdot]) * Msun/yr,
        redshift=np.array([z])
    )

    gal = ParticleGalaxy(
        redshift = z,
        bh_mass = bh_mass,
        accretion_rate = bh_mdot,
        to_gal_prop_idx = to_gal_prop_idx,
        stars=stars, 
        black_holes=black_holes)

    return gal

def get_galaxies(subvol="0_0_0", N=None):
    """Load galaxies for a given subvolume (or all subvolumes)"""

    bh_mass, bh_mdot, redshift = [], [], []
    sfh, z_hist, sfh_t_bins = [], [], []
    sv = subvol

    with h5py.File(f'{sam_dir}/volume.hdf5', 'r') as file:

        bh_mass_sub = file[f'{sv}/Galprop/GalpropMBH'][:]
        bh_mdot_sub = file[f'{sv}/Galprop/GalpropMaccdot_bright'][:]
        bh_pos_sub = file[f'{sv}/Galprop/GalpropPos'][:]
        
        sfh.append(file[f'{sv}/Histprop/HistpropSFH'][:])
        z_hist.append(file[f'{sv}/Histprop/HistpropZt'][:])

        to_gal_prop = file[f'{sv}/Linkprop/LinkproptoGalprop'][:]
        redshift.append(file[f'{subvol}/Linkprop/LinkpropRedshift'][:])
        for i in to_gal_prop:
            bh_mass.append(bh_mass_sub[i])
            bh_mdot.append(bh_mdot_sub[i])

        sfh_t_bins = file[f'{sv}/Header/SFH_tbins'][:]
        sfh = np.vstack(sfh)
        z_hist = np.concatenate(z_hist)
        redshift = np.concatenate(redshift)

    indices = np.arange(len(bh_mass))
    if N:
        indices = np.random.choice(indices, size=N, replace=False)

    galaxies = [
        get_single_galaxy(SFH=sfh[i],
                          age_lst=sfh_t_bins,
                          Z_hist=z_hist[i],
                          bh_mass=bh_mass[i],
                          bh_mdot=bh_mdot[i],
                          z=redshift[i],
                          to_gal_prop_idx=to_gal_prop[i])
        for i in indices
    ]
    return galaxies

def emission_model():
    """Define combined emission model using loaded grids."""

    stellar_incident = StellarEmissionModel(
        "stellar_incident", grid=grid_sps, extract="incident", fesc=0.3
    )
    agn_incident = BlackHoleEmissionModel(
        "agn_incident", grid=grid, extract="incident", fesc=0.3
    )
    combined_emission = GalaxyEmissionModel(
        "total", combine=(stellar_incident, agn_incident)
    )
    return combined_emission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive synthetic observations for SC-SAM.")
    parser.add_argument("--nthreads", type=int, default=1)
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
    galaxies = get_galaxies(subvol=subvolume)
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
    pipeline.add_analysis_func(lambda gal: gal.bh_mass, "BlackHoleMass")
    pipeline.add_analysis_func(lambda gal: gal.accretion_rate, "AccretionRate")
    pipeline.add_analysis_func(lambda gal: gal.sfh_to_gal_prop, "GalPropIndex")

    pipeline.run()
    pipeline.write(f"/mnt/home/snewman/ceph/pipeline_results/pipeline_no_dust_{subvolume}.hdf5", verbose=0)
