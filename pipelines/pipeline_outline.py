"""
Hybrid MPI + threading version using synthesizer.Pipeline
to process SCSAM galaxies and compute spectra/photometry
for one or all subvolumes.

e.g. mpirun -np 4 python scsam_pipeline.py --nthreads 2 --N 50
This will run with 4 MPI ranks, each using 2 threads, processing up to 50 galaxies
per subvolume (200 total if all ranks get 50).

e.g. mpirun -np 4 python scsam_pipeline.py --nthreads 2 --N 50 --subvol 0_0_0
This will only process subvolume "0_0_0".
"""

import argparse
import os
import time
import numpy as np
import h5py
from unyt import angstrom, Msun, yr
from astropy.cosmology import Planck15 as cosmo

# MPI setup
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
except Exception:
    comm, rank, size = None, 0, 1

# Synthesizer imports
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

def get_single_galaxy():

    # load properties

    # gal = Galaxy(...)

    return gal



def get_galaxies():


# Distribute galaxies


# Get all the galaxies using multiprocessing


# Return galaxies



def emission_model():

    # Set up emission model

    return emission_model


# Define the subvolume tags

subvolumes = [""]


if __name__ == "__main__":

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Derive synthetic observations for SC-SAM."
    )

    # Add other arguments here

    # Get grids

    # model = get_emission_model()

    # sam_path = 

    # Print n CPUs for reference

    # Get MPI info
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get the galaxies
    read_start = time.time()
    galaxies = get_flares_galaxies(...)

    read_end = time.time()
    _print(
        f"Creating {len(galaxies)} galaxies took "
        f"{read_end - read_start:.2f} seconds."
    )

    # Start Pipeline object
    pipeline = Pipeline(
        emission_model=model,
        #instruments=instruments,
        nthreads=nthreads,
        verbose=1,
        comm=mpi.COMM_WORLD,
    )

    pipeline.add_galaxies(galaxies)
    pipeline.get_spectra()
    pipeline.get_observed_spectra(cosmo=cosmo)

    # Add galaxy info
    # Replace below with actual properties that we have from SC-SAM
    pipeline.add_analysis_func(lambda gal: gal.region, "Region")
    pipeline.add_analysis_func(lambda gal: gal.grp_id, "GroupID")
    pipeline.add_analysis_func(lambda gal: gal.subgrp_id, "SubGroupID")
    pipeline.add_analysis_func(lambda gal: gal.master_index, "MasterRegionIndex")
    pipeline.add_analysis_func(lambda gal: gal.redshift, "Redshift")

    pipeline.run()
    pipeline.write(f"/mnt/home/snewman/ceph/pipeline_results/pipeline_no_dust_{subvolume}.hdf5", verbose=0)