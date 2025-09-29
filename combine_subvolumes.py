"""After running the pipeline with multiple ranks and/or subvolumes,
Use this code to combine the results into a single file."""

import os
import re
import h5py
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# CONFIG
# -------------------------
indir = "/mnt/home/snewman/ceph/pipeline_results"
outdir = os.path.join(indir, "combined")
os.makedirs(outdir, exist_ok=True)

pattern = re.compile(r"pipeline_no_dust_(\d+)_(\d+)_(\d+)_(\d+)\.hdf5")

# Dataset to check for consistency
reference_dataset = "Galaxies/AccretionRate"

# -------------------------
# Collect files by subvolume
# -------------------------
subvolumes = defaultdict(list)
for fname in os.listdir(indir):
    match = pattern.match(fname)
    if match:
        subvol = "_".join(match.groups()[:3])
        subvolumes[subvol].append(os.path.join(indir, fname))

print(f"Found {len(subvolumes)} subvolumes.")

# -------------------------
# Helper functions
# -------------------------
def get_dataset_paths(h5file):
    paths = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)
    with h5py.File(h5file, "r") as f:
        f.visititems(visitor)
    return set(paths)


def append_dataset(src_dset, dst_dset, chunk_size=10000):
    oldsize = dst_dset.shape[0]
    newsize = oldsize + src_dset.shape[0]
    dst_dset.resize((newsize,) + dst_dset.shape[1:])

    for start in range(0, src_dset.shape[0], chunk_size):
        end = min(start + chunk_size, src_dset.shape[0])
        dst_dset[oldsize + start : oldsize + end, ...] = src_dset[start:end, ...]


def merge_group(fin_group, fout_group, chunk_size=10000):
    for key, item in fin_group.items():
        if isinstance(item, h5py.Group):
            if key not in fout_group:
                fout_group.create_group(key)
            merge_group(item, fout_group[key], chunk_size)
        elif isinstance(item, h5py.Dataset):
            if key not in fout_group:
                maxshape = (None,) + item.shape[1:]
                fout_group.create_dataset(
                    key,
                    shape=(0,) + item.shape[1:],
                    maxshape=maxshape,
                    dtype=item.dtype,
                    chunks=True
                )
            append_dataset(item, fout_group[key], chunk_size=chunk_size)


def combine_files(infiles, outfile, check_dataset, chunk_size=10000):
    all_paths = set()
    for f in infiles:
        all_paths |= get_dataset_paths(f)

    expected_len = 0
    for f in infiles:
        with h5py.File(f, "r") as fin:
            if check_dataset in fin:
                expected_len += fin[check_dataset].shape[0]

    with h5py.File(outfile, "w") as fout:
        for f in infiles:
            with h5py.File(f, "r") as fin:
                merge_group(fin, fout, chunk_size=chunk_size)

    output_paths = get_dataset_paths(outfile)
    missing = all_paths - output_paths
    if missing:
        raise ValueError(f"Missing datasets in {outfile}: {missing}")

    with h5py.File(outfile, "r") as fout:
        if check_dataset not in fout:
            raise ValueError(f"Reference dataset {check_dataset} missing in {outfile}")
        out_len = fout[check_dataset].shape[0]
        if out_len != expected_len:
            raise ValueError(
                f"Dataset {check_dataset} length mismatch in {outfile}: "
                f"expected {expected_len}, got {out_len}"
            )
    print(f"✅ {outfile}: {check_dataset} length = {expected_len} (consistent)")

# -------------------------
# Parallel merging
# -------------------------
def process_subvolume(subvol, files):
    files.sort(key=lambda f: int(pattern.match(os.path.basename(f)).group(4)))
    outfile = os.path.join(outdir, f"pipeline_no_dust_{subvol}.hdf5")
    print(f"Combining {len(files)} ranks -> {outfile}")
    combine_files(files, outfile, reference_dataset, chunk_size=10000)
    return outfile

test_subvol = "0_0_0"  # None or "0_0_0" for testing a single subvolume

if test_subvol:
    process_subvolume(test_subvol, subvolumes[test_subvol])
else:
    # Use ThreadPoolExecutor to process subvolumes in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_subvolume, sv, files) for sv, files in subvolumes.items()]
        for future in as_completed(futures):
            future.result()  # raises exceptions if any
