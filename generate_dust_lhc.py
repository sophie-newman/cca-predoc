import numpy as np
from scipy.stats import qmc

# Dust priors
LHC_RANGES = {
    'UV_slope': [0, 50.0], # c1
    'FUV_slope': [-1, 75], # c3
    'bump': [0, 0.06], # c4
}

DUST_PARAMS = list(LHC_RANGES.keys())
DUST_LIMITS = np.array(list(LHC_RANGES.values()))

def generate_lhc_li08_samples(n_samples):
    """Generate Latin Hypercube samples for Li & Draine (2008) dust parameters."""
    sampler = qmc.LatinHypercube(d=len(DUST_PARAMS))
    lhs_samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(lhs_samples, DUST_LIMITS[:, 0], DUST_LIMITS[:, 1])
    samples = {param: scaled_samples[:, i] for i, param in enumerate(DUST_PARAMS)}
    return samples


def save_samples_to_txt(samples, file_loc):
    """Save the generated samples to a .txt file."""
    header = "\t".join(DUST_PARAMS)
    data = np.column_stack([samples[param] for param in DUST_PARAMS])
    np.savetxt(file_loc, data, header=header, comments='', fmt='%.6f')
    print(f"✅ Saved {data.shape[0]} samples to '{file_loc}'.")


if __name__ == "__main__":
    n_samples = 5
    samples = generate_lhc_li08_samples(n_samples)

    output_dir = '/mnt/home/snewman/ceph/lhc_samples'
    # Get name by rounding n_samples to nearest integer
    output_name = f"samples_{int(round(n_samples))}.txt"
    output_path = f"{output_dir}/{output_name}"

    # Save to file
    save_samples_to_txt(samples, output_path)
