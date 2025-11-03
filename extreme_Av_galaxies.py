import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import h5py

mpl.rcParams.update({'font.size': 16})

# First let's get all of the local SAM galaxies to compare our most extreme galaxies to

sam_dir = '/mnt/ceph/users/lperez/AGNmodelingSCSAM/sam_newAGNcode_allzs_Sophie/h5files'
#sam_dir = '/mnt/ceph/users/lperez/CAMELS-SAM-phot/CV/CV_1/fid-sam_oldAGN/h5files'
sv = "0_0_0"

with h5py.File(f'{sam_dir}/volume.hdf5', 'r') as f:
    group = f[f"{sv}/Galprop"]

    # Load redshift first to make the mask
    redshift = group['GalpropRedshift'][:]
    redshift_mask = (redshift >= 0) & (redshift < 0.1)

    # Load only the subset you need using the mask
    stellar_mass = group['GalpropMstar'][:][redshift_mask] * 1e9 # Msolar
    metal_mass = group['GalpropZstar'][:][redshift_mask] * 1e9 # Msolar Zsolar
    disk_radius = group['GalpropRdisk'][:][redshift_mask] # kpc
    halo_radius = group['GalpropRhalo'][:][redshift_mask] # Mpc
    gas_mass = group['GalpropMcold'][:][redshift_mask] * 1e9 # Msolar
    redshift = redshift[redshift_mask]
    stellar_mass_merge = group['GalpropMstar_merge'][:][redshift_mask] * 1e9 # Msolar
    maccdot = group['GalpropMaccdot_bright'][:][redshift_mask]
    halo_mass = group['GalpropMhalo'][:][redshift_mask] * 1e9 # Msolar
    outflow_rate_mass = group['GalpropOutflowRate_Mass'][:][redshift_mask] # Msolar/yr
    outflow_rate_metal = group['GalpropOutflowRate_Metal'][:][redshift_mask] # Msolar/yr
    fric_radius = group['GalpropRfric'][:][redshift_mask] # kpc
    sfr = group['GalpropSFR'][:][redshift_mask] # Msolar/yr
    t_merger = group['GalpropTmerger'][:][redshift_mask] # Gyr
    t_merger_major = group['GalpropTmerger_major'][:][redshift_mask] # Gyr

# Compute derived properties
Z_gas = metal_mass / gas_mass # Zsolar

print('N galaxies:', len(Z_gas))

# Calculate Av for all galaxies with a fixed inclination
cosine_incs = 1
k_v = 3.4822e4  # cm^2/g
log_Av_values = np.log(4.4e-3) - np.log(cosine_incs) + np.log(k_v) + np.log(metal_mass / 1e10) - 2*np.log(disk_radius)
Av_values = np.exp(log_Av_values)

# ---------------------------------------------------------------------
# Gather data into a structured list for easy looping
# ---------------------------------------------------------------------
properties = [
    (disk_radius, r'$\rm R_{disk}$ (kpc)'),
    (halo_radius, r'$\rm R_{halo}$ (Mpc)'),
    (stellar_mass, r'$\rm M_{*}/M_{\odot}$'),
    (gas_mass, r'$\rm M_{gas}/M_{\odot}$'),
    (metal_mass, r'$\rm M_{metal}/M_{\odot}Z_{\odot}$'),
    (Z_gas, r'$\rm Z_{gas}/Z_{\odot}$'),
    (stellar_mass_merge, r'$\rm M_{*,merge}/M_{\odot}$'),
    (maccdot, r'$\rm \dot{M}$'),
    (halo_mass, r'$\rm M_{halo}/M_{\odot}$'),
    (outflow_rate_mass, r'Rate of outflowing gas mass ($\rm M_{\odot}/yr$)'),
    (outflow_rate_metal, r'Rate of outflowing metal mass ($\rm M_{\odot}/yr$)'),
    (fric_radius, r'Distance from halo center (kpc)'),
    (sfr, r'SFR averaged over 100 Myr ($\rm M_{\odot}/yr$)'),
    (t_merger, r'Time since last merger (Gyr)'),
    (t_merger_major, r'Time since last major merger (Gyr)'),
]

print("\n===== DATA VALIDITY CHECK =====")
for arr, label in properties:
    total = len(arr)
    n_zero = np.sum(arr == 0)
    n_nan = np.sum(~np.isfinite(arr))
    n_valid = np.sum((arr > 0) & np.isfinite(arr))
    print(f"{label:45s} → total: {total:6d}, zeros: {n_zero:6d}, NaNs: {n_nan:6d}, valid: {n_valid:6d}")
print("================================\n")

# ---------------------------------------------------------------------
# Select top 5 galaxies by Av and assign colors
# ---------------------------------------------------------------------
valid_mask = np.isfinite(Av_values) & (Av_values > 0)
Av_valid = Av_values[valid_mask]

# Get indices within the valid subset
sorted_idx_within_valid = np.argsort(Av_valid)

# Map back to the original indexing
valid_indices = np.where(valid_mask)[0]
top5_idx = valid_indices[sorted_idx_within_valid[-5:]]

colors = cmr.take_cmap_colors('cmr.tropical', 5, cmap_range=(0.15, 0.85))
print(Av_values[top5_idx])

# Print values which don't show as lines on histograms
print("Top 5 Av galaxy values for maccdot:", maccdot[top5_idx])
print("Top 5 Av galaxy values for fric_radius:", fric_radius[top5_idx])

# ---------------------------------------------------------------------
# Create subplots grid
# ---------------------------------------------------------------------
n_props = len(properties)
ncols = 5 
nrows = int(np.ceil(n_props / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 14))
axes = axes.flatten()

# ---------------------------------------------------------------------
# Plot each histogram
# ---------------------------------------------------------------------
for i, (data, xlabel) in enumerate(properties):
    ax = axes[i]
    mask = np.isfinite(data) & (data > 0)
    data_masked = data[mask]

    # Define log bins
    n_bins = 50
    log_min = np.log10(data_masked.min())
    log_max = np.log10(data_masked.max())
    bins = np.logspace(log_min, log_max, n_bins)

    # Plot histogram
    ax.hist(data_masked, bins=bins, alpha=0.7, color='lemonchiffon', edgecolor='black')

    # Plot top 5 galaxies (skip invalids, but show label if missing)
    for j, idx in enumerate(top5_idx):
        val = properties[i][0][idx]
        if np.isfinite(val) and val > 0:
            ax.axvline(val, color=colors[j], linestyle='--', linewidth=2.5,
                       label=f'#{j+1}: {val:.2e}')

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('N Galaxies', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=7, loc='best', frameon=False)

# Hide unused axes if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Increase vertical space between subplots
fig.tight_layout(h_pad=3.0, w_pad=2.0)  # ✅ extra space between rows & columns
plt.savefig('/mnt/home/snewman/extreme_Av_galaxies.png', dpi=300, bbox_inches='tight')
plt.show()




# ---------------------------------------------------------------------
# Plot halo_radius / disk_radius in a separate figure
# ---------------------------------------------------------------------
ratio = halo_radius / disk_radius  # halo_radius in Mpc, disk_radius in kpc
# Convert halo_radius to kpc to make the ratio dimensionless
ratio_kpc = (halo_radius * 1e3) / disk_radius  

fig_ratio, ax_ratio = plt.subplots(figsize=(7, 5))

mask = np.isfinite(ratio_kpc) & (ratio_kpc > 0)
data_masked = ratio_kpc[mask]

# Define log bins
n_bins = 50
log_min = np.log10(data_masked.min())
log_max = np.log10(data_masked.max())
bins = np.logspace(log_min, log_max, n_bins)

# Plot histogram
ax_ratio.hist(data_masked, bins=bins, alpha=0.7, color='lemonchiffon', edgecolor='black')

# Mark top 5 Av galaxies
for j, idx in enumerate(top5_idx):
    val = ratio_kpc[idx]
    if np.isfinite(val) and val > 0:
        ax_ratio.axvline(val, color=colors[j], linestyle='--', linewidth=2.5,
                         label=f'#{j+1}: {val:.2e}')

# Formatting
ax_ratio.set_xscale('log')
ax_ratio.set_yscale('log')
ax_ratio.set_xlabel(r'$\rm R_{halo} / R_{disk}$', fontsize=12)
ax_ratio.set_ylabel('N Galaxies', fontsize=12)
ax_ratio.tick_params(axis='both', which='major', labelsize=10)
ax_ratio.legend(fontsize=9, loc='best', frameon=False)
fig_ratio.tight_layout()
plt.savefig('/mnt/home/snewman/halo_to_disk_ratio.png', dpi=300, bbox_inches='tight')
plt.show()