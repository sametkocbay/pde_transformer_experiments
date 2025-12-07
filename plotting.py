import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. Find the HDF5 file automatically
output_folder = Path("dataset/sf_64x128_reynolds_2_00e+03_schmidt_1_00e+00_width_3_50e-01_nshear_2_nblobs_2") # Change to your folder name
files = list(output_folder.glob("*.h5"))

if not files:
    print("No .h5 files found. Check your folder path.")
    exit()

filename = files[1]
print(f"Visualizing: {filename}")

# 2. Open file and load data
with h5py.File(filename, "r") as f:
    # Available fields: 'vorticity', 'tracer', 'pressure', 'shear_velocity'
    # Shape is usually (Time, X, Z)
    vorticity = f['tasks']['vorticity'][:]
    tracer = f['tasks']['tracer'][:]
    time_points = f['scales']['sim_time'][:]

print(len(vorticity))
# 3. Plot the last time step
for t_index in [0,10,20,30,40,len(vorticity)-1]:
  # Last frame
    print(f"Plotting time: {time_points[t_index]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Vorticity
    im1 = axes[0].imshow(vorticity[t_index].T, cmap='RdBu_r', origin='lower')
    axes[0].set_title(f"Vorticity (t={time_points[t_index]:.2f})")
    plt.colorbar(im1, ax=axes[0])

    # Plot Tracer
    im2 = axes[1].imshow(tracer[t_index].T, cmap='viridis', origin='lower')
    axes[1].set_title(f"Tracer (t={time_points[t_index]:.2f})")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()