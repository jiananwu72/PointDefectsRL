from itertools import product
import ase
import ase.io
import numpy as np
import matplotlib.pyplot as plt
import abtem
from tqdm import tqdm

import sys
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from util.crop import Crop

batch = 'FeLu'

# Load structure; cif_path will be edited to load from command line
plain_cif_path = project_root / 'data' / 'structures' / 'LFO_Orth2.cif'
defect_cif_path = project_root / 'data' / 'structures' / f'LFO_Orth_{batch}.cif'
layer_plain = ase.io.read(plain_cif_path)
layer_defect = ase.io.read(defect_cif_path)
save_dir = project_root / 'data' / 'simulations' / f'{batch}'
save_dir.mkdir(parents=True, exist_ok=True)

# Set simulation on GPU
abtem.config.set({"device": "gpu", "fft": "fftw"})

# layers_list = [5, 10, 15, 20, 25]
# layers_list = [30, 35]
layers_list = [40]
# layers_list = [45]
# layers_list = [50]

energy = 100e3         # eV
Cs = 5.593e4           # in Angstroms
semiangle_cutoff = 30  # in milliradians
defocus = 17.14        # in Angstroms

# layers_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# energy = 100e3         # eV
# Cs = 5.593e4           # in Angstroms
# semiangle_cutoff = 30  # in milliradians
# defocus = 17.14        # in Angstroms

def run_simulation(layer_plain, layer_defect, current_layer_index, total_layers):
    """
    Runs a single abtem simulation with the defect placed at current_layer_index.
    """
    print(f"\nRunning simulation with defect on layer={current_layer_index}...")

    n_top = current_layer_index
    n_bottom = total_layers - current_layer_index - 1

    # Use the height of a single pristine unit cell as the step size
    h = layer_plain.cell[2, 2]
    atoms_top = layer_plain * (1, 1, n_top)
    atoms_bottom = layer_plain * (1, 1, n_bottom)
    defect_instance = layer_defect.copy()
    defect_instance.translate([0, 0, h * n_top])
    bottom_instance = atoms_bottom.copy()
    bottom_instance.translate([0, 0, h * (n_top + 1)])
    
    atoms = atoms_top + defect_instance + bottom_instance
    atoms.set_cell([layer_plain.cell[0,0], layer_plain.cell[1,1], h * total_layers])
    atoms.set_pbc((True, True, True))

    frozen_phonons = abtem.FrozenPhonons(atoms, num_configs=10, sigmas=0.1)
    potential = abtem.Potential(frozen_phonons, sampling=0.03, plane='xy')
    
    probe = abtem.Probe(energy=energy, semiangle_cutoff=semiangle_cutoff, Cs=Cs, defocus=defocus)
    probe.grid.match(potential)

    grid_scan = abtem.GridScan(
        start=(0, 0), end=(3/4, 3/4),
        sampling=probe.aperture.nyquist_sampling,
        fractional=True, potential=potential
    )
    
    detector = abtem.FlexibleAnnularDetector()
    flexible_measurement = probe.scan(potential, scan=grid_scan, detectors=detector)
    flexible_measurement.compute()
    
    # Post-processing
    haadf = flexible_measurement.integrate_radial(80, 200)
    output = haadf.interpolate(0.05).gaussian_filter(0.3).array.T
    output_noisy = haadf.interpolate(0.05).gaussian_filter(0.3).poisson_noise(dose_per_area=1e7).array.T
    filename = f"Layer{current_layer_index}_Total{total_layers}.npy"
    filename_noisy = f"Layer{current_layer_index}_Total{total_layers}_noisy.npy"
    np.save(save_dir / filename, output)
    np.save(save_dir / filename_noisy, output_noisy)
    return output

print(f"Starting batch {batch} for thicknesses {layers_list}...")

for total_layers in layers_list:
    print(f"\n--- Starting simulation for Total Layers: {total_layers} ---")

    for i in tqdm(range(total_layers), desc=f"Layers (Total={total_layers})"):
        run_simulation(layer_plain, layer_defect, i, total_layers)

print("All simulations complete.")