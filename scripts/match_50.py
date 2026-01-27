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

# Load structure; cif_path will be edited to load from command line
cif_path = project_root / 'data' / 'structures' / 'LFO_Orth.cif'
unit_cell = ase.io.read(cif_path)
save_dir = project_root / 'data' / 'simulations' / 'match'
save_dir.mkdir(parents=True, exist_ok=True)

# Set simulation on GPU
abtem.config.set({"device": "gpu", "fft": "fftw"})

sweep_configs = {
    'energy': [100e3],      # eV
    'Cs': [5.593e4],              # in Angstroms
    'layers': [50],     # number of layers along z
    'semiangle_cutoff': [10, 15, 20, 22, 25, 30],      # in milliradians
    'defocus': [17.14],               # in Angstroms
}

# sweep_configs = {
#     'energy': [100e3],      # eV
#     'Cs': [5.593e4],              # in Angstroms
#     'layers': [20, 30, 40, 50, 60],     # number of layers along z
#     'semiangle_cutoff': [10, 15, 20, 22, 25, 30],      # in milliradians
#     'defocus': [17.14],               # in Angstroms
# }

def run_simulation(unit_cell, energy, Cs, layers, semiangle_cutoff, defocus):
    """
    Runs a single abtem simulation with specific parameters.
    """
    print(f"Running simulation with energy={energy/1e3}keV, Cs={Cs}Å, layers={layers}, \
          semiangle_cutoff={semiangle_cutoff}mrad, defocus={defocus}Å")
    atoms = unit_cell * (2, 2, layers)  # Repeat unit cell along z
    frozen_phonons = abtem.FrozenPhonons(atoms, num_configs=10, sigmas=0.1)
    potential = abtem.Potential(frozen_phonons, sampling=0.03, plane='xy')
    probe = abtem.Probe(
        energy=energy, 
        semiangle_cutoff=semiangle_cutoff, 
        Cs=Cs, 
        defocus=defocus
    )
    probe.grid.match(potential)

    grid_scan = abtem.GridScan(
        start=(0, 0),
        end=(1, 1),
        sampling=probe.aperture.nyquist_sampling,
        fractional=True,
        potential=potential,
    )
    
    detector = abtem.FlexibleAnnularDetector()
    flexible_measurement = probe.scan(potential, scan=grid_scan, detectors=detector)
    flexible_measurement.compute()
    
    haadf = flexible_measurement.integrate_radial(80, 200)
    interpolated = haadf.interpolate(0.05)
    filtered = interpolated.gaussian_filter(0.3)
    noisy = filtered.poisson_noise(dose_per_area=1e7)

    output = filtered.array.T
    filename = f"Layers{layers}_Cutoff{semiangle_cutoff}.npy"
    save_path = save_dir / filename
    np.save(save_path, output)

    output_noisy = noisy.array.T
    filename_noisy = f"Layers{layers}_Cutoff{semiangle_cutoff}_noisy.npy"
    save_path_noisy = save_dir / filename_noisy
    np.save(save_path_noisy, output_noisy)

    return output

param_names = list(sweep_configs.keys())
param_values = list(sweep_configs.values())
combinations = list(product(*param_values))

# Run simulations for all combinations
print(f"Running {len(combinations)} simulations...")
for params in tqdm(combinations):
    current_kwargs = dict(zip(param_names, params))
    sim_result = run_simulation(unit_cell, **current_kwargs)    
print("All simulations complete.")