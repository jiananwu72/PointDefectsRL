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
save_dir = project_root / 'data' / 'simulations'

# Set simulation on GPU
abtem.config.set({"device": "gpu", "fft": "fftw"})

sweep_configs = {
    'energy': [100e3, 300e3],      # eV
    'Cs': [0, 5.6e4],              # in Angstroms
    'layers': [10],     # number of layers along z
    'semiangle_cutoff': [15, 22, 30, None],      # in milliradians
}

def run_simulation(unit_cell, energy, Cs, layers, semiangle_cutoff):
    """
    Runs a single abtem simulation with specific parameters.
    """
    print(f"Running simulation with energy={energy/1e3}keV, Cs={Cs}Å, layers={layers}, semiangle_cutoff={semiangle_cutoff}mrad")
    atoms = unit_cell * (2, 2, layers)  # Repeat unit cell along z
    frozen_phonons = abtem.FrozenPhonons(atoms, num_configs=10, sigmas=0.1)
    potential = abtem.Potential(frozen_phonons, sampling=0.05, plane='xy')
    probe = abtem.Probe(
        energy=energy, 
        semiangle_cutoff=semiangle_cutoff, 
        Cs=Cs, 
        defocus=0
    )
    probe.grid.match(potential)

    grid_scan = abtem.GridScan(
        start=(0, 0),
        end=(3/4, 3/4),
        sampling=probe.aperture.nyquist_sampling,
        fractional=True,
        potential=potential,
    )
    
    detector = abtem.FlexibleAnnularDetector()
    flexible_measurement = probe.scan(potential, scan=grid_scan, detectors=detector)
    flexible_measurement.compute()
    
    haadf = flexible_measurement.integrate_radial(90, 200)
    interpolated = haadf.interpolate(0.05)
    filtered = interpolated.gaussian_filter(0.3)
    noisy = filtered.poisson_noise(dose_per_area=1e7)

    output = noisy.array.T
    filename = f"noisy_E{energy/1000:.0f}keV_Cs{Cs:.1e}_Layers{layers}_Cutoff{semiangle_cutoff}mrad.npy"
    save_path = save_dir / filename
    np.save(save_path, output)

    return output

param_names = list(sweep_configs.keys())
param_values = list(sweep_configs.values())
combinations = list(product(*param_values))

results = []

print(f"Running {len(combinations)} simulations...")

for params in tqdm(combinations):
    current_kwargs = dict(zip(param_names, params))
    sim_result = run_simulation(unit_cell, **current_kwargs)    
    
print("All simulations complete.")