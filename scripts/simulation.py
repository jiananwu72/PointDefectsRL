import ase
import ase.io
import numpy as np
import matplotlib.pyplot as plt
import abtem

import sys
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from util.crop import Crop

# Load structure; cif_path will be edited to load from command line
cif_path = 'data/structures/LFO_Orth.cif'
atoms = ase.io.read(cif_path)

# Set simulation on GPU
abtem.config.set({"device": "cpu", "fft": "fftw"})

# atoms *= (3, 1, 1)  # Repeat unit cell if needed

potential = abtem.Potential(atoms, sampling=0.05, plane='xy')
probe = abtem.Probe(energy=100e3, semiangle_cutoff=32, Cs=0, defocus="scherzer")
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
haadf_measurement = flexible_measurement.integrate_radial(90, 200)

interpolated_measurements = haadf_measurement.interpolate(0.05)
filtered_measurements = interpolated_measurements.gaussian_filter(0.3)
noisy_measurements = filtered_measurements.poisson_noise(dose_per_area=1e7)

output = noisy_measurements.array.T
np.save('noisy_data.npy', output)