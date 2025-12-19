#!/bin/bash
#BSUB -q general
#BSUM -R 'gpuhost'
#BSUB -gpu "num=1"
#BSUB -a "docker(jiananwu72/miniconda-cupy)"
#BSUB -oo simulation_log.txt
#BSUB -eo simulation_err.txt

# Run the simulation
python scripts/simulation.py