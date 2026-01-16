#!/bin/bash
#BSUB -q general
#BSUB -R 'gpuhost'
#BSUB -gpu "num=1"
#BSUB -a "docker(jiananwu72/miniconda-cupy)"
# #BSUB -oo simulation_log.txt
#BSUB -eo simulation_err_LuFe.txt

export HOME=/tmp

export CUDA_PATH=/opt/conda/envs/apdd
export CPATH=$CUDA_PATH/include:$CPATH

/opt/conda/envs/apdd/bin/python scripts/test_LuFe.py