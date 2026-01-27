#!/bin/bash
#BSUB -q general
#BSUB -R "select[gpuhost && hname!='compute1-exec-212.ris.wustl.edu']"
#BSUB -gpu "num=1"
#BSUB -a "docker(jiananwu72/miniconda-cupy)"
# #BSUB -oo simulation_log.txt
# #BSUB -eo simulation_err.txt

export HOME=/tmp

export CUDA_PATH=/opt/conda/envs/apdd
export CPATH=$CUDA_PATH/include:$CPATH

/opt/conda/envs/apdd/bin/python scripts/LuFe_50.py