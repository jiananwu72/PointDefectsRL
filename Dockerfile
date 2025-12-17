# To build this docker image to be used on HPCs (directory at APDD):
#     docker build --platform linux/amd64 -t jiananwu72/miniconda-cupy:latest .
# To run the docker image, use jiananwu72/miniconda-cupy:latest.

# In WashU RIS, use: bsub -Is -q general-interactive -a 'docker(jiananwu72/miniconda-cupy)' /bin/bash

# For simulations on a machine with NVIDIA GPU support
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Code adapted and revised from ContinuumIO Miniconda3 Dockerfile
# Source: https://hub.docker.com/r/continuumio/miniconda3/dockerfile

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    /opt/conda/bin/conda clean -afy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate base" >> /etc/bash.bashrc && \
    echo "export HOME=/tmp" >> /etc/bash.bashrc

ENV TINI_VERSION=v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Install packages from environment.yml and CuPy for CUDA 12.x
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
RUN conda run -n apdd pip install --no-cache-dir \
    cupy-cuda12x
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/compat:$LD_LIBRARY_PATH
ENV PATH=/opt/conda/envs/apdd/bin:$PATH

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]