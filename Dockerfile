FROM continuumio/miniconda3 

# Set a working directory
WORKDIR /env

COPY environment.yml .

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get clean \
&& conda env create -f environment.yml

SHELL ["conda", "run", "-n", "point_defects", "/bin/bash", "-c"]

RUN git clone <repository_url> /app