# hash:sha256:ae1da8447949d6f5e919c66083b662e17c0eb4d88489d3c81b6c9b13044d16b7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    codeocean==0.3.0 \
    dask==2023.7.1 \
    hdf5plugin==5.1.0 \
    kerchunk==0.2.7 \
    matplotlib==3.9.2 \
    numpy==1.24.4 \
    pyyaml==6.0.1 \
    s3fs==2025.7.0 \
    scikit-image==0.21.0 \
    scipy==1.16.1 \
    seaborn==0.13.2 \
    tifffile==2023.7.10 \
    xmltodict==0.14.2 \
    zarr==2.16.0
