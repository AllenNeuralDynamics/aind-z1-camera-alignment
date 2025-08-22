# hash:sha256:82984ff08222caed20287a221e638cde00daa66c19a9333d85dbf496f811f974
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG CUSTOM_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
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
    seaborn==0.13.2 \
    tifffile==2023.7.10 \
    xmltodict==0.14.2 \
    zarr==2.16.0
