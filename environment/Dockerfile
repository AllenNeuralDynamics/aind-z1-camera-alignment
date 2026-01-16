# hash:sha256:3ccb28c030a975186b16d4054bc54cb4677fcf15ac29d42885a80e5c509f28b4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

LABEL maintainer="Carson Berry <carson.berry@alleninstitute.org>" \
      org.opencontainers.image.title="Z1 Camera Alignment" \
      org.opencontainers.image.description="Z1 Camera Alignment" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    codeocean==0.3.0 \
    dask==2023.7.1 \
    hdf5plugin==5.1.0 \
    kerchunk==0.2.7 \
    matplotlib==3.9.2 \
    numpy==1.26.4 \
    pypdf2==3.0.1 \
    pyyaml==6.0.1 \
    s3fs==2025.7.0 \
    scikit-image==0.24.0 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    tifffile==2023.7.10 \
    xmltodict==0.14.2 \
    zarr==2.16.0
