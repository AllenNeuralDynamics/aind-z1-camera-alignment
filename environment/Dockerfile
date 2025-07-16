# hash:sha256:79a4445349c8934420ea546633941eb240894d9e96ab61e55c206dda92c31be3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab-py39:1.0

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG CUSTOM_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-transfer[imaging]==0.25.2 \
    codeocean==0.3.0 \
    cupy-cuda11x==12.3.0 \
    dask==2023.7.1 \
    hdf5plugin==5.1.0 \
    kerchunk==0.2.7 \
    matplotlib==3.9.2 \
    numpy==1.24.4 \
    pyyaml==6.0.1 \
    scikit-image==0.21.0 \
    seaborn==0.13.2 \
    tifffile==2023.7.10 \
    zarr==2.16.0
