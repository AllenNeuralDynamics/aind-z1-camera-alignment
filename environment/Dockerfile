# hash:sha256:d8810196ae3d37cf39d7b3ec18a9d9b523ca67a3b8342acff45e413b60c4a94e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab:2.3

# enter SHA or just main
ARG QC_PKG_COMMIT=6ee2c1adbd876619217cccb4144614e4a346558f

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
    cupy-cuda11x==12.3.0 \
    dask==2023.7.1 \
    matplotlib==3.9.2 \
    numpy==1.24.4 \
    pyyaml==6.0.1 \
    scikit-image==0.21.0 \
    seaborn==0.13.2 \
    tifffile==2023.7.10 \
    zarr==2.16.0 \
    aind-data-transfer[imaging]==0.25.2 \

RUN pip install -U --no-cache-dir --upgrade \
    git+https://github.com/AllenNeuralDynamics/hcr_data_qc.git@${QC_PKG_COMMIT}#egg=hcr_data_qc

COPY postInstall /
RUN /postInstall
