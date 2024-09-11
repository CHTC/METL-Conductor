# Modified from https://github.com/pytorch/pytorch/blob/main/Dockerfile to run on CHTC 
FROM nvcr.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04 as base

FROM base as base-amd64

ENV NV_CUDNN_VERSION 8.9.0.131
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda12.1"

FROM base as base-arm64

ENV NV_CUDNN_VERSION 8.9.0.131
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda12.1"

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"
LABEL com.nvidia.volumes.needed="nvidia_driver"


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        pciutils \
        && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} curl\
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

COPY environment.yml .


## TODO: Change this if you changed the name of the conda environment for any dev work
ARG CONDA_ENV=metl

# Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda env create -f environment.yml && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:${PATH}
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH
ENV CONDA_DEFAULT_ENV $CONDA_ENV
RUN /bin/bash -c "source activate ${CONDA_ENV}"

WORKDIR /workspace
