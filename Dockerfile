# Start from nvidia base image 
# - cuda only: https://hub.docker.com/r/nvidia/cuda
# - torch: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch has conda by default
FROM nvcr.io/nvidia/pytorch:20.12-py3

# OpenCV bug https://github.com/NVIDIA/nvidia-docker/issues/864 
RUN apt-get update
RUN apt-get install -y dialog apt-utils
RUN apt-get install -y libsm6 libxext6 libxrender-dev

ARG USER_ID
ARG GROUP_ID
ARG USER

# Create a non-root user
RUN groupadd --gid $GROUP_ID $USER
RUN useradd --uid $USER_ID --gid $GROUP_ID $USER

# Create conda env
WORKDIR /workspace
COPY env_dgx.yml .
RUN conda env create -f env_dgx.yml
