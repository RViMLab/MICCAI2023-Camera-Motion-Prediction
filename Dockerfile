# Start from nvidia base image 
# - cuda only: https://hub.docker.com/r/nvidia/cuda
# - torch: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch has conda by default
FROM nvcr.io/nvidia/pytorch:21.12-py3

# OpenCV bug https://github.com/NVIDIA/nvidia-docker/issues/864 
RUN apt-get update
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN apt-get install -y tmux

ARG USER_ID
ARG GROUP_ID
ARG USER

# Create a non-root user
RUN groupadd --gid $GROUP_ID $USER
RUN useradd --uid $USER_ID --gid $GROUP_ID $USER

# Create conda env
WORKDIR /workspace
COPY env_torch110.yml .
RUN conda update conda
RUN conda update conda-build
RUN conda install mamba -c conda-forge
RUN conda create -n torch110 python=3.9
RUN mamba env update -n torch110 -f env_torch110.yml
