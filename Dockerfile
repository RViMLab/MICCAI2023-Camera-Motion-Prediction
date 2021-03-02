# Start from nvidia base image 
# - cuda only: https://hub.docker.com/r/nvidia/cuda
# - torch: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch has conda by default
FROM pytorch/conda-builder:cuda101

# OpenCV bug https://github.com/NVIDIA/nvidia-docker/issues/864 
RUN yum install -y libSM libXext libXrender

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
