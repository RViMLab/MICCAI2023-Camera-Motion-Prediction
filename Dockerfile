# Start from nvidia base image 
# - cuda only: https://hub.docker.com/r/nvidia/cuda
# - torch: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch has conda by default
FROM nvidia/cuda:10.1-base-ubuntu18.04

ARG USER_ID
ARG GROUP_ID
ARG USER

# Create a non-root user
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

# Create conda env
WORKDIR /workspace
COPY env.yml .
RUN conda env create -f env.yml
