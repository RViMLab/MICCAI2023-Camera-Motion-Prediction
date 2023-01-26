# Start from miniconda
FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y tmux
RUN apt-get install build-essential -y

ARG USER_ID
ARG GROUP_ID
ARG USER

# Create a non-root user
RUN groupadd --gid $GROUP_ID $USER
RUN useradd --uid $USER_ID --gid $GROUP_ID $USER

# Create conda env
WORKDIR /workspace
COPY env_torch110.yml .
RUN conda update --name base conda
RUN conda install mamba -c conda-forge
RUN conda create -n torch110 python=3
RUN mamba env update -n torch110 -f env_torch110.yml
