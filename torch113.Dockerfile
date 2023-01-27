# Start from miniconda
FROM continuumio/miniconda3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y tmux

ARG USER_ID
ARG GROUP_ID
ARG USER

# Create a non-root user
RUN groupadd --gid $GROUP_ID $USER
RUN useradd --uid $USER_ID --gid $GROUP_ID $USER

# Create conda env
WORKDIR /workspace
COPY env_torch113.yml .
RUN conda update --name base conda
RUN conda install mamba -c conda-forge
RUN conda create -n torch113 python=3
RUN mamba env update -n torch113 -f env_torch113.yml
