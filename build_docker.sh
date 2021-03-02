#!/bin/bash
docker_tag=10.202.67.201:32581/$USER:hil_04

docker build . -f Dockerfile \
  --tag $docker_tag \
  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=$USER
docker push $docker_tag
