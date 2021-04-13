#!/bin/bash
docker_tag=10.202.67.207:5000/$USER:hil_01

docker build . -f Dockerfile \
  --tag $docker_tag \
  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=$USER \
  --network=host
docker push $docker_tag
