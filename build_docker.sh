#!/bin/bash
docker_tag=aicregistry:5000/$USER:torch113

docker build . -f torch113.Dockerfile \
  --tag $docker_tag \
  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=$USER \
  --network=host
docker push $docker_tag
