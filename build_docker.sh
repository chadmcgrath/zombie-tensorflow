#!/bin/sh

DOCKER_TAG=${1:-zombie-tensorflowp}
DOCKER_DEFAULT_PLATFORM=${2:-linux/amd64}

docker build --platform $DOCKER_DEFAULT_PLATFORM -t $DOCKER_TAG .

