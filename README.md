# METL-Conductor

This repository contains scripts and Dockerfiles required to setup METL
fine-tuning on CHTC servers. The pre-built Docker image is currently hosted
on ```https://hub.docker.com/repository/docker/arnvsharma/metl-finetuning```.

## Building Docker container

Use ``` docker build -f Dockerfile -t <username>/<docker_hub_repo>:<repo_tag> . ``` command to build the image


Use ```docker push <username>/<docker_hub_repo>:<repo_tag>``` to push container to your docker hub repository

