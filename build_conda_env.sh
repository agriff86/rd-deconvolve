#!/bin/bash

#
# build a conda environment in ./env for development
#
# requires "conda" command to be available

if [[ ! -d ./env ]]
then
    conda create --prefix ./env
fi
conda env update --prefix ./env --file environment.yml  --prune
