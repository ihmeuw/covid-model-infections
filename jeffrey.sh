#!/bin/bash

#this parameter comes from Jenkins
env_name=$1
src_root=$2
conda_root=$3

cmd=$4

export PATH=$PATH:$conda_root/bin
eval "$(conda shell.bash hook)"

export ENV_NAME=$env_name
export CONDA_PREFIX=$conda_root
export SGE_ENV=prod-el7
export SGE_CELL=ihme
export SGE_ROOT=/opt/sge
export SGE_CLUSTER_NAME=cluster


echo "Installing python dependencies for environment $env_name" &&
cd $src_root &&
cd 'covid-model-infections' &&
make install_env &&
# check status of previous command
if [ $? -eq 0 ]; then
    echo "Installed dependencies ${env_name}"
else
    echo "Installing dependencies failed"
    exit 1
fi


echo "Activating environment $env_name"
source activate "$env_name"
# check status of previous command
if [ $? -eq 0 ]; then
    echo "Activated environment ${env_name}"
else
    echo "Activating conda env failed"
    exit 1
fi

echo "Conda packages:"
conda list

echo "Run model infections"
echo $cmd
if eval "$cmd"; then
    echo "Jeffrey completed"
else
    echo "Jeffrey failed"
    exit 1
fi
