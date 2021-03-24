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

echo "Troubleshooting location"
echo "Before"
pwd
ls
cd $src_root
cd 'covid-model-infections'
echo "After"
pwd
ls
echo "Installing python dependencies for environment $env_name" &&
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

echo "Running conda list"
conda list

echo "Run model infections"
echo $cmd
if eval "$cmd"; then
    echo "Jeffrey completed"
else
    echo "Jeffrey failed"
    exit 1
fi
