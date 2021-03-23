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
pwd
echo "Source root is ${src_root}"
cd $src_root
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

echo "Run model infections"
echo $cmd
if eval "$cmd"; then
    echo "Jeffrey completed"
else
    echo "Jeffrey failed"
    exit 1
fi
