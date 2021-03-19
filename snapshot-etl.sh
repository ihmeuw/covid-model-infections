#!/bin/bash

#this parameter comes from Jenkins
env_name=$1
src_root=$2
conda_root=$3

snapshot_cmd=$4
etl_cmd=$5

export PATH=$PATH:$conda_root/bin
eval "$(conda shell.bash hook)"

echo "Creating environment $env_name"
umask 002
conda create -y --name="$env_name" python=3.7
# check status of previous command
if [ $? -eq 0 ]; then
    echo "Created environment ${env_name}"
else
    echo "Creating conda env failed"
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
echo "Installing python dependencies for environment $env_name" &&
pip install -e $src_root/covid-shared/ &&
echo "Installing non-python dependencies for environment $env_name" &&
conda install -y -c hcc rclone unzip &&
echo "Installing covid-input-snapshot $env_name" &&
pip install -e $src_root/covid-input-snapshot/ --extra-index-url https://artifactory.ihme.washington.edu/artifactory/api/pypi/pypi-shared/simple/&&
pip install -e $src_root/covid-input-etl/ --extra-index-url https://artifactory.ihme.washington.edu/artifactory/api/pypi/pypi-shared/simple/ &&
# check status of previous command
if [ $? -eq 0 ]; then
    echo "Installed dependencies ${env_name}"
else
    echo "Installing dependencies failed"
    exit 1
fi

echo "Snapshotting..."
echo $snapshot_cmd
if eval "$snapshot_cmd"; then
    echo "Snapshot completed"
else
    echo "Snapshot failed"
    exit 1
fi

echo "Etling..."
echo $etl_cmd
if eval "$etl_cmd"; then
    echo "Etl completed"
else
    echo "Etl failed"
    exit 1
fi

