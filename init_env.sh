#!/bin/bash

echo "NOTE: Launch this script like so: 'source ./init_env.sh'. (See: https://askubuntu.com/questions/53177/bash-script-to-set-environment-variables-not-working)"

# Activate Conda Env:
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate labelar_demo

# Add paths to PYTHONPATH:
echo "Appending tfmodels paths to PYTHONPATH..."
cd ./training/tfmodels/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim