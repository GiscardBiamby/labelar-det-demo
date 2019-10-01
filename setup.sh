#!/bin/bash


function program_exists() {
    command -v "$1" >/dev/null 2>&1
}

function get_platform() {
# removed local declaration of outvar_platform to avoid bash version issues with -n flag
    outvar_platform=$1 # use an output var
    outvar_platform="cpu"
    set +e
    if program_exists nvcc; then
        if [[ "$H4D_PLATFORM" ]]; then
            outvar_platform="$H4D_PLATFORM"
        else
            ask_for_confirmation \
                "CUDA detected! Do you want to setup the conda environment with GPU support? (recommended)"
            if answer_is_yes; then
                outvar_platform="gpu"
            fi
        fi
    fi
    platform=$outvar_platform
    set -e
}

# Detect platform:
platform="cpu"
get_platform $platform

# Delete and recreate conda env:
eval "$(conda shell.bash hook)"
conda deactivate
conda env remove -n labelar_demo
conda env create -f environment-$platform.yml
if [[ "${platform}" == "cpu" ]]; then
    mv environment-cpu.yml.bak environment-cpu.yml
fi

# Enable using the h4d_env conda environment from jupyter notebooks:
conda activate labelar_demo
python -m ipykernel install --user --name h4d_env --display-name "labelar_demo"
# Install pycocotools:
cd cocoapi/PythonAPI
python3 setup.py build_ext install
cd ../..
# Install def_conv:
cd vendor/def_conv
python3 setup.py build_ext install
cd -


if [[ "${platform}" == "gpu" ]]; then
    # Do GPU specific setup here, e.g., CenterNet, DeformConvVxx.
fi


# Download weights:
mkdir -p ./training/weights/pretrained/
cd ./training/weights/pretrained/
wget https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth