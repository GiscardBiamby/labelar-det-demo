#!/bin/bash
. utils.sh

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
            outvar_platform="gpu"
        fi
    fi
    platform=$outvar_platform
    set -e
}

function check_conda_env_exists() {
    set -e
    ENVS=$(conda info --envs | awk '{print $1}')
    echo "ENVS: ${ENVS}"
    if [[ $ENVS = *"$1"* ]]; then
        ENV_EXISTS=true
    else
        ENV_EXISTS=false
    fi
}

# Need this on some systems to enable conda from bash:
eval "$(conda shell.bash hook)"
check_conda_env_exists "labelar_demo"
echo "labelar_demo conda env EXISTS?: ${ENV_EXISTS}"
# exit 0

# Assume we are in ./setup/ when script is called

# Detect platform:
platform="cpu"
get_platform $platform

# Delete and recreate conda env:
REBUILD_ENV=true
if [[ "$REBUILD_ENV" = true ]]; then
    echo "(Re)-building labelar_demo conda env ${REBUILD_ENV}"
    conda deactivate
    if [[ $ENV_EXISTS = true ]]; then
        conda env remove -n labelar_demo
    fi
    conda env create -f environment-$platform.yml
    if [[ "${platform}" == "cpu" ]]; then
        if [[ -f environment-cpu.yml.bak ]]; then
            mv environment-cpu.yml.bak environment-cpu.yml
        fi
    fi
fi

##
## Enable using the h4d_env conda environment from jupyter notebooks:
conda activate labelar_demo
python -m ipykernel install --user --name labelar_demo --display-name "labelar_demo"
# Install pycocotools:
cd ../vendor/cocoapi/PythonAPI
python3 setup.py build_ext install
cd -

##
## Setup SSD:
echo "${platform}"
if [[ "${platform}" == "gpu" ]]; then
    echo "Performing step(s) that require GPU"
    echo "============================================="
    echo "Installing SSD dependencies..."
    # Do GPU specific setup here, e.g., CenterNet, DeformConvVxx, any compilation steps that require CUDA,etc
    cd ../training/ssd-det/ext
    selcompiler
    cd -
else
    echo "Skipping step(s) that require GPU"
fi

##
## Setup tensorflow object-detection:
echo "============================================="
echo "Installing tensorflow object detection dependencies..."
cd ../training/tfmodels/research/
protoc object_detection/protos/*.proto --python_out=.
echo "Verifying tensorflow object detection installation..."
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
cd -

# cd ../training/
# git clone --recurse-submodules https://github.com/tensorflow/tensorflow
# cd -


##
## Download weights: We may or may not want to grab some models off the model zoo from
## here:
## https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
mkdir -p ../training/weights/pretrained/
cd ../training/weights/pretrained/
if [[ ! -f squeezenet1_1-f364aa15.pth ]]; then
    wget https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth
fi
if [[ ! -f coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip ]]; then
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
fi
unzip -f coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d ./coco_ssd_mobilenet_v1_1.0_quant
cd -