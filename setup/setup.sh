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

function check_conda_env_exists() {
    source activate labelar_demo
    if [ $? -eq 0 ]; then
        echo "EXISTS"
    else
        echo "DOES NOT EXIST"
    fi
}

# Need this on some systems to enable conda from bash:
eval "$(conda shell.bash hook)"
check_conda_env_exists
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
    conda env remove -n labelar_demo
    conda env create -f environment-$platform.yml
    if [[ "${platform}" == "cpu" ]]; then
        if [[ -f environment-cpu.yml.bak ]]; then
            mv environment-cpu.yml.bak environment-cpu.yml
        fi
    fi
fi


# Enable using the h4d_env conda environment from jupyter notebooks:
conda activate labelar_demo
python -m ipykernel install --user --name labelar_demo --display-name "labelar_demo"
# Install pycocotools:
cd ../vendor/cocoapi/PythonAPI
python3 setup.py build_ext install
cd -


# if [[ "${platform}" == "gpu" ]]; then
#     # Do GPU specific setup here, e.g., CenterNet, DeformConvVxx.
# fi


# Download weights: We may or may not want to grab some models off the model zoo from
# here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
mkdir -p ../training/weights/pretrained/
cd ../training/weights/pretrained/
if [[ ! -f squeezenet1_1-f364aa15.pth ]]; then
    wget https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth
fi
if [[ ! -f coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip ]]; then
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
fi
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d ./coco_ssd_mobilenet_v1_1.0_quant
cd -