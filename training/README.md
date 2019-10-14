## Training

## Setup

Make sure you've run [../setup/setup.sh](../setup/setup.sh)

## Run Training

``` bash
export LABELAR_DEMO_ROOT=./path/to/this/repo
export TRAIN_DIR="${LABELAR_DEMO_ROOT}/training/tfmodels/research"
# Activate conda environment, and set environment/path/python vars:
source ./init_env.sh

cd "${TRAIN_DIR}"
# Convert dataset to .tfrecord format:
# NOTE!: This script is hardcoded to convert the "bidmugs" dataset. You'll
#   need to copy-n-modify the following files to handle a different dataset:
#       1. $"{TRAIN_DIR}"/object_detection/dataset_tools/create_bidmugs_tf_record.py
#       2. $"{TRAIN_DIR}"/create_bidmugs_tfrecords.sh
./create_bidmugs_tfrecords.sh
# Train:
cd "${LABELAR_DEMO_ROOT}/training/tfmodels/research"
./train_mobile_model.sh 2>&1 | tee train_mobile_model.log
```

## TODO:
- [X] Collect fake dataset (used mugs in bid)
- [X] Buy mugs
- [X] Collect validation dataset
- [X] Merge/Transform labelar .json's into single .json for the COCO-formatted dataset
- [X] Add Pipeline step: Train model
- [X] Add Pipeline step: Convert model to tflite
- [X] Deploy & Test model on android (good; ~48-50ms per inference, (see NNAPI bullet point for updated inference time))
- [ ] Convert dataset transformations from notebook into script
- [ ] Add dataset transformation script call to train_mobile_model.sh
- [ ] Add tfrecord format conversion to train_mobile_model.sh
- [ ] Add generation of labelmap.txt (values retrieved from the dataset's coco-formatted json)
- [ ] Create json to manage models on mobile device. Store model metadata, model paths
- [ ] Allow android app to switch between models
## Maybe TODO:
- [ ] Add android build to pipeline
- [ ] Add pipeline step to deploy to android if connected via usb
- [X] Test NNAPI (Nice, inference time went from ~48ms to ~25-28ms)
    * Using (1/60fps)*1000=16.67ms, we have: 60fps=>"<= 16.67ms/frame", and 30fps=>"<= 33.34ms/frame" so we are good to get ~35.7fps on the Pixel 3XL with ssd_mobilenet_v2_quant_bigmugs and NNAPI enabled (a 300x300 input size model).
