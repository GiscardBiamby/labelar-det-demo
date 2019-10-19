#!/bin/bash
set -e


DS_NAME="uist-mugs-v2"
DATA_DIR=../training/data
OUTPUT_DIR="${DATA_DIR}/${DS_NAME}-tfrecords"

if [[ -d "${OUTPUT_DIR}" ]]; then
    rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

python ../training/tfmodels/research/object_detection/dataset_tools/create_labelar_tf_record.py \
    --output_dir="${OUTPUT_DIR}" \
    --train_image_dir="${DATA_DIR}/${DS_NAME}-train"/images/"${DS_NAME}_train"/ \
    --val_image_dir="${DATA_DIR}/${DS_NAME}-val"/images/"${DS_NAME}_val"/ \
    --train_annotations_file="${DATA_DIR}/${DS_NAME}-train"/annotations/"instances_${DS_NAME}_train.json" \
    --val_annotations_file="${DATA_DIR}/${DS_NAME}-val"/annotations/"instances_${DS_NAME}_val.json" \
    --ds_name="${DS_NAME}"