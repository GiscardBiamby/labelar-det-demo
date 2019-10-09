#!/bin/bash

UIST_DATA_NAME="bid-mugs"
python object_detection/dataset_tools/create_bidmugs_tf_record.py \
    --output_dir=../../data/"${UIST_DATA_NAME}"/ \
    --train_image_dir=../../data/"${UIST_DATA_NAME}"/images/demo-mugs_train/ \
    --val_image_dir=../../data/"${UIST_DATA_NAME}"/images/demo-mugs_val/ \
    --train_annotations_file=../../data/"${UIST_DATA_NAME}"/annotations/instances_demo-mugs_train.json \
     --val_annotations_file=../../data/"${UIST_DATA_NAME}"/annotations/instances_demo-mugs_val.json
