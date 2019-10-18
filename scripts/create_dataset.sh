#!/bin/bash
set -e

DS_NAME="uist-mugs"
COLLECT_IDS_TRAIN=("4UB4" "9VJZ" "51JS" "BKAY" "DT1V" "F2C1" "F291" "GFQU" "GX5I" "JOJR" "OATT" "Y39U" "ZUPH")
COLLECT_IDS_VAL=("4EL8" "D9JF" "DCWQ" "SC7A")
DS_PATH="../data/${DS_NAME}"

# Create collect folders:
if [[ ! -d "${DS_PATH}" ]]; then
    mkdir -p "${DS_PATH}"
fi
if [[ ! -d "${DS_PATH}/train" ]]; then
    mkdir -p "${DS_PATH}/train"
fi
if [[ ! -d "${DS_PATH}/val" ]]; then
    mkdir -p "${DS_PATH}/val"
fi

# Delete collects in DS_PATH train and val subdirs:
cd "${DS_PATH}/train"
find -mindepth 1 -maxdepth 1 -type d -exec rm -r {} \;
cd -
cd "${DS_PATH}/val"
find -mindepth 1 -maxdepth 1 -type d -exec rm -r {} \;
cd -

# Re-copy collects to DS_PATH train and val subdirs:
# Train:
for cid in "${COLLECT_IDS_TRAIN[@]}"; do
    rsync -azvh "../data/uploads/${cid}/" "${DS_PATH}/train/${cid}/"
done
# Val:
for cid in "${COLLECT_IDS_VAL[@]}"; do
    rsync -azvh "../data/uploads/${cid}/" "${DS_PATH}/val/${cid}/"
done

## Create COCO-formatted train dataset:
python ./create_dataset.py \
    --ds_name uist-mugs \
    --split train \
    --collect_ids all \
    --collect_path "${DS_PATH}/train"


## Create COCO-formatted val dataset:
python ./create_dataset.py \
    --ds_name uist-mugs \
    --split val \
    --collect_ids all \
    --collect_path "${DS_PATH}/val"
