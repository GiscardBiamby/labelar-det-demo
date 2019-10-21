#!/bin/bash
set -e

#
# UISTmugsv1 (2 metal, 2 glass, dundermiflin, bestboss):
# DS_NAME="uist-mugs"
# COLLECT_IDS_TRAIN=(
#     "4UB4" "9VJZ" "51JS" "BKAY" "DT1V" "F2C1" "F291" "GFQU" "GX5I" "JOJR" \
#     "OATT" "Y39U" "ZUPH"
# )
# COLLECT_IDS_VAL=("4EL8" "D9JF" "DCWQ" "SC7A")

#
# UISTmugsv2 (blue, cal, dundermiflin, gold, flower):
DS_NAME="uist-mugs-v2"
COLLECT_IDS_TRAIN=( \
    "HVSU" "SQR4" "8NPZ" "8H6L" "TKWI" "AMT1" "ZH7O" "OUPY" "4AIN" "O5NO" \
    "M93C" "MRUX" "E9J9" "X9U5" "IFRL" "UCB4" "9NY4" "8SZF" "1JV4" "2669" \
    "9NQ6" "T829" "UAIA" "FWSJ" "W82D" "N5U1" "9YLE" "7R02" "NTO2" "G7ON" \
    "M0H4" "GHUC" "Z9H9" "KWX4" "US2P" "YXF2" "DLDA" \
    "EVMW" "D2VS" "LA24" "O5XF" "X2K3"
)
COLLECT_IDS_VAL=(
    "25HZ" "NU6K" "QVBF" "DMIW" "I4NB" "ETYY" "ZQVN" "G7F1" "74RP" \
    "KMXP" "Z0TD" "J1WM"
)
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
    echo "Copyin train files: '../data/uploads/${cid}/'"
    rsync -azvh "../data/uploads/${cid}/" "${DS_PATH}/train/${cid}/"
done
# Val:
for cid in "${COLLECT_IDS_VAL[@]}"; do
    rsync -azvh "../data/uploads/${cid}/" "${DS_PATH}/val/${cid}/"
done

## Create COCO-formatted train dataset:
python ./create_dataset.py \
    --ds_name "${DS_NAME}" \
    --split train \
    --collect_ids all \
    --collect_path "${DS_PATH}/train"

## Create COCO-formatted val dataset:
python ./create_dataset.py \
    --ds_name "${DS_NAME}" \
    --split val \
    --collect_ids all \
    --collect_path "${DS_PATH}/val" 
