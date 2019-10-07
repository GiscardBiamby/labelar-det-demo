#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate labelar_demo

# xview_coco_v2_train_chipped: 10557, xview_coco_v2_tiny_train_chipped: 128, COCO:82783, VOC0712: 16551
DATASET_SIZE=648

# batch sizes: SSD300: { TitanXp: 48, RTX: 96 }, SSD512 { TitanXp: 20, }
BATCH_SIZE=96
ITERS_PER_EPOCH=$((DATASET_SIZE/$BATCH_SIZE))
EPOCHS=1000
# MAX_ITER is what ssd-master uses to know how many training iterations to run.
# If resuming from previous training,t he iteration number picks up from where
# the previous training left off, so you'll want to hand-pick this value accordingly:
MAX_ITER=$(($ITERS_PER_EPOCH*$EPOCHS*2))
LOG_STEP=$(($ITERS_PER_EPOCH/25))
SAVE_STEP=$ITERS_PER_EPOCH
EVAL_STEP=$((ITERS_PER_EPOCH*40))

# Ensure LOG_STEP>0:
LOG_STEP=$(( LOG_STEP > 0 ? LOG_STEP : ITERS_PER_EPOCH ))

echo "DATASET_SIZE: ${DATASET_SIZE}"
echo "ITERS_PER_EPOCH: ${ITERS_PER_EPOCH}"
echo "EPOCHS: ${EPOCHS}"
echo "MAX_ITER: ${MAX_ITER}"
echo "LOG_STEP: ${LOG_STEP}"
echo "EVAL_STEP: ${EVAL_STEP}"


## GPU COCO sigma:
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-file ./configs/mobilenet_v2_ssd320_uist.yaml \
    --log_step $LOG_STEP \
    --save_step $SAVE_STEP \
    --eval_step $EVAL_STEP \
    SOLVER.MAX_ITER $MAX_ITER \
    SOLVER.BATCH_SIZE $BATCH_SIZE \
    TEST.BATCH_SIZE 32 \
    # MODEL.BACKBONE.CONV6_SIGMA True \
    # FOO 1 & # Keep this as last line of script: \
