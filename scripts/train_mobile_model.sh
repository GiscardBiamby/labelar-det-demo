#!/bin/bash
set -e

source ../setup/utils.sh

start_task() {
    print_in_green "\n"
    print_in_green "\n"
    print_in_green "[!] ==============================================================================================================================================\n"
    print_in_green "[!] ==============================================================================================================================================\n"
    print_in_green "[!] $1\n"
}

cd ../training/tfmodels/research

export DS_NAME="uist-mugs-v2"
export SAVE_CKPT_STEPS=160
export INPUT_SIZE=300
# export MODEL_NAME="mobilenet_v2_quant_aug_uistmugsv2"
# export MODEL_NAME="mobilenet_v2_quant_aug_uistmugsv2_lrdebug"
export MODEL_NAME="mobilenet_v2_quant_aug_uistmugsv2_final"
# export MODEL_NAME="resnet50_fpn_uistmugsv2"
# export MODEL_NAME="mobilenet_v1_fpn_uistmugsv2"
# export MODEL_NAME="mobilenet_v1_fpn_shared_box_predictor_640x640_uistmugs_sync"
# You may have to edit this .config to configure any of: dataset, model, training:
export PIPELINE_CONFIG_PATH=../../exp_configs/"ssd_${MODEL_NAME}.config"
export NUM_TRAIN_STEPS=3200
export QUANTIZED_TRAINING=true
export SAMPLE_1_OF_N_EVAL_EXAMPLES=1
export MODEL_DIR=../../weights/"ssd_${MODEL_NAME}"
export CHECKPOINT_PATH="${MODEL_DIR}/model.ckpt-${NUM_TRAIN_STEPS}"
export OUTPUT_DIR="${MODEL_DIR}/tflite"
export USE_QUANTIZED=true
export ANDROID_ASSET_PATH=../../../android/app/src/main/assets/"${MODEL_NAME}"

# DELETE_EXISTING=false
# if [[ -d "${MODEL_DIR}" ]]; then
#     echo "Deleting directory: '${MODEL_DIR}''"
#     rm -rf "${MODEL_DIR}"
# fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Creating directory: '${MODEL_DIR}''"
    mkdir -p "${MODEL_DIR}"
fi


##
## Train model:
export TRAIN=true
if [[ "${TRAIN}" == true ]]; then
    start_task "Training model: ${MODEL_NAME}..."
    python -u object_detection/model_main.py \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --model_dir=${MODEL_DIR} \
        --save_checkpoints_steps=${SAVE_CKPT_STEPS} \
        --eval_throttle_secs=0 \
        --sample_1_of_n_eval_examples="${SAMPLE_1_OF_N_EVAL_EXAMPLES}" \
        --alsologtostderr
        # --num_train_steps=${NUM_TRAIN_STEPS} \
fi

##
## Eval model:
start_task "Evaluating model: ${MODEL_NAME}..."
echo ""
echo "pipeline_config_path: ${PIPELINE_CONFIG_PATH}"
echo "checkpoint_dir: ${MODEL_DIR}"
echo "checkpoint_path: ${CHECKPOINT_PATH}"
# If you specify --checkpoint_path here it will force the eval to run on a specifick checkpoint,
# for example you would want to do that if you want eval the model checkpoint from a specific step number.
# we pass in --checkpoint_path=$CHECKPOINT_PATH, which uses $NUM_TRAIN_STEPS as the step number
python -u object_detection/model_main.py \
    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
    --checkpoint_dir=$"${MODEL_DIR}" \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --run_once \
    --sample_1_of_n_eval_examples=1 \
    --alsologtostderr 

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Creating directory: '${OUTPUT_DIR}''"
    mkdir -p "${OUTPUT_DIR}"
fi

##
## Export tflite frozen graph (e.g., make tflite_graph.ph):
start_task "Creating tflite frozen graph for model ${MODEL_NAME}..."
python object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --trained_checkpoint_prefix=$CHECKPOINT_PATH \
    --output_directory=$OUTPUT_DIR \
    --add_postprocessing_op

##
## Generate detect.tflite:
# https://heartbeat.fritz.ai/8-bit-quantization-and-tensorflow-lite-speeding-up-mobile-inference-with-low-precision-a882dfcafbbd
# https://www.tensorflow.org/lite/performance/post_training_quantization
if [[ "${USE_QUANTIZED}" == true ]]; then
    start_task "Generating quantized tflite frozen graph for model ${MODEL_NAME}..."
    if [[ "${QUANTIZED_TRAINING}" == true ]]; then
        tflite_convert \
            --graph_def_file="${OUTPUT_DIR}/tflite_graph.pb" \
            --output_file="${OUTPUT_DIR}/detect.tflite" \
            --input_shapes="1,${INPUT_SIZE},${INPUT_SIZE},3" \
            --input_arrays=normalized_input_image_tensor \
            --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
            --allow_custom_ops \
            --mean_values=128 \
            --std_dev_values=128 \
            --inference_type=QUANTIZED_UINT8 \
            --inference_input_type=QUANTIZED_UINT8
    else
        tflite_convert \
            --graph_def_file="${OUTPUT_DIR}/tflite_graph.pb" \
            --output_file="${OUTPUT_DIR}/detect.tflite" \
            --input_shapes="1,${INPUT_SIZE},${INPUT_SIZE},3" \
            --input_arrays=normalized_input_image_tensor \
            --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
            --allow_custom_ops \
            --mean_values=128 \
            --std_dev_values=128 \
            --default_ranges_min=0 \
            --default_ranges_max=6 \
            --post_training_quantize
    fi
else
    start_task "Generating NON-quantized tflite frozen graph for model ${MODEL_NAME}..."
    tflite_convert \
        --graph_def_file="${OUTPUT_DIR}/tflite_graph.pb" \
        --output_file="${OUTPUT_DIR}/detect.tflite" \
        --output_format=TFLITE \
        --input_shapes="1,${INPUT_SIZE},${INPUT_SIZE},3" \
        --input_arrays=normalized_input_image_tensor \
        --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
        --allow_custom_ops \
        --mean_values=128 \
        --std_dev_values=127
fi

##
##
start_task "Copy to android project for model ${MODEL_NAME}..."
mkdir -p "${ANDROID_ASSET_PATH}"
cp "${OUTPUT_DIR}/detect.tflite" "${ANDROID_ASSET_PATH}"
cp "../../data/${DS_NAME}-tfrecords/labelmap.txt" "${ANDROID_ASSET_PATH}"