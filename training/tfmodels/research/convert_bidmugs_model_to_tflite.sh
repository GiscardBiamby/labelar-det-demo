#!/bin/bash

export CONFIG_FILE=./object_detection/samples/configs/ssd_mobilenet_v2_bidmugs.config
export CHECKPOINT_PATH=../../weights/ssd_mobilenet_v2_bidmugs/model.ckpt-50000
export OUTPUT_DIR=../../weights/ssd_mobilenet_v2_bidmugs/tflite/

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Creating directory: '${OUTPUT_DIR}''"
    mkdir -p "${OUTPUT_DIR}"
fi



# Export tflite frozen graph (e.g., make tflite_graph.ph):
# python object_detection/export_tflite_ssd_graph.py \
#     --pipeline_config_path=$CONFIG_FILE \
#     --trained_checkpoint_prefix=$CHECKPOINT_PATH \
#     --output_directory=$OUTPUT_DIR \
#     --add_postprocessing_op=true

tflite_convert \
    --output_file="${OUTPUT_DIR}detect.tflite" \
    --graph_def_file="${OUTPUT_DIR}tflite_graph.pb" \
    --input_shapes=1,320,320,3 \
    --input_arrays=image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --allow_custom_ops


# The below Bazel command doesn't work... but if the tflite_convert command works above, then we won't need
# this (we only need one or the bother, they should both accomplish the same thing):

# Use Bazel + Tensorflow Optimizing Converter
# (TOCO; https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/toco)
# to convert frozen graph (tflite_graph.ph) to TensorFlow Lite flatbuffer format
# (detect.tflite):
# cd ../../tensorflow/tensorflow
# echo "Current dir: "
# echo `pwd`
# bazel run -c opt tensorflow/lite/toco:toco -- \
#     --input_file=$OUTPUT_DIR/tflite_graph.pb \
#     --output_file=$OUTPUT_DIR/detect.tflite \
#     --input_shapes=1,300,300,3 \
#     --input_arrays=normalized_input_image_tensor \
#     --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
#     --inference_type=QUANTIZED_UINT8 \
#     --mean_values=128 \
#     --std_values=128 \
#     --change_concat_input_ranges=false \
#     --allow_custom_ops
# cd -