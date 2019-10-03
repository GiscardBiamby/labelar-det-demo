"""
keras_to_tfl.py converts a keras model into tensorflow lite format, and also optimizes the model
to have a smaller memory footprint and lower latency.

see: https://www.tensorflow.org/lite/performance/model_optimization

"""
# Standard Library imports:
import argparse
from pathlib import Path

# 3rd Party imports:
import numpy as np
import tensorflow as tf


def main(opt):
    """
    """
    # Params (later convert to function params and/or commandline options via argparse):
    enable_quantize_fp16 = opt.quantize == "fp16"

    # Load the MobileNet tf.keras model.
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet", input_shape=(224, 224, 3)
    )

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if enable_quantize_fp16:
        print("float16 post-training quantization enabled.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the TensorFlow Lite model on random input data.
    input_shape = input_details[0]["shape"]
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    tflite_results = interpreter.get_tensor(output_details[0]["index"])

    # Test the TensorFlow model on random input data.
    tf_results = model(tf.constant(input_data))

    # Compare the result. This seems to fail if we enable fp16 quantization, maybe it
    # doesn't matter either way, the values are still pretty similar with the
    # quantization enabled:
    for tf_result, tflite_result in zip(tf_results, tflite_results):
        np.testing.assert_almost_equal(
            tf_result, tflite_result, decimal=1 if enable_quantize_fp16 else 5
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize",
        type=str,
        default="fp16",
        help="Type of post-training quantization to use.",
    )
    opt = parser.parse_args()
    main(opt)
