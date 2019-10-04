# Standard Library imports:
import argparse
import os
from pathlib import Path
from typing import Dict, List

# 3rd Party imports:
import keras.backend as K
from keras.layers import *
from keras.models import Model
import tensorflow as tf
from tensorflow.python.framework import graph_io, graph_util
from tensorflow.python.tools import import_pb_to_tensorboard


def keras_to_tensorflow(
    keras_model,
    output_dir: Path,
    model_name,
    out_prefix="output_",
    log_tensorboard=True,
):
    """Convert from keras to tf"""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_dir: str = str(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(keras_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()

    init_graph = sess.graph.as_graph_def()

    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    if log_tensorboard:
        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name), output_dir
        )


"""
We explicitly redefine the SqueezNet architecture since Keras has no predefined
SqueezNet
"""


def squeezenet_fire_module(input, input_channel_small=16, input_channel_large=64):

    channel_axis = 3

    input = Conv2D(input_channel_small, (1, 1), padding="valid")(input)
    input = Activation("relu")(input)

    input_branch_1 = Conv2D(input_channel_large, (1, 1), padding="valid")(input)
    input_branch_1 = Activation("relu")(input_branch_1)

    input_branch_2 = Conv2D(input_channel_large, (3, 3), padding="same")(input)
    input_branch_2 = Activation("relu")(input_branch_2)

    input = concatenate([input_branch_1, input_branch_2], axis=channel_axis)

    return input


def SqueezeNet(input_shape=(224, 224, 3)):
    """Returns a new keras SqueezeNet model"""
    image_input = Input(shape=input_shape)

    network = Conv2D(64, (3, 3), strides=(2, 2), padding="valid")(image_input)
    network = Activation("relu")(network)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(
        input=network, input_channel_small=16, input_channel_large=64
    )
    network = squeezenet_fire_module(
        input=network, input_channel_small=16, input_channel_large=64
    )
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(
        input=network, input_channel_small=32, input_channel_large=128
    )
    network = squeezenet_fire_module(
        input=network, input_channel_small=32, input_channel_large=128
    )
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(
        input=network, input_channel_small=48, input_channel_large=192
    )
    network = squeezenet_fire_module(
        input=network, input_channel_small=48, input_channel_large=192
    )
    network = squeezenet_fire_module(
        input=network, input_channel_small=64, input_channel_large=256
    )
    network = squeezenet_fire_module(
        input=network, input_channel_small=64, input_channel_large=256
    )

    # Remove layers like Dropout and BatchNormalization, they are only needed in training
    # network = Dropout(0.5)(network)

    network = Conv2D(1000, kernel_size=(1, 1), padding="valid", name="last_conv")(
        network
    )
    network = Activation("relu")(network)

    network = GlobalAvgPool2D()(network)
    network = Activation("softmax", name="output")(network)

    input_image = image_input
    model = Model(inputs=input_image, outputs=network)

    return model


def get_tf_filename(keras_filename) -> str:
    return keras_filename.replace(".h5", ".pb")


def main(opt):
    """Convert a model from keras to tensorflow lite."""
    weights_path: Path = Path("../weights")
    model_path = weights_path / opt.model_path
    if not model_path.exists():
        raise ValueError(f"Invalid model path: {model_path}")

    print(f"Loading keras model: '{model_path}'")
    keras_model = SqueezeNet()
    keras_model.load_weights(model_path)
    output_file = get_tf_filename(str(model_path))
    keras_to_tensorflow(keras_model, output_dir=weights_path, model_name=output_file)
    print("MODEL SAVED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="squeezenet.h5",
        help="filename of model to convert. Path should be relative to the ./training/models/ folder",
    )
    opt = parser.parse_args()
    main(opt)
