# Standard Library imports:
import argparse
from pathlib import Path
from typing import Dict

# 3rd Party imports:
import keras.backend as K
from keras.layers import *
from keras.models import *
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import squeezenet1_1


class PytorchToKeras(object):
    def __init__(self, pModel, kModel):
        super(PytorchToKeras, self)
        self.__source_layers = []
        self.__target_layers = []
        self.pModel = pModel
        self.kModel = kModel

        K.set_learning_phase(0)

    def __retrieve_k_layers(self):

        for i, layer in enumerate(self.kModel.layers):
            if len(layer.weights) > 0:
                self.__target_layers.append(i)

    def __retrieve_p_layers(self, input_size):

        input = torch.randn(input_size)

        input = Variable(input.unsqueeze(0))

        hooks = []

        def add_hooks(module):
            def hook(module, input, output):
                if hasattr(module, "weight"):
                    self.__source_layers.append(module)

            if (
                not isinstance(module, nn.ModuleList)
                and not isinstance(module, nn.Sequential)
                and module != self.pModel
            ):
                hooks.append(module.register_forward_hook(hook))

        self.pModel.apply(add_hooks)

        self.pModel(input)
        for hook in hooks:
            hook.remove()

    def convert(self, input_size):
        self.__retrieve_k_layers()
        self.__retrieve_p_layers(input_size)

        for i, (source_layer, target_layer) in enumerate(
            zip(self.__source_layers, self.__target_layers)
        ):

            weight_size = len(source_layer.weight.data.size())

            transpose_dims = []

            for i in range(weight_size):
                transpose_dims.append(weight_size - i - 1)

            self.kModel.layers[target_layer].set_weights(
                [
                    source_layer.weight.data.numpy().transpose(transpose_dims),
                    source_layer.bias.data.numpy(),
                ]
            )

    def save_model(self, output_file):
        self.kModel.save(output_file)

    def save_weights(self, output_file):
        self.kModel.save_weights(output_file)


"""
We explicitly redefine the Squeezenet architecture since Keras has no predefined Squeezenet
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


def main(opt):
    weights_path = Path("./weights")
    model_path = weights_path / Path(opt.model_path)
    if not model_path.exists():
        raise ValueError(f"Invalid model path: {model_path}")

    # PyTorch includes a predefined Squeezenet
    pytorch_model = squeezenet1_1()
    print(f"Loading pytorch model: '{model_path}'")
    pytorch_model.load_state_dict(torch.load(model_path))

    # Convert:
    keras_model = SqueezeNet()
    converter = PytorchToKeras(pytorch_model, keras_model)
    converter.convert((3, 224, 224))

    # Save the weights of the converted keras model for later use
    converter.save_weights(weights_path / "squeezenet.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained/squeezenet1_1-f364aa15.pth",
        help="filename of model to convert",
    )
    opt = parser.parse_args()
    main(opt)