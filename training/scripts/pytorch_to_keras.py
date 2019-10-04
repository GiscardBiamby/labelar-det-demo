# Standard Library imports:
import argparse
from pathlib import Path
import sys

# 3rd Party imports:
import keras.backend as K
from keras.layers import *
from keras.models import *
import numpy as np
from pytorch2keras import pytorch_to_keras
import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as torch_models


# class PytorchToKeras(object):
#     def __init__(self, pModel, kModel):
#         super(PytorchToKeras, self)
#         self.__source_layers = []
#         self.__target_layers = []
#         self.pModel = pModel
#         self.kModel = kModel

#         K.set_learning_phase(0)

#     def __retrieve_k_layers(self):
#         for i, layer in enumerate(self.kModel.layers):
#             if len(layer.weights) > 0:
#                 self.__target_layers.append(i)

#     def __retrieve_p_layers(self, input_size):

#         input = torch.randn(input_size)
#         input = Variable(input.unsqueeze(0))
#         hooks = []

#         def add_hooks(module):
#             def hook(module, input, output):
#                 if hasattr(module, "weight"):
#                     self.__source_layers.append(module)

#             if (
#                 not isinstance(module, nn.ModuleList)
#                 and not isinstance(module, nn.Sequential)
#                 and module != self.pModel
#             ):
#                 hooks.append(module.register_forward_hook(hook))

#         self.pModel.apply(add_hooks)
#         self.pModel(input)
#         for hook in hooks:
#             hook.remove()

#     def convert(self, input_size):
#         self.__retrieve_k_layers()
#         self.__retrieve_p_layers(input_size)
#         for i, (source_layer, target_layer) in enumerate(
#             zip(self.__source_layers, self.__target_layers)
#         ):
#             print(dir(source_layer))
#             print(
#                 f"i: {i}, source_layer: {source_layer}, target_layer: {target_layer}, {self.kModel.layers[target_layer]}"
#             )
#             weight_size = len(source_layer.weight.data.size())
#             transpose_dims = []
#             for i in range(weight_size):
#                 transpose_dims.append(weight_size - i - 1)
#             if source_layer.bias is None:
#                 self.kModel.layers[target_layer].set_weights(
#                     [source_layer.weight.data.numpy().transpose(transpose_dims)]
#                 )
#             else:
#                 self.kModel.layers[target_layer].set_weights(
#                     [
#                         source_layer.weight.data.numpy().transpose(transpose_dims),
#                         source_layer.bias.data.numpy(),
#                     ]
#                 )

#     def save_model(self, output_file):
#         self.kModel.save(output_file)

#     def save_weights(self, output_file):
#         self.kModel.save_weights(output_file)


# """
# We explicitly redefine the Squeezenet architecture since Keras has no predefined SqueezeNet
# """


# def squeezenet_fire_module(input, input_channel_small=16, input_channel_large=64):
#     channel_axis = 3
#     input = Conv2D(input_channel_small, (1, 1), padding="valid")(input)
#     input = Activation("relu")(input)
#     input_branch_1 = Conv2D(input_channel_large, (1, 1), padding="valid")(input)
#     input_branch_1 = Activation("relu")(input_branch_1)
#     input_branch_2 = Conv2D(input_channel_large, (3, 3), padding="same")(input)
#     input_branch_2 = Activation("relu")(input_branch_2)
#     input = concatenate([input_branch_1, input_branch_2], axis=channel_axis)
#     return input


# def SqueezeNet(input_shape=(224, 224, 3)):
#     image_input = Input(shape=input_shape)
#     network = Conv2D(64, (3, 3), strides=(2, 2), padding="valid")(image_input)
#     network = Activation("relu")(network)
#     network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)
#     network = squeezenet_fire_module(
#         input=network, input_channel_small=16, input_channel_large=64
#     )
#     network = squeezenet_fire_module(
#         input=network, input_channel_small=16, input_channel_large=64
#     )
#     network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)
#     network = squeezenet_fire_module(
#         input=network, input_channel_small=32, input_channel_large=128
#     )
#     network = squeezenet_fire_module(
#         input=network, input_channel_small=32, input_channel_large=128
#     )
#     network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)
#     network = squeezenet_fire_module(
#         input=network, input_channel_small=48, input_channel_large=192
#     )
#     network = squeezenet_fire_module(
#         input=network, input_channel_small=48, input_channel_large=192
#     )
#     network = squeezenet_fire_module(
#         input=network, input_channel_small=64, input_channel_large=256
#     )
#     network = squeezenet_fire_module(
#         input=network, input_channel_small=64, input_channel_large=256
#     )
#     # Remove layers like Dropout and BatchNormalization, they are only needed in training
#     # network = Dropout(0.5)(network)
#     network = Conv2D(1000, kernel_size=(1, 1), padding="valid", name="last_conv")(
#         network
#     )
#     network = Activation("relu")(network)
#     network = GlobalAvgPool2D()(network)
#     network = Activation("softmax", name="output")(network)
#     input_image = image_input
#     model = Model(inputs=input_image, outputs=network)

#     return model


def torch_zoo_mobilenetv1ssd(opt):
    url = "https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth"
    model = torch.utils.model_zoo.load_url(
        url, model_dir=opt.weights_path, progress=True
    )
    return model


def convert_to_keras(opt, pytorch_model: nn.Module):
    num_channels = 3
    # Create dummy variable with correct shape:
    input_np = np.random.uniform(
        0, 1, (1, num_channels, opt.input_size, opt.input_size)
    )
    input_var = Variable(torch.FloatTensor(input_np))

    keras_model = pytorch_to_keras(
        pytorch_model,
        input_var,
        (num_channels, opt.input_size, opt.input_size),
        verbose=opt.verbose,
    )

    if opt.compare_outputs:
        pytorch_output = pytorch_model(input_var).data.numpy()
        keras_output = keras_model.predict(input_np)
        error = np.max(pytorch_output - keras_output)
        print(f"pytorch_output - keras_output: {error}")
    return keras_model


def create_pytorch_model(opt):
    model_factories = {
        "squeezenet": torch_models.squeezenet1_1,
        "mobilenetv2": torch_models.mobilenet_v2,
    }
    pytorch_model = model_factories[opt.arch](pretrained=opt.pretrained)
    print(f"loaded pytorch model: {opt.arch}, pretrained: {opt.pretrained}")
    # Set model to eval mode:
    pytorch_model.eval()
    for m in pytorch_model.modules():
        m: nn.Module = m
        m.training = False
    # print(pytorch_model)
    return pytorch_model


# Old method, required model matching model definitions in both keras and pytorch format
# def create_keras_model(opt, pytorch_model):
#     model_factories = {
#         "squeezenet": SqueezeNet,
#         "mobilenetv2": tf.keras.applications.MobileNetV2,
#     }
#     keras_model = model_factories[opt.arch]()
#     print(f"loaded keras model: {type(keras_model)}")
#     print(keras_model.summary())
#     return keras_model


def main(opt):
    pytorch_model = create_pytorch_model(opt)
    if not opt.pretrained:
        print(f"Loading pytorch model: '{opt.model_path}'")
        pytorch_model.load_state_dict(torch.load(opt.model_path))


    # Convert:
    #
    # Old method, required model matching model definitions in both keras and pytorch
    # format:
    # keras_model = create_keras_model(opt)
    # input_dim = (3, opt.input_size, opt.input_size)
    # converter = PytorchToKeras(pytorch_model, keras_model)
    # converter.convert(input_dim)
    # converter.save_weights(opt.output_path)
    #
    # New Method, use pytorch2keras library to create keras model on-the-fly:
    keras_model = convert_to_keras(opt, pytorch_model)

    # Save the weights of the converted keras model for later use
    print(f"Saving keras formatted weights to {opt.output_path}")
    keras_model.save(opt.output_path)


class opts(object):
    """
    Handle parsing of command line args.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--model_path",
            type=str,
            default="pretrained/squeezenet1_1-f364aa15.pth",
            help="filename of model to convert",
        )
        self.parser.add_argument(
            "--pretrained",
            action="store_true",
            help="True = use a pretrained model (in which case there is no need to specify model_path)",
        )
        self.parser.add_argument(
            "--arch", type=str, default="squeezenet", help="squeezenet | mobilenetv2"
        )
        self.parser.add_argument(
            "--input_size", type=int, default=224, help="input dimensions for CNN"
        )
        self.parser.add_argument(
            "--pretrain_dataset",
            type=str,
            default="imagenet",
            help="imagenet (which dataset was the pretrained model trained on)",
        )
        self.parser.add_argument(
            "--output_file",
            type=str,
            help="file name. If not specified, and if model_path *is* specified, then this is automatically derrived from model_path. Otherwise this parameter is required.",
        )
        self.parser.add_argument(
            "--compare_outputs",
            type=bool,
            default=True,
            help="If true, a random tensor input is passed through the pytorch and keras models, and the difference in output is compared (we expect the difference to be very small)",
        )
        self.parser.add_argument(
            "--verbose", action="store_true"
        )

    def parse(self, args=""):
        """
        Sets up the command line params, checks for validity, converts options to types
        that are easier to use later on in the code, fills in default values, etc.
        """
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        # weights_path:
        opt.weights_path = Path("../weights")
        if opt.pretrained:
            print("")
        else:
            # model_path:
            model_path = opt.weights_path / Path(opt.model_path)
            if not model_path.exists():
                raise ValueError(f"Invalid model path: {model_path}")
            opt.model_path: Path = model_path
        # output_path:
        if opt.pretrained:
            if opt.output_file is None:
                raise ValueError("output_file is required if --pretrained is set")
            else:
                opt.output_path = opt.weights_path / opt.output_file
        else:
            if opt.output_file is None:
                opt.output_path = Path(
                    str(opt.model_path.resolve()).replace(".pth", ".h5")
                )
            else:
                opt.output_path = opt.weights_path / opt.output_file

        return opt


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)
