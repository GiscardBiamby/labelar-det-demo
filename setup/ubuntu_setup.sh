###
### General Dependencies:
# Used in setup.sh to unzip pretrained models that are downloaded from web:
sudo apt-get install zip

###
### protocol-buffers compiler, used by tensorflow object detection (e.g., ./training/tfmodels/)
sudo apt-get install protobuf-compiler
# See here if you have trouble with it, or for instructions to install on macOS: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

###
### Android setup
# You only have to care about this section if you want to deploy to android from your
# computer:

# If you do want to deploy to android, follow these instructions:
#   https://developer.android.com/studio/run/device.html

#This page has info on how to configure the phone to enable usb debugging, which is
#   required if you want to deploy to the phone:
#   https://developer.android.com/studio/debug/dev-options.html

sudo apt-get install adb

## Ensure NDK is installed:
## Open the android project in android studio, goto File->Project Structure->Download NDK



## (Old, will delete this soon):
# ##
# ## Bazel setup
# ## (only need if you are converting tensorflow models to tflite)
# # These setup commands were adapted from installation instructions here:
# #   https://docs.bazel.build/versions/master/install-ubuntu.html
# sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
# wget https://github.com/bazelbuild/bazel/releases/download/1.0.0/bazel-1.0.0-installer-linux-x86_64.sh
# chmod +x bazel-1.0.0-installer-linux-x86_64.sh
# ./bazel-1.0.0-installer-linux-x86_64.sh --user

# # Might be good idea to add this to your ~/.bashrc:
# # Note we have init_env.sh in the root of our repo, which performs this
# # export along with other exports and environment init commands. So the typical workflow
# # for using this repo would be to run 'source ./init_env.sh' from the root of the repo
# # before you do anything from the command line.
# export PATH="$PATH:$HOME/bin"

