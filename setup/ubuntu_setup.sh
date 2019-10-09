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
