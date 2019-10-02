# Setup

Setups up conda environment and downloads pretrained weights, etc:

```bash
PROJ_ROOT=path/to/this/github/repo
cd "${PROJ_ROOT}"/setup
./setup.sh
```

# Weight conversion:

Right now all we have is a convert.py script that converts squeezenet from pytorch to keras format. But we're building on top of this with the goal to soon be able to convert multiple pytorch models into the right format(s) and deploy them to android/iOS.

Android:
(./training/scripts/pytorch_to_keras.py)
pytorch -> keras -> tensorflow lite

iOS:
pytorch -> keras -> CoreML

We'll want to try also running SSD/MobileNet, and maybe CenterNet. Centernet would be interesting if it can work on mobile because the paper's main claim to fame is a better tradeoff in terms of FPS vs. detection performance compared to pretty much all existing single and two stage detectors (Faster-RCNN, YOLO, SSD, etc). But the question is does CenterNet require any operations that tensorflow lite cannot yet accelerate on mobile? And how much work to define a keras model for Centernet? etc.
