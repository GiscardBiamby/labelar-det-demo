# Setup

Setups up conda environment and downloads pretrained weights, etc:

``` bash
PROJ_ROOT=path/to/this/github/repo
cd "${PROJ_ROOT}"/setup
./setup.sh
```

# Training
See [./training/README.md](./training/README.md) for training instructions.

We have two models and best so far is ssd_mobilenet_v2_quant_bidmugs: mAP50=0.670, runs at ~35fps on the Google PX3L.

Also have trained and run a bidmugs model on mobilenet_v1_quant_bidmugs. AP was low because that test was just to see if the pipeline was working.

