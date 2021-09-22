#!/bin/bash

# start jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# mkdirs
mkdir runs

# start tensorboard
tensorboard --logdir=runs
