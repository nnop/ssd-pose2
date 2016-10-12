#!/bin/bash
<<<<<<< HEAD

./build/tools/caffe train \
  --solver=examples/mnist/mnist_autoencoder_solver_nesterov.prototxt
=======
set -e

./build/tools/caffe train \
  --solver=examples/mnist/mnist_autoencoder_solver_nesterov.prototxt $@
>>>>>>> 38a20293b36d973eb72e4d1d4737d43aa8a9e0be
