#!/usr/bin/env sh
<<<<<<< HEAD

./build/tools/caffe train --solver=examples/mnist/lenet_solver_rmsprop.prototxt
=======
set -e

./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_rmsprop.prototxt $@
>>>>>>> 38a20293b36d973eb72e4d1d4737d43aa8a9e0be
