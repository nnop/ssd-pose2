#!/usr/bin/env sh
<<<<<<< HEAD
=======
set -e
>>>>>>> 38a20293b36d973eb72e4d1d4737d43aa8a9e0be

TOOLS=./build/tools

$TOOLS/caffe train \
<<<<<<< HEAD
    --solver=examples/cifar10/cifar10_full_sigmoid_solver.prototxt
=======
    --solver=examples/cifar10/cifar10_full_sigmoid_solver.prototxt $@
>>>>>>> 38a20293b36d973eb72e4d1d4737d43aa8a9e0be

