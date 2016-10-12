#!/usr/bin/env sh
<<<<<<< HEAD

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt
=======
set -e

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt $@
>>>>>>> 38a20293b36d973eb72e4d1d4737d43aa8a9e0be
