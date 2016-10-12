#!/usr/bin/env sh
<<<<<<< HEAD

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt \
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate.h5
=======
set -e

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt \
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate.h5 \
    $@
>>>>>>> 38a20293b36d973eb72e4d1d4737d43aa8a9e0be
