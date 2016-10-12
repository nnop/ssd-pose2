from .pycaffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver
<<<<<<< HEAD
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list
=======
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list, set_random_seed
>>>>>>> 38a20293b36d973eb72e4d1d4737d43aa8a9e0be
from ._caffe import __version__
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto
