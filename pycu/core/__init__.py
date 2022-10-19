from ..driver import *
from ..nvrtc import *
from ..nvvm import *
from ..driver.core import *
from ..nvrtc.core import NVRTCPtr, NVRTC
from ..nvvm.core import NVVMPtr, NVVM
from ..config import auto_driver_init, auto_context_init, include_numba_extentions

from .jitify import Jitify
from .compiler import compile_source, compile_file, jitify

#should be done after entirety of core has been imported
if include_numba_extentions:
	from .numba_extension import *
	#need to import the mathfunc stubs into the pycu namespace so they are usable
	from .numba_extension.mathfuncs import *
	from .numba_extension.vector.vectorfuncs import *


def div_ru(x, y):
	return (x + y - 1)//y


# https://gamedevelopment.tutsplus.com/articles/forward-rendering-vs-deferred-rendering--gamedev-12342
# https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/
# https://github.com/aman-tiwari/MeshToSDF


#https://stackoverflow.com/questions/2831934/how-to-use-if-inside-define-in-the-c-preprocessor








# /usr/local/cuda-10.0/lib64
# PATH:/usr/local/cuda-10.0/bin


#Rendering
	#https://computergraphics.stackexchange.com/questions/1768/how-can-virtual-texturing-actually-be-efficient

#Numerical methods for PDEs
	#https://mediaspace.illinois.edu/channel/channelid/148703211
	#Andreas Kloeckner



#https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5

#enums in python
	#https://v4.chriskrycho.com/2015/ctypes-structures-and-dll-exports.html
	#https://github.com/numba/numba/blob/master/numba/cuda/cudadrv/enums.py
		#https://github.com/numba/numba/blob/master/numba/cuda/cudadrv/drvapi.py

#https://cuda_d.dpldocs.info/source/cuda_d.cuda.d.html#L326
#http://manpages.ubuntu.com/manpages/eoan/man3/CUDA_TYPES.3.html





#https://github.com/apache/arrow/blob/master/python/pyarrow/includes/libarrow_cuda.pxd
#https://github.com/apache/arrow/blob/master/python/pyarrow/_cuda.pxd
#https://github.com/apache/arrow/blob/master/python/pyarrow/_cuda.pyx
#https://github.com/apache/arrow/blob/master/cpp/src/arrow/gpu/cuda_memory.h
#https://github.com/apache/arrow/blob/master/python/pyarrow/includes/libarrow_cuda.pxd

#https://github.com/pytorch/pytorch/blob/1f09f7ea448f404fff135b5e4c1c3d63d157de79/torch/cuda/__init__.py
#https://github.com/pytorch/pytorch/blob/master/torch/csrc/cuda/Module.cpp
#https://github.com/pytorch/pytorch/blob/master/torch/cuda/streams.py

#move dict to different class
#create variables for each dict entry
#have dict pull thes with the getattr






# https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi


# PATH /usr/local/cuda-10.0/bin
# LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64

