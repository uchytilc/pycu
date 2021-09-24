from pycu.driver import *
from pycu.nvrtc import *
from pycu.nvvm import *
from pycu.driver.core import *
from pycu.nvrtc.core import NVRTCPtr, NVRTC
from pycu.nvvm.core import NVVMPtr, NVVM

# from pycu.driver import (init, stream, opengl_resource, to_device, to_host, copy_to_device, copy_to_host, to_device_buffer,
# 						 device_array, device_ndarray, device_buffer, pinned_buffer, pin_host_array, ContextManager)

context_manager = ContextManager()

def get_current_context():
	return context_manager.get_current()

def retain_primary_context(dev, flags = None):
	return context_manager.retain_primary(dev, flags)

def create_context(dev, flags = None):
	return context_manager.create_context(dev, flags)

def push_context(ctx):
	context_manager.push(ctx)

def pop_context(ctx):
	return context_manager.pop(ctx)

def set_context(ctx):
	return context_manager.set_current(ctx)

def add_module(module):
	context_manager.add_module(module)

def synchronize():
	context_manager.synchronize()

auto_driver_init = True
auto_context_init = True

if auto_driver_init:
	init()
	if auto_context_init:
		pctx = context_manager.retain_primary(0)
		context_manager.set_current(pctx)

#compiler needs the context_manager defined above
from .jitify import Jitify
from .compiler import compile_source, compile_file, jitify





# from .py_allocator import py_malloc as malloc, py_free as free


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


