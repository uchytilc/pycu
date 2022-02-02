from ctypes import *
from .enums import CUresult

# import sys

# #https://docs.python.org/3/library/platform.html#cross-platform
# CUdeviceptr           = c_uint
# if sys.maxsize > 2**32: #64 bit architecture
# 	CUdeviceptr       = c_ulonglong

# #https://github.com/numba/numba/blob/d7953a18dbf5ea231dc16e967ce8e9b754578ea6/numba/cuda/tests/cudadrv/test_nvvm_driver.py
#is64bit = sizeof(c_size_t) == sizeof(c_uint64)

CUarray               = c_void_p
CUcontext             = c_void_p #all c_void_p's are opaque handles
CUdevice              = c_int
CUdeviceptr           = c_size_t #typedef unsigned int on 32-bit and typedef unsigned long long on 64-bit machine
# CUeglStreamConnection = c_void_p
CUevent               = c_void_p
CUexternalMemory      = c_void_p
CUexternalSemaphore   = c_void_p
CUfunction            = c_void_p
CUgraph               = c_void_p
CUgraphExec           = c_void_p
CUgraphNode           = c_void_p
CUgraphicsResource    = c_void_p
CUhostFn              = CFUNCTYPE(None, py_object) #c_void_p
CUmemoryPool          = c_void_p
CUmipmappedArray      = c_void_p
CUmodule              = c_void_p
CUoccupancyB2DSize    = CFUNCTYPE(c_size_t, c_int)
CUstream              = c_void_p
CUstreamCallback      = CFUNCTYPE(None, CUstream, CUresult, py_object) #c_void_p
CUsurfObject          = c_ulonglong
CUsurfref             = c_void_p
CUtexObject           = c_ulonglong
CUtexref              = c_void_p

CUlinkState = c_void_p

cuuint32_t = c_uint32
cuuint64_t = c_uint64
# #ifdef _MSC_VER
# typedef unsigned __int32 cuuint32_t;
# typedef unsigned __int64 cuuint64_t;
# #else
# typedef uint32_t cuuint32_t;
# typedef uint64_t cuuint64_t;
# #endif

try:
	from OpenGL.raw.GL import _types
	GLuint = _types.GLuint
	GLenum = _types.GLenum
except:
	GLuint = c_uint
	GLenum = c_uint
