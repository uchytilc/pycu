from ctypes import *

enum = c_int

nvvmResult = enum
NVVM_SUCCESS                        = nvvmResult(0)
NVVM_ERROR_OUT_OF_MEMORY            = nvvmResult(1)
NVVM_ERROR_PROGRAM_CREATION_FAILURE = nvvmResult(2)
NVVM_ERROR_IR_VERSION_MISMATCH      = nvvmResult(3)
NVVM_ERROR_INVALID_INPUT            = nvvmResult(4)
NVVM_ERROR_INVALID_PROGRAM          = nvvmResult(5)
NVVM_ERROR_INVALID_IR               = nvvmResult(6)
NVVM_ERROR_INVALID_OPTION           = nvvmResult(7)
NVVM_ERROR_NO_MODULE_IN_PROGRAM     = nvvmResult(8)
NVVM_ERROR_COMPILATION              = nvvmResult(9)
