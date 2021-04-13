from ctypes import c_int

enum = c_int

nvrtcResult = enum
NVRTC_SUCCESS                                     = nvrtcResult(0)
NVRTC_ERROR_OUT_OF_MEMORY                         = nvrtcResult(1)
NVRTC_ERROR_PROGRAM_CREATION_FAILURE              = nvrtcResult(2)
NVRTC_ERROR_INVALID_INPUT                         = nvrtcResult(3)
NVRTC_ERROR_INVALID_PROGRAM                       = nvrtcResult(4)
NVRTC_ERROR_INVALID_OPTION                        = nvrtcResult(5)
NVRTC_ERROR_COMPILATION                           = nvrtcResult(6)
NVRTC_ERROR_BUILTIN_OPERATION_FAILURE             = nvrtcResult(7)
NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = nvrtcResult(8)
NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION   = nvrtcResult(9)
NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID             = nvrtcResult(10)
NVRTC_ERROR_INTERNAL_ERROR                        = nvrtcResult(11)
