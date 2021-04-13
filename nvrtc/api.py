from .enums import nvrtcResult
from .typedefs import nvrtcProgram

from ctypes import *

API_PROTOTYPES = {
	# nvrtcResult nvrtcVersion ( int* major, int* minor )
	'nvrtcVersion': (nvrtcResult, POINTER(c_int), POINTER(c_int)),
	# nvrtcResult nvrtcAddNameExpression ( nvrtcProgram prog, const char* name_expression )
	'nvrtcAddNameExpression': (nvrtcResult, nvrtcProgram, c_char_p),
	# nvrtcResult nvrtcCompileProgram ( nvrtcProgram prog, int  numOptions, const char** options )
	'nvrtcCompileProgram': (nvrtcResult, nvrtcProgram, c_int, POINTER(c_char_p)),
	# nvrtcResult nvrtcCreateProgram ( nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames )
	'nvrtcCreateProgram': (nvrtcResult, POINTER(nvrtcProgram), c_char_p, c_char_p, c_int, POINTER(c_char_p), POINTER(c_char_p)),
	# nvrtcResult nvrtcDestroyProgram ( nvrtcProgram* prog )
	'nvrtcDestroyProgram': (nvrtcResult, POINTER(nvrtcProgram)),
	# nvrtcResult nvrtcGetLoweredName ( nvrtcProgram prog, const char* name_expression, const char** lowered_name )
	'nvrtcGetLoweredName': (nvrtcResult, nvrtcProgram, c_char_p, POINTER(c_char_p)),
	# nvrtcResult nvrtcGetPTX ( nvrtcProgram prog, char* ptx )
	'nvrtcGetPTX': (nvrtcResult, nvrtcProgram, c_char_p),
	# nvrtcResult nvrtcGetPTXSize ( nvrtcProgram prog, size_t* ptxSizeRet )
	'nvrtcGetPTXSize': (nvrtcResult, nvrtcProgram, POINTER(c_size_t)),
	# nvrtcResult nvrtcGetProgramLog ( nvrtcProgram prog, char* log )
	'nvrtcGetProgramLog': (nvrtcResult, nvrtcProgram, c_char_p),
	# nvrtcResult nvrtcGetProgramLogSize ( nvrtcProgram prog, size_t* logSizeRet )
	'nvrtcGetProgramLogSize': (nvrtcResult, nvrtcProgram, POINTER(c_size_t))
}
