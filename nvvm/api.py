from .enums import nvvmResult
from .typedefs import nvvmProgram

from ctypes import *

API_PROTOTYPES = {
	#const char* nvvmGetErrorString ( nvvmResult result )
	"nvvmGetErrorString":(c_char_p, nvvmResult),
	#nvvmResult nvvmIRVersion ( int* majorIR, int* minorIR, int* majorDbg, int* minorDbg )
	"nvvmIRVersion":(nvvmResult, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)),
	#nvvmResult nvvmVersion ( int* major, int* minor )
	"nvvmVersion":(nvvmResult, POINTER(c_int), POINTER(c_int)),
	# nvvmResult nvvmAddModuleToProgram ( nvvmProgram prog, const char* buffer, size_t size, const char* name )
	"nvvmAddModuleToProgram":(nvvmResult, nvvmProgram, c_char_p, c_size_t, c_char_p),
	# nvvmResult nvvmCompileProgram ( nvvmProgram prog, int  numOptions, const char** options )
	"nvvmCompileProgram":(nvvmResult, nvvmProgram, c_int, POINTER(c_char_p)),
	# nvvmResult nvvmCreateProgram ( nvvmProgram* prog )
	"nvvmCreateProgram":(nvvmResult, POINTER(nvvmProgram)),
	# nvvmResult nvvmDestroyProgram ( nvvmProgram* prog )
	"nvvmDestroyProgram":(nvvmResult, POINTER(nvvmProgram)),
	# nvvmResult nvvmGetCompiledResult ( nvvmProgram prog, char* buffer )
	"nvvmGetCompiledResult":(nvvmResult, nvvmProgram, c_char_p),
	# nvvmResult nvvmGetCompiledResultSize ( nvvmProgram prog, size_t* bufferSizeRet )
	"nvvmGetCompiledResultSize":(nvvmResult, nvvmProgram, POINTER(c_size_t)),
	# nvvmResult nvvmGetProgramLog ( nvvmProgram prog, char* buffer )
	"nvvmGetProgramLog":(nvvmResult, nvvmProgram, c_char_p),
	# nvvmResult nvvmGetProgramLogSize ( nvvmProgram prog, size_t* bufferSizeRet )
	"nvvmGetProgramLogSize":(nvvmResult, nvvmProgram, POINTER(c_size_t)),
	# nvvmResult nvvmLazyAddModuleToProgram ( nvvmProgram prog, const char* buffer, size_t size, const char* name )
	"nvvmLazyAddModuleToProgram":(nvvmResult, nvvmProgram, c_char_p, c_size_t, c_char_p),
	# nvvmResult nvvmVerifyProgram ( nvvmProgram prog, int  numOptions, const char** options )
	"nvvmVerifyProgram":(nvvmResult, nvvmProgram, c_int, POINTER(c_char_p))
}
