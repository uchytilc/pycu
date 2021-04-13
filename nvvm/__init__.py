from .driver import Driver, get_libdevice, get_closest_arch
from .api import API_PROTOTYPES
# from .enums import *
from .typedefs import nvvmProgram
from .error import check_nvvm_error

from ctypes import byref, c_char, c_int, c_size_t, c_char_p

nvvm = Driver()
# NVVM = nvvm

# def __getattr__(name):
# 	if name in API_PROTOTYPES:
# 		return getattr(nvvm, name)
# 	raise ImportError(f'cannot import name "{name}" from {__name__} ({__file__})')

#add driver functions to global namespace (so you can call, for example, cuInit instead of driver.cuInit)
for name, cfunc in vars(nvvm).items():
	globals()[name] = cfunc

def get_nvvm_log(program, err = 1):
	#use error code value, if provided, to prevent needlessly checking the error log if there is no error
	if err:
		return get_program_log(program)
	return ''

def check_nvvm_error_verbose(err, msg, program):
	check_nvvm_error(err, "%s\n%s" % (msg, get_nvvm_log(program, err)))

def check_nvvm_error_either(err, msg, program):
	# # if pycu.driver.error.VERBOSE_NVVM_ERRORS:	
	if True and program:
		check_nvvm_error_verbose(err, msg, program)
	else:
		check_nvvm_error(err, msg)

def get_error_string(error):
	result = nvvmGetErrorString(error)
	return result

def ir_version():
	major_ir  = c_int()
	minor_ir  = c_int()
	major_dbg = c_int()
	minor_dbg = c_int()
	err = nvvm.nvvmIRVersion(byref(major_ir), byref(minor_ir), byref(major_dbg), byref(minor_dbg))
	check_nvvm_error(err, "nvvmIRVersion error")

	return major_ir.value, minor_ir.value, major_dbg.value, minor_dbg.value

def version():
	major = c_int()
	minor = c_int()
	err = nvvm.nvvmVersion(byref(major), byref(minor))
	check_nvvm_error(err, "nvvmVersion error")

	return major.value, minor.value

def add_module_to_program(program, buff, size, name):
	# buff = (c_char_p*size)(buff)
	# name = (c_char_p*len(name))(name)

	err = nvvm.nvvmAddModuleToProgram(program, buff, size, name)
	# err = nvvmAddModuleToProgram(program, byref(buff), size, byref(name))
	check_nvvm_error_either(err, "nvvmCreateProgram error", program)

def compile_program(program, options = None):
	err = nvvm.nvvmCompileProgram(program, len(options) if options else 0, options)
	check_nvvm_error_either(err, "nvvmCompileProgram error", program)

def create_program():
	program = nvvmProgram()
	err = nvvm.nvvmCreateProgram(byref(program))
	check_nvvm_error_either(err, "nvvmCreateProgram error", program)

	return program

def destroy_program(program):
	#check for if/when weakref calls destroy on a handle with None as its value
	if program:
		err = nvvm.nvvmDestroyProgram(byref(program))
		check_nvvm_error_either(err, "nvvmDestroyProgram error", program)

def get_compiled_result(program, size = 0):
	if not size:
		size = get_compiled_result_size(program)

	buff = (c_char * size)()
	err = nvvm.nvvmGetCompiledResult(program, buff)
	check_nvvm_error_either(err, "nvvmGetCompiledResult error", program)

	return buff.value.decode('utf8')

def get_compiled_result_size(program):
	size = c_size_t()
	err = nvvm.nvvmGetCompiledResultSize(program, byref(size))
	check_nvvm_error_either(err, "nvvmGetCompiledResultSize error", program)

	return size.value

def get_program_log(program, size = 0):
	if not size:
		size = get_program_log_size(program)

	log = (c_char * size)()
	err = nvvm.nvvmGetProgramLog(program, log)
	check_nvvm_error(err, "nvvmGetProgramLog error")

	return log.value.decode('utf8')

def get_program_log_size(program):
	size = c_size_t()
	err = nvvm.nvvmGetProgramLogSize(program, byref(size))
	check_nvvm_error(err, "nvvmGetProgramLogSize error")

	return size.value

def lazy_add_module_to_program():
	# nvvmResult nvvmLazyAddModuleToProgram ( nvvmProgram prog, const char* buffer, size_t size, const char* name )
	pass

def verify_program(program, options = (c_char_p * 0)()): #options = None
	err = nvvm.nvvmVerifyProgram(program, len(options), options) #len(options) if options else 0
	check_nvvm_error(err, "nvvmVerifyProgram error")

from .core import *
