from .driver import Driver, get_libcudadevrt
from .api import API_PROTOTYPES
# from .enums import *
from .typedefs import nvrtcProgram
from .error import check_nvrtc_error

from ctypes import byref, c_char_p, c_char, c_int, c_size_t

nvrtc = Driver()

#add driver functions to global namespace (so you can call, for example, cuInit instead of driver.cuInit)
for name, cfunc in vars(nvrtc).items():
	globals()[name] = cfunc

def get_nvrtc_log(program, err = 1):
	#use error code value, if provided, to prevent needlessly checking the error log if there is no error
	if err:
		return get_program_log(program)
	return ''

def check_nvrtc_error_verbose(err, msg, program):
	check_nvrtc_error(err, "%s\n%s" % (msg, get_nvrtc_log(program, err)))

def check_nvrtc_error_either(err, msg, program):
	# if pycu.driver.error.VERBOSE_NVRTC_ERRORS:
	if True and program:
		check_nvrtc_error_verbose(err, msg, program)
	else:
		check_nvrtc_error(err, msg)

def add_name_expression(program, name_expression):
	err = nvrtcAddNameExpression(program, name_expression)
	check_nvrtc_error_either(err, 'nvrtcAddNameExpression error', program)

def compile_program(program, options = (c_char_p * 0)()):
	err = nvrtcCompileProgram(program, len(options), options)
	check_nvrtc_error_either(err, 'nvrtcCompileProgram error', program)

def create_program(src, header_src = None, header_names = None, name = None):
	program = nvrtcProgram()
	err = nvrtcCreateProgram(program, src, name, len(header_src) if header_src else 0, header_src, header_names)
	check_nvrtc_error_either(err, 'nvrtcCreateProgram error', program)

	return program

def destroy_program(program):
	#check for if/when weakref calls destroy with None for handle
	if program:
		err = nvrtcDestroyProgram(byref(program))
		check_nvrtc_error_either(err, 'nvrtcDestroyProgram error', program)

def get_lowered_name(program, name_expression):
	lowered_name = c_char_p()
	err = nvrtcGetLoweredName(program, name_expression, byref(lowered_name))
	check_nvrtc_error_either(err, 'nvrtcGetLoweredName error', program)

	return lowered_name.value.decode('utf8')

def get_ptx(program, size = 0):
	if not size:
		size = get_ptx_size(program)

	ptx = (c_char*size)()
	err = nvrtcGetPTX(program, ptx)
	check_nvrtc_error_either(err, 'nvrtcGetPTX error', program)

	return ptx.value.decode('utf8')

def get_ptx_size(program):
	size = c_size_t()
	err = nvrtcGetPTXSize(program, byref(size))
	check_nvrtc_error_either(err, 'nvrtcGetPTXSize error', program)

	return size.value

def get_program_log(program, size = 0):
	if not size:
		size = get_program_log_size(program)

	log = (c_char * size)()
	err = nvrtcGetProgramLog(program, log)
	check_nvrtc_error(err, "nvrtcGetProgramLog error")

	return log.value.decode('utf8')

def get_program_log_size(program):
	size = c_size_t()
	err = nvrtcGetProgramLogSize(program, byref(size))
	check_nvrtc_error(err, "nvrtcGetProgramLogSize error")

	return size.value

from .core import *
