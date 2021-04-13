VERBOSE_NVRTC_ERRORS = True

class NvrtcSupportError(ImportError):
	pass

class NvrtcError(Exception):
	def __str__(self):
		return '\n'.join(map(str, self.args))

def check_nvrtc_error(error, msg = '', exit = False):
	if error:
		exc = NvrtcError(msg, nvrtcResultCodes[error])
		if exit:
			print(exc)
			sys.exit(1)
		else:
			raise exc

nvrtcResultCodes = {
 0:"NVRTC_SUCCESS",
 1:"NVRTC_ERROR_OUT_OF_MEMORY",
 2:"NVRTC_ERROR_PROGRAM_CREATION_FAILURE",
 3:"NVRTC_ERROR_INVALID_INPUT",
 4:"NVRTC_ERROR_INVALID_PROGRAM",
 5:"NVRTC_ERROR_INVALID_OPTION",
 6:"NVRTC_ERROR_COMPILATION",
 7:"NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
 8:"NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
 9:"NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION",
10:"NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",
11:"NVRTC_ERROR_INTERNAL_ERROR"
}

# def get_nvrtc_log(handle, driver = None):
	# # if driver is None:
	# # 	from pycu.nvrtc.driver import Driver
	# # 	driver = Driver()

	# reslen = c_size_t()
	# err = driver.nvrtcGetProgramLogSize(handle, byref(reslen))
	# check_nvrtc_error(err, 'Failed to get compilation log size.')

	# if reslen.value > 1:
	# 	logbuf = (c_char * reslen.value)()
	# 	err = driver.nvrtcGetProgramLog(handle, logbuf)
	# 	check_nvrtc_error(err, 'Failed to get compilation log.')

	# 	# populate log attribute
	# 	return logbuf.value.decode('utf8')

	# return ''