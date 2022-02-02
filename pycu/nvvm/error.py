VERBOSE_NVVM_ERRORS = True

class NvvmSupportError(ImportError):
    pass

class NvvmError(Exception):
    def __str__(self):
        return '\n'.join(map(str, self.args))

def check_nvvm_error(error, msg = '', exit = False):
	if error:
		exc = NvvmError(msg, nvvmResultCodes[error])
		if exit:
			print(exc)
			sys.exit(1)
		else:
			raise exc

nvvmResultCodes = {
 0:"NVVM_SUCCESS",
 1:"NVVM_ERROR_OUT_OF_MEMORY",
 2:"NVVM_ERROR_PROGRAM_CREATION_FAILURE",
 3:"NVVM_ERROR_IR_VERSION_MISMATCH",
 4:"NVVM_ERROR_INVALID_INPUT",
 5:"NVVM_ERROR_INVALID_PROGRAM",
 6:"NVVM_ERROR_INVALID_IR",
 7:"NVVM_ERROR_INVALID_OPTION",
 8:"NVVM_ERROR_NO_MODULE_IN_PROGRAM",
 9:"NVVM_ERROR_COMPILATION"
}


# # def get_nvrtc_log(handle, driver = None):
# 	# # if driver is None:
# 	# # 	from pycu.nvrtc.driver import Driver
# 	# # 	driver = Driver()

# 	# reslen = c_size_t()
# 	# err = driver.nvrtcGetProgramLogSize(handle, byref(reslen))
# 	# check_nvrtc_error(err, 'Failed to get compilation log size.')

# 	# if reslen.value > 1:
# 	# 	logbuf = (c_char * reslen.value)()
# 	# 	err = driver.nvrtcGetProgramLog(handle, logbuf)
# 	# 	check_nvrtc_error(err, 'Failed to get compilation log.')

# 	# 	# populate log attribute
# 	# 	return logbuf.value.decode('utf8')

# 	# return ''
