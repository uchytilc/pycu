from ..driver import (CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
						 CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE, CU_JIT_TARGET_FROM_CUCONTEXT,
						 CU_JIT_INPUT_OBJECT, CU_JIT_INPUT_PTX, CU_JIT_INPUT_LIBRARY, CU_JIT_INPUT_CUBIN, CU_JIT_INPUT_FATBINARY)

from ..driver.core import module_load_data, module_get_function, Module, Linker
from ..nvrtc.core import NVRTC, nvrtc_headers
from ..nvvm.core import NVVM
from ..utils import open_file, generate_hash

import numpy as np
import ctypes
import os

# 4.2.1.16. --cudart {none|shared|static} (-cudart)
# Specify the type of CUDA runtime library to be used: no CUDA runtime library, shared/dynamic CUDA runtime library, or static CUDA runtime library.

# Allowed Values
# none
# shared
# static
# Default
# The static CUDA runtime library is used by default.

# 4.2.1.17. --cudadevrt {none|static} (-cudadevrt)
# Specify the type of CUDA device runtime library to be used: no CUDA device runtime library, or static CUDA device runtime library.

# Allowed Values
# none
# static
# Default
# The static CUDA device runtime library is used by default.


#these are not real CUDA enums
CU_JIT_INPUT_FILEPATH = -1
# CU_JIT_INPUT_SOURCE = -2
# CU_JIT_INPUT_NVVM = -3

def compile_options_as_str(options):
	#converts option dict to string for hashing purposes
	opts = []
	for key in sorted(list(options.keys())):
		val = options[key]
		#convert list/tuple inputs to single string
		if isinstance(val, (tuple, list)):
			val = f"[{','.join([str(v) for v in sorted(val)])}]"
		opts.append(f"{key}:{val}")
	return f"[{','.join(opts)}]"

class JitifyCache:
	def __init__(self):
		#key: hash of precompiled code (source, nvvmir) and compiler options
		#value: ptx source
		self.ptx = {}
		self.lowered_names = {}

	def clear(self):
		self.ptx.clear()
		self.lowered_names.clear()
		# self.cubins.clear()

class Jitify:
	# #For threadsafe dict (even though it isn't needed because of the GIL)
	# lock = threading.Lock()
	# with lock:
	# 	ptx[key] = value

	#TO DO:
		#make this thread specific and include context in key
	cache = JitifyCache()

	def __init__(self, cache = True, save = False, **options):
		#will check cache for precompiled ptx of nvvmir and source
		self.check_cache = True
		#cache in jitify
		self.do_cache = cache
		#save to disk
		self.do_save  = save

		jitdir = os.path.dirname(os.path.realpath(__file__))
		cwd    = os.getcwd()

		#default paths to check when compiling code
		# self.defaults = [jitdir, cwd]
		
		self.cubindir = os.path.join(jitdir, "cache")

		#TO DO:
			#make this thread specific and include context in key
		self._compilation_queue = []

	def reset(self):
		self._compilation_queue.clear()

	def add_file(self, path, CUjit_option = {}):
		self._compilation_queue.append((CU_JIT_INPUT_FILEPATH, path, '', CUjit_option))

	def add_source(self, source, name = '<pycu-source>', CUjit_option = {}, nvrtc_options = {}):
		ptx, lowered_names = self.source_to_ptx(source, name, nvrtc_options)
		self._compilation_queue.append((CU_JIT_INPUT_PTX, ptx, name, CUjit_option))
		if isinstance(nvrtc_options.get('name_expression', None), str):
			return lowered_names[0]
		return lowered_names

	def add_nvvmir(self, nvvmir, name = '<pycu-nvvmir>', CUjit_option = {}, nvvm_options = {}):
		ptx = self.nvvmir_to_ptx(nvvmir, name, nvvm_options)
		self._compilation_queue.append((CU_JIT_INPUT_PTX, ptx, name, CUjit_option))

	def add_ptx(self, ptx, name = '<pycu-ptx>', CUjit_option = {}):
		self._compilation_queue.append((CU_JIT_INPUT_PTX, ptx, name, CUjit_option))

	def add_object(self, obj, name = '<pycu-object>', CUjit_option = {}):
		self._compilation_queue.append((CU_JIT_INPUT_OBJECT, obj, name, CUjit_option))

	def add_library(self, library, name = '<pycu-library>', CUjit_option = {}):
		self._compilation_queue.append((CU_JIT_INPUT_LIBRARY, library, name, CUjit_option))

	def add_cubin(self, cubin, name = '<pycu-cubin>', CUjit_option = {}):
		self._compilation_queue.append((CU_JIT_INPUT_CUBIN, cubin, name, CUjit_option))

	def add_fatbin(self, fatbin, name = '<pycu-fatbin>', CUjit_option = {}):
		self._compilation_queue.append((CU_JIT_INPUT_FATBINARY, fatbin, name, CUjit_option))

	def compile(self, linker_options = {}, module_options = {}, get_cubin = False):
		logsize = 1024 #c_void_p
		
		linker_options = {
			CU_JIT_INFO_LOG_BUFFER.value: None, #addressof(_linker_info),
			CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES.value: logsize,
			CU_JIT_ERROR_LOG_BUFFER.value: None, #addressof(_linker_errors)
			CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES.value: logsize,
			CU_JIT_LOG_VERBOSE.value: 1,
			CU_JIT_TARGET_FROM_CUCONTEXT.value: None
		}

		linker = Linker(options = linker_options)

		while self._compilation_queue:
			jittype, data, name, CUjit_option = self._compilation_queue.pop(0)
			if jittype == CU_JIT_INPUT_FILEPATH:
				linker.add_file(data, CUjit_option)
			else:
				# print(name)
				# print(data[:200])
				# if jittype != CU_JIT_INPUT_LIBRARY:
				# 	print('############')
				# 	print(data)
				linker.add_data(jittype, data, name, CUjit_option)

		cubin, size = linker.complete()

		if get_cubin:
			cubin_ptr = ctypes.cast(cubin, ctypes.POINTER(ctypes.c_char))
			return bytes(np.ctypeslib.as_array(cubin_ptr, shape = (size,)))

		# module_options = {
		# 	CU_JIT_INFO_LOG_BUFFER: None,
		# 	CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: 1024,
		# 	CU_JIT_ERROR_LOG_BUFFER: None,
		# 	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: 1024,
		# 	CU_JIT_LOG_VERBOSE: 1,
		# }

		#note: cubin must be turned into a module before the Linker is destroyed
		module = Module(cubin) #, options = module_options

		return module

	def source_to_ptx(self, source, name = '<pycu-source>', nvrtc_options = {}):
		#name_expression example: "kernel_name<float, float, int>", &kernel_name, ...
		name_expressions = nvrtc_options.get("name-expression", '')
		if not isinstance(name_expressions, (list, tuple)):
			name_expressions = [name_expressions]

		nvrtc = NVRTC(source, name = name, headers = nvrtc_headers) #I = nvrtc_options.get('I', [])

		for name_expression in name_expressions:
			if name_expression:
				nvrtc.add_name_expression(name_expression)

		ptx = None
		if True: #self.check_cache:
			key = generate_hash(compile_options_as_str(nvrtc_options) + source)
			ptx = self.cache.ptx.get(key, None)
			lowered_names = self.cache.lowered_names.get(key, [])
		if ptx is None:
			ptx = nvrtc.compile_program(nvrtc_options)
			for name_expression in name_expressions:
				if name_expression:
					lowered_names.append(nvrtc.get_lowered_name(name_expression))
			if self.do_cache: #note: if check_cache is true then do_cache must be true
				self.cache.ptx[key] = ptx 
				if name_expressions:
					self.cache.lowered_names[key] = lowered_names

		return ptx, lowered_names

	def nvvmir_to_ptx(self, nvvmirs, name = '<pycu-nvvmir>', nvvm_options = {}):
		nvvm = NVVM()

		if isinstance(nvvmirs, str):
			nvvmirs = [nvvmirs]

		ptx = None
		if True: #self.check_cache:
			key = generate_hash(compile_options_as_str(nvvm_options) + ' '.join(nvvmirs))
			ptx = self.cache.ptx.get(key, None)
			# if ptx is None:
			# 	ptx = check_cache_dir(key)
		if ptx is None:
			for nvvmir in nvvmirs:
				nvvm.add_module(nvvmir, name)
			if nvvm_options.get('libdevice', False):
				nvvm.add_module(nvvm.get_libdevice(nvvm_options.get('arch', None)), 'libdevice')
			ptx = nvvm.compile(nvvm_options)
			if self.do_cache: #note: if check_cache is true then do_cache must be true
				self.cache.ptx[key] = ptx

		return ptx

jitify = Jitify()
