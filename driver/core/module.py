from pycu.driver import (link_create, link_complete, link_destroy, link_add_data, link_add_file,
						 module_get_function, module_get_global, module_get_surf_ref,
						 module_get_tex_ref, module_load, module_load_data, module_load_data_ex,
						 module_load_fat_binary, module_unload)
from pycu.driver.enums import (CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
							   CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE, CU_JIT_TARGET_FROM_CUCONTEXT,
							   CU_JIT_MAX_REGISTERS, CU_JIT_THREADS_PER_BLOCK, CU_JIT_WALL_TIME,
							   CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_TARGET, CU_JIT_FALLBACK_STRATEGY,
							   CU_JIT_GENERATE_DEBUG_INFO, CU_JIT_GENERATE_LINE_INFO, CU_JIT_CACHE_MODE,
							   CU_JIT_INPUT_CUBIN, CU_JIT_INPUT_PTX, CU_JIT_INPUT_FATBINARY,
							   CU_JIT_INPUT_OBJECT, CU_JIT_INPUT_LIBRARY,
							   CUjit_option, CUjit_fallback, CUjit_cacheMode)

import os.path
import weakref
from ctypes import c_void_p, c_char_p, c_char, c_uint, c_int, c_float, addressof

ext_map = {
	'o': CU_JIT_INPUT_OBJECT,
	'ptx': CU_JIT_INPUT_PTX,
	'a': CU_JIT_INPUT_LIBRARY,
	'cubin': CU_JIT_INPUT_CUBIN,
	'fatbin': CU_JIT_INPUT_FATBINARY,
}
	# 'lib': CU_JIT_INPUT_LIBRARY,

#converts option arguments to ctypes
def as_CUjit_options(options): #compiler_only = True
	# if CU_JIT_THREADS_PER_BLOCK in options and CU_JIT_TARGET in options:
	# 	raise ValueError("CU_JIT_THREADS_PER_BLOCK and CU_JIT_TARGET cannot be used at the same time.")

	# if not compiler_only:
	# 	# if any in options:
	# 		# raise ValueError
	# 	# CU_JIT_MAX_REGISTERS
	# 	# CU_JIT_THREADS_PER_BLOCK
	# 	# CU_JIT_OPTIMIZATION_LEVEL
	# 	# CU_JIT_FALLBACK_STRATEGY
	# 	# CU_JIT_GENERATE_LINE_INFO
	# 	# CU_JIT_CACHE_MODE

	keep_alive = []

	for option, value in options.items():
		if CU_JIT_INFO_LOG_BUFFER.value == option:
			#linker_info
			if value is None:
				logsize = options.get(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES.value, 1024)
				# options[CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES] = logsize
				value = (c_char * logsize)()
			keep_alive.append(value)
			options[option] = addressof(value)

		elif CU_JIT_ERROR_LOG_BUFFER.value == option:
			#linker_errors
			if value is None:
				logsize = options.get(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES.value, 1024)
				# options[CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES] = logsize
				value = (c_char * logsize)()
			keep_alive.append(value)
			options[option] = addressof(value)
		else:
			options[option] = c_void_p(value)

		# elif CU_JIT_TARGET_FROM_CUCONTEXT == option:
		# 	options[option] = c_void_p(None)
		# elif CU_JIT_MAX_REGISTERS == option:
		# 	options[option] = c_void_p(c_uint(value))
		# elif CU_JIT_THREADS_PER_BLOCK == option:
		# 	options[option] = c_void_p(c_uint(value))
		# elif CU_JIT_WALL_TIME == option:
		# 	options[option] = c_void_p(c_float(value))
		# elif CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES == option:
		# 	options[option] = c_void_p(c_uint(value))
		# elif CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES == option:
		# 	options[option] = c_void_p(c_uint(value))
		# elif CU_JIT_OPTIMIZATION_LEVEL == option:
		# 	options[option] = c_void_p(c_uint(value))
		# elif CU_JIT_TARGET == option:
		# 	options[option] = c_void_p(c_uint(value))
		# elif CU_JIT_FALLBACK_STRATEGY == option:
		# 	options[option] = c_void_p(CUjit_fallback(value))
		# elif CU_JIT_GENERATE_DEBUG_INFO == option:
		# 	options[option] = c_void_p(c_int(value))
		# elif CU_JIT_LOG_VERBOSE == option:
		# 	options[option] = c_void_p(c_int(value))
		# elif CU_JIT_GENERATE_LINE_INFO == option:
		# 	options[option] = c_void_p(c_int(value))
		# elif CU_JIT_CACHE_MODE == option:
		# 	options[option] = c_void_p(CUjit_cacheMode(value))

		# CU_JIT_INPUT_CUBIN,
		# CU_JIT_INPUT_PTX,
		# CU_JIT_INPUT_FATBINARY,
		# CU_JIT_INPUT_OBJECT,
		# CU_JIT_INPUT_LIBRARY,

	return keep_alive

def prepare_options(options):
	#generates option keys, and option values for function call

	optkeys = None
	optvals = None
	keep_alive = []
	if options:
		keep_alive = as_CUjit_options(options) #compiler_only = True

		optkeys = (CUjit_option * len(options))(*options.keys())
		optvals = (c_void_p * len(options))(*options.values())

		keep_alive.extend([optkeys, optvals])

	return optkeys, optvals, keep_alive

class LinkerPtr:
	def __init__(self, handle):
		self.handle = handle

	def __repr__(self):
		return f"LinkerPtr() <{self.handle.value}>"

    # @property
    # def info_log(self):
    #     return self.linker_info_buff.value.decode('utf8')

    # @property
    # def error_log(self):
    #     return self.linker_error_buff.value.decode('utf8')

		# optkeys, optvals, keep_alive = prepare_options(options)
		# self._keep_alive.extend(keep_alive)
		# del options

		##########

		# if isinstance(name, str):
		# 	name = name.encode('utf8')

		# link_add_data(self.handle, jittype, data, size, name, optkeys, optvals)
		# self._keep_alive.extend([data, name, optkeys, optvals])

	def add_ptx(self, ptx, name = '<pycu-ptx>', options = {}):
		optkeys, optvals, keep_alive = prepare_options(options)
		self._keep_alive.extend(keep_alive)
		# del options

		if isinstance(ptx, str):
			ptx = ptx.encode('utf8')
		if isinstance(name, str):
			name = name.encode('utf8')

		size = len(ptx)
		data = c_char_p(ptx)

		link_add_data(self.handle, CU_JIT_INPUT_PTX, data, size, name, optkeys, optvals)
		self._keep_alive.extend([ptx, name, optkeys, optvals])

	def add_cubin(self, cubin, name = '<pycu-cubin>', options = {}):
		pass
		# optkeys, optvals, keep_alive = prepare_options(options)
		# self._keep_alive.extend(keep_alive)

		# size = len(cubin)
		# data = c_char_p(cubin.encode('utf8'))
		# name = name.encode('utf8')

		# link_add_data(self.handle, CU_JIT_INPUT_CUBIN, data, size, name, optkeys, optvals)
		# self._keep_alive.extend([cubin, name, optkeys, optvals])

	def add_object(self, obj, name = '<pycu-object>', options = {}):
		pass
		# link_add_data(self.handle, CU_JIT_INPUT_OBJECT, data, size, name, optkeys, optvals)
		# self._keep_alive.extend([obj, name, optkeys, optvals])

	def add_library(self, lib, name = '<pycu-library>', options = {}):
		optkeys, optvals, keep_alive = prepare_options(options)
		self._keep_alive.extend(keep_alive)
		# del options

		if isinstance(name, str):
			name = name.encode('utf8')

		size = len(lib)
		data = c_char_p(lib)

		link_add_data(self.handle, CU_JIT_INPUT_LIBRARY, data, size, name, optkeys, optvals)
		self._keep_alive.extend([lib, name, optkeys, optvals])

	def add_fatbinary(self, fatbin, name = '<pycu-fatbin>', options = {}):
		#note: fatcubin is a collection of cubins of the same device code but compile/optimized for several architectures

		pass
		# link_add_data(self.handle, CU_JIT_INPUT_FATBINAR, data, size, name, optkeys, optvals)
		# self._keep_alive.extend([fatbin, name, optkeys, optvals])

	def add_data(self, jittype, data, name = '<pycu-data>', options = {}): #size
		if jittype == CU_JIT_INPUT_PTX:
			add_data = self.add_ptx
		elif jittype == CU_JIT_INPUT_CUBIN:
			add_data = self.add_cubin
		elif jittype == CU_JIT_INPUT_OBJECT:
			add_data = self.add_object
		elif jittype == CU_JIT_INPUT_LIBRARY:
			add_data = self.add_library
		elif jittype == CU_JIT_INPUT_FATBINAR:
			add_data = self.add_fatbinary
		else:
			raise ValueError('jittype not understood')

		add_data(data, name, options)

	def add_file(self, path, options = {}):
		_, file = os.path.split(path)
		jittype = ext_map[file.split('.')[-1]]

		optkeys, optvals, keep_alive = prepare_options(options)
		self._keep_alive.extend(keep_alive)

		path = c_char_p(path.encode('utf8'))

		link_add_file(self.handle, jittype, path, optkeys, optvals)
		self._keep_alive.extend([path, optkeys, optvals])

	def complete(self):
		cubin, size = link_complete(self.handle)
		del self._keep_alive

		if size <= 0:
			raise ValueError('Linker returned a zero size cubin')

		return cubin, size

class Linker(LinkerPtr):
	def __init__(self, options = {}, auto_free = True):
		optkeys, optvals, self._keep_alive = prepare_options(options)
		# del options

		handle = link_create(optkeys, optvals)
		if auto_free:
			weakref.finalize(self, link_destroy, handle)

		super().__init__(handle)

	def __repr__(self):
		return f"Linker() <{self.handle.value}>"

class ModulePtr:
	def __init__(self, handle):
		self.handle = handle

	def __repr__(self):
		return f"ModulePtr() <{self.handle.value}>"

	def get_function(self, entry):
		return module_get_function(self.handle, entry.encode('utf8'))

	def get_global(self, name):
		return module_get_global(self.handle, name.encode('utf8'))

	def get_surf_ref(self, name):
		return module_get_surf_ref(self.handle, name.encode('utf8'))

	def get_tex_ref(self, name):
		return module_get_tex_ref(self.handle, name.encode('utf8'))

class Module(ModulePtr):
	def __init__(self, image, options = {}, auto_free = True):
		#module_load_fat_binary
		loader = module_load_data
		opt = []
		# if options:
			# loader = module_load_data_ex
			# optkeys, optvals, keep_alive = prepare_options(options)
			# opt.extend([optkeys, optvals])
			# self._keep_alive.extend(keep_alive)

		handle = loader(image, *opt)
		if auto_free:
			weakref.finalize(self, module_unload, handle)

		super().__init__(handle)

	def __repr__(self):
		return f"Module() <{self.handle.value}>"
