from pycu.nvvm import (get_libdevice, ir_version, version, add_module_to_program, compile_program,
					   create_program, destroy_program, get_compiled_result, get_compiled_result_size,
					   get_program_log, get_program_log_size, lazy_add_module_to_program, verify_program)

import os
import sys
from ctypes import c_char_p
import weakref

class NVVMPtr:
	#key: arch associated with libdevice (None indicates libdevice is not arch specific)
	#value: libdevice source
	libdevice = {}

	#key:given arch
	#value: closest available arch found
	searched_arch = {}

	def __init__(self, handle, arch = 20):
		self.get_libdevice(arch)

		self.handle = handle

	def get_libdevice(self, arch = 20):
		libdevice = self.libdevice.get(arch, None)
		if libdevice is None:
			#note: use False instead of None in searched_arch.get when indicating failure to prevent getting None key from libdevice (libdevice with no "compute_" is stored under None key)
			libdevice = self.libdevice.get(self.searched_arch.get(arch, False), None)
		if libdevice is None:
			found_arch, libdevice = next(iter(get_libdevice(arch).items()))
			self.searched_arch[arch] = found_arch
			self.libdevice[arch] = libdevice
		return libdevice

	def get_version(self):
		return version()

	def get_ir_version(self):
		return ir_version()

	def add_module(self, buff, name = "<unnamed>"):
		buff = buff.encode('utf8')
		name = name.encode('utf8')
		size = len(buff)

		add_module_to_program(self.handle, buff, size, name)

	def compile(self, options = {}):
		"""
		https://docs.nvidia.com/cuda/libnvvm-api/group__compilation.html#group__compilation_1g76ac1e23f5d0e2240e78be0e63450346

		Valid compiler options are
			-g (enable generation of debugging information, valid only with -opt=0)
			-generate-line-info (generate line number information)

			-opt=
				0 (disable optimizations)
				3 (default, enable optimizations)

			-arch=
				compute_30 (default)
				compute_32
				compute_35
				compute_37
				compute_50
				compute_52
				compute_53
				compute_60
				compute_61
				compute_62
				compute_70
				compute_72

			-ftz=
				0 (default, preserve denormal values, when performing single-precision floating-point operations)
				1 (flush denormal values to zero, when performing single-precision floating-point operations)

			-prec-sqrt=
				0 (use a faster approximation for single-precision floating-point square root)
				1 (default, use IEEE round-to-nearest mode for single-precision floating-point square root)

			-prec-div=
				0 (use a faster approximation for single-precision floating-point division and reciprocals)
				1 (default, use IEEE round-to-nearest mode for single-precision floating-point division and reciprocals)

			-fma=
				0 (disable FMA contraction)
				1 (default, enable FMA contraction)
	
			-g (enable generation of debugging information, valid only with -opt=0)

			-generate-line-info (generate line number information)
		"""

		opt = options.get("opt", 3)
		arch = options.get("arch", 30)
		ftz = options.get("ftz", 0)
		prec_sqrt = options.get("prec_sqrt", 1)
		prec_div = options.get("prec_div", 1)
		fma = options.get("fma", 0)

		opts = [f"-opt={opt}",
				f"-arch=compute_{arch}",
				f"-ftz={ftz}",
				f"-prec-sqrt={prec_sqrt}",
				f"-prec-div={prec_div}",
				f"-fma={fma}",]

		if options.get("g", False) and opt == 0:
			if opt == 0:
				opts.append("-g")
			else:
				#raise warning (g is only valid when -opt=0)
				pass

		if options.get("generate-line-info", True):
			opts.append("-generate-line-info")

		options = (c_char_p * len(opts))(*[c_char_p(opt.encode('utf8')) for opt in opts])

		compile_program(self.handle, options)

		ptx = get_compiled_result(self.handle)

		#TO DO
			#Apply Numba's debug patch to ptx
		# #From numba/numba/cuda/cudadrv/nvvm.py
		# patch_ptx_debug_pubnames(ptx)

		return ptx

	def verify_program(self, options = {}):
		pass
		# verify_program(self.handle, )

class NVVM(NVVMPtr):
	def __init__(self, arch = 20):

		# self.handle = handle = create_program()
		handle = create_program()
		weakref.finalize(self, destroy_program, handle)

		super().__init__(handle, arch)
