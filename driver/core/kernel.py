from pycu.driver import launch_kernel

import numpy as np
import ctypes
# import _ctypes 

#TO DO
	#allow a preparer to create an object to hold until the kernel has finished
		#This would allow numpy arrays to be given as input
			#simply create a buffer, hold it for the length of the kernel, write back into the numpy array, and destroy the device array when the kernel ends

class ArgPreparer:
	#prepares args for the kernel by moving them to the device.
	types = {}
	prepare = True

	def __call__(self, args, sig):
		if not self.prepare:
			return args

		args = list(args)

		if not sig:
			#first argument is the return type
			sig = [None] + [type(arg) for arg in args]

		if len(args) != len(sig[1:]):
			raise ValueError("The number of args provided does not match the signature length.")

		for n in range(len(args)):
			arg = args.pop(0)
			typ = sig[n + 1]
			prepared = self.get_preparer(typ)(typ, arg)
			#note: if preparer expends `arg` to multiple input types it returns a list
			if isinstance(prepared, (list, tuple)):
				args.extend(prepared)
			else:
				args.append(prepared)

		return args

	@staticmethod
	def get_preparer(typ):
		preparer = ArgPreparer.types.get(typ, None)
		if preparer is None:
			raise TypeError(f"Unkown type {typ} was provided to kernel")
		return preparer

	@staticmethod
	def add_preparer(typ, preparer):
		ArgPreparer.types[typ] = preparer

	@staticmethod
	def remove_preparer(typ):
		return ArgPreparer.types.pop(typ)

def unsupported_type(typ, arg):
	raise TypeError(f"{type(arg)} is currently not supported")

def ctypes_type(typ, arg):
	return arg

def numpy_type(typ, arg):
	ctype = np.ctypeslib.as_ctypes_type(typ)
	return ctype(arg)

def bool_type(typ, arg):
	return ctypes.c_bool(arg)

def int_type(typ, arg):
	return ctypes.c_int(arg)

def float_type(typ, arg):
	return ctypes.c_float(arg)

ArgPreparer.add_preparer(bool, bool_type)
ArgPreparer.add_preparer(int, int_type)
ArgPreparer.add_preparer(float, float_type)

#TO DO
ArgPreparer.add_preparer(list, unsupported_type)
ArgPreparer.add_preparer(tuple, unsupported_type)
ArgPreparer.add_preparer(complex, unsupported_type)
ArgPreparer.add_preparer(bytes, unsupported_type)
ArgPreparer.add_preparer(str, unsupported_type)

def expand():
	pass
	# def expand_struct(struct):
	# 	c_args = []
	# 	for field in struct._fields_:
	# 		#TO DO
	# 			#find better way to identify a struct contained within a struct (isinstance(_ctypes.Structure) doesn't work)
	# 		if getattr(field[1], "_fields_", None):
	# 			c_args.extend(expand_struct(getattr(struct, field[0])))
	# 		else:
	# 			c_args.append(field[1](getattr(struct, field[0])))
	# 	return c_args


	# def prepare_args(args, sig, expand):
	# 	'''
	# 	converts input args to ctypes for kernel input
	# 	if ctypes args are given they are skipped
	# 	'''

	# 	c_args = []
	# 	for arg, typ in zip(args, sig[1:]):
	# 		#convert input to ctypes type
	# 		if not isinstance(arg, (_ctypes._SimpleCData, _ctypes.Array,
	# 								_ctypes._Pointer, _ctypes.Union,
	# 								_ctypes.Structure)):
	# 			#TO DO
	# 				#change this
	# 			##########################
	# 			if isinstance(typ, types.Pointer):
	# 				arg = arg.handle
	# 			##########################
	# 			else:
	# 				if isinstance(arg, (tuple, list)):
	# 					ctype = typ.as_ctypes()(*arg)
	# 				else:
	# 					ctype = typ.as_ctypes()(arg)
	# 				#expand the struct into individual values for Numba
	# 				if isinstance(ctype, _ctypes.Structure) and expand:
	# 					c_args.extend(expand_struct(ctype))
	# 				else:
	# 					c_args.append(ctype)
	# 		#already a ctypes value, do nothing
	# 		else:
	# 			c_args.append(arg)

	# 	return c_args

types = [ctypes.c_byte, ctypes.c_ubyte, ctypes.c_char, ctypes.c_char_p,
		 ctypes.c_double, ctypes.c_longdouble, ctypes.c_float,
		 ctypes.c_int, ctypes.c_int8, ctypes.c_int16, ctypes.c_int32, ctypes.c_int64,
		 ctypes.c_uint, ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64,
		 ctypes.c_long, ctypes.c_longlong, ctypes.c_short, ctypes.c_size_t, ctypes.c_ssize_t,
		 ctypes.c_ulong, ctypes.c_ulonglong, ctypes.c_ushort, ctypes.c_void_p, ctypes.c_bool]
# class ctypes.c_wchar
# class ctypes.c_wchar_p
# class ctypes.HRESULT
# class ctypes.py_object

for typ in types:
	ArgPreparer.add_preparer(typ, ctypes_type)

#note: some of these types are alias to other types in the list
types = [np.byte, np.short, np.intc, np.int_, np.longlong,
		 np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong,
		 np.int8, np.int16, np.int32, np.int64,
		 np.uint8, np.uint16, np.uint32, np.uint64,
		 np.float_, np.float16, np.float32, np.float64,
		 np.half, np.single, np.double, np.longdouble,
		 np.bool_, np.uintp, np.intp]
		 # np.complex_, np.complex64, np.complex128
		 # np.str_, timedelta64, datetime64
		 # np.void

for typ in types:
	ArgPreparer.add_preparer(typ, numpy_type)








def check_dim(dim, name):
	if not isinstance(dim, (tuple, list)):
		dim = [dim]
	else:
		dim = list(dim)

	if len(dim) > 3:
		raise ValueError('%s must be a sequence of 1, 2 or 3 integers, '
						 'got %r' % (name, dim))
	while len(dim) < 3:
		dim.append(1)
	return dim

def normalize_griddim_blockdim(griddim, blockdim):
	if griddim is None or blockdim is None:
		return griddim, blockdim

	griddim = check_dim(griddim, 'griddim')
	blockdim = check_dim(blockdim, 'blockdim')

	return griddim, blockdim

class Kernel:
	arg_preparer = ArgPreparer()
	def __init__(self, handle, sig = None, prepare = True, griddim = None, blockdim = None, stream = 0, sharedmem = 0):
		self.handle = handle
		self.configure(sig, prepare, griddim, blockdim, stream, sharedmem)

	def configure(self, sig, prepare, griddim, blockdim, stream, sharedmem):
		#if a signature is provided, parse the input args to convert them to the proper format for the kernel call
		#if the signature is not provided it is left up to the user to pass in the correct input arguments in the correct format
		self.sig = sig
		self.prepare = prepare
		self.griddim, self.blockdim = normalize_griddim_blockdim(griddim, blockdim)
		self.stream = stream.handle if stream else 0 #stream.handle is isinstance(stream, Stream) else stream
		self.sharedmem = sharedmem

	#<<
	def __lshift__(self, config):
		#set kernel configuration

		if len(config) not in [2, 3, 4]:
			raise ValueError('must specify at least the griddim and blockdim')
		kernel = Kernel(self.handle, self.sig, self.prepare, *config)
		return kernel

	#>>
	def __rshift__(self, args):
		#call kernel

		if isinstance(args, (tuple, list)):
			return self(*args)
		return self(args)

	def __call__(self, *args):
		if not self.griddim or not self.blockdim:
			raise ValueError("The kernel's griddim and blockdim must be set before the kernel can be called.")
		if self.prepare:
			args = self.arg_preparer(args, self.sig)

		launch_kernel(self.handle, self.griddim, self.blockdim, args, self.sharedmem, self.stream)
