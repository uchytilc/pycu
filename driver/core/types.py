from . import CUdeviceptr
from .preparer import preparer

from .memory import CuBuffer, CuArray, CuNDArray

import ctypes as ct
import numpy as np

class Type:
	pass

class VoidPtrType(Type):
	def as_ctypes(self):
		return ct.c_void_p

class ScalarType(Type):
	def __init__(self, dtype):
		self.dtype = dtype

	def as_ctypes(self):
		return np.ctypeslib.as_ctypes_type(self.dtype)

	def __getitem__(self, args):
		def determine_array_spec(args):
			ndim = 1
			layout = 'C'
			if isinstance(args, (tuple, list)):
				ndim = len(args)
				order = 'C'
				if args[0].step == 1:
					layout = 'F'

			return ndim, layout

		ndim, order = determine_array_spec(args)
		return CuNDArrayType(self.dtype, ndim) #, order = order)

class BoolType(ScalarType):
	pass

class IntegerType(ScalarType):
	pass

class FloatingType(ScalarType):
	pass

voidptr_t = VoidPtrType()

bool_t = BoolType(np.bool_)

int8_t = IntegerType(np.int8)
int16_t = IntegerType(np.int16)
int32_t = IntegerType(np.int32)
int64_t = IntegerType(np.int64)

uint8_t = IntegerType(np.uint8)
uint16_t = IntegerType(np.uint16)
uint32_t = IntegerType(np.uint32)
uint64_t = IntegerType(np.uint64)

# float16 = FloatingType(np.float16)
float32_t = FloatingType(np.float32)
float64_t = FloatingType(np.float64)

preparer.add_mapping(bool, bool_t)
preparer.add_mapping(int, int64_t)
preparer.add_mapping(float, float32_t) #float64
# #TO DO
# preparer.add_preparer(list, unsupported_arg)
# preparer.add_preparer(tuple, unsupported_arg)
# preparer.add_preparer(complex, unsupported_arg)
# preparer.add_preparer(bytes, unsupported_arg)
# preparer.add_preparer(str, unsupported_arg)

pycu_types = [int8_t, int16_t, int32_t, int64_t,
			  uint8_t, uint16_t, uint32_t, uint64_t, 
			  float32_t, float64_t, #float16_t
			  bool_t, voidptr_t]

ctypes_types = [ct.c_int8, ct.c_int16, ct.c_int32, ct.c_int64,
				ct.c_uint8, ct.c_uint16, ct.c_uint32, ct.c_uint64,
				ct.c_float, ct.c_double, #None
				ct.c_bool, ct.c_void_p] # ct.c_longdouble
# # class ct.c_wchar
# # class ct.c_wchar_p
# # class ct.HRESULT
# # class ct.py_object

numpy_types = [np.int8, np.int16, np.int32, np.int64,
			   np.uint8, np.uint16, np.uint32, np.uint64,
			   np.float32, np.float64, #np.float16
			   np.bool_, None] #, np.longdouble, np.intc, np.uintc
			   # np.complex_, np.complex64, np.complex128
			   # np.str_, timedelta64, datetime64
			   # np.void

for pycu_type, ctypes_type, numpy_type in zip(pycu_types, ctypes_types, numpy_types):
	if ctypes_type:
		preparer.add_mapping(ctypes_type, pycu_type)
	if numpy_type:
		preparer.add_mapping(numpy_type, pycu_type)

class CuBufferType(Type):
	def as_ctypes(self):
		class c_CuBuffer(CUdeviceptr):
			def __init__(self, cubuffer):
				self.value = cubuffer.handle.value
		return c_CuBuffer

class CuArrayType(CuBufferType):
	pass

class CuNDArrayType(Type):
	_cache = {}
	def __init__(self, dtype, ndim):
		self.dtype = dtype
		self.ndim = ndim

	def as_ctypes(self):
		dtype = self.dtype
		ndim = self.ndim
		key = (dtype, ndim)

		c_CuNDArray = self._cache.get(key, None)
		if c_CuNDArray is None:
			class c_CuNDArray(ct.Structure):
				_fields_ = [('meminfo', ct.c_void_p),
							('parent', ct.c_void_p),
							('nitems', ct.c_ssize_t),
							('itemsize', ct.c_ssize_t),
							('data', ct.POINTER(self.dtype))] + \
							[(f'shape_{ax}', ct.c_ssize_t) for ax in range(self.ndim)] + \
							[(f'stride_{ax}', ct.c_ssize_t) for ax in range(self.ndim)]
				def __init__(self, cundarray):
					meminfo = 0
					parent = 0
					nitems = cundarray.size
					itemsize = cundarray.dtype.itemsize
					data = ct.cast(cundarray.device_pointer.value, ct.POINTER(np.ctypeslib.as_ctypes_type(self.dtype))) # ct.c_void_p(arg.device_pointer.value)
					shape = [cundarray.shape[ax] for ax in range(cundarray.ndim)]
					stride = [cundarray.strides[ax] for ax in range(cundarray.ndim)]

					super().__init__(meminfo, parent, nitems, itemsize, data, *shape, *stride)

			self._cache[key] = c_CuNDArray
		return c_CuNDArray

	@staticmethod
	def construct(self, cundarray): #template, initialize
		return CuNDArrayType(cundarray.dtype, cundarray.ndim)

		# # #ty, val
		# # # # https://github.com/numba/numba/blob/4bcea520c98b6c2aa75b0e18739da6dd61fc7526/numba/cuda/compiler.py#L768
		# # # if isinstance(ty, types.Array):
		# # 	# devary = wrap_arg(val).to_device(retr, stream)

		# # 	# c_intp = ct.c_ssize_t

		# # 	# meminfo = ct.c_void_p(0)
		# # 	# parent = ct.c_void_p(0)
		# # 	# nitems = c_intp(devary.size)
		# # 	# itemsize = c_intp(devary.dtype.itemsize)
		# # 	# data = ct.c_void_p(driver.device_pointer(devary))
		# # 	# kernelargs.append(meminfo)
		# # 	# kernelargs.append(parent)
		# # 	# kernelargs.append(nitems)
		# # 	# kernelargs.append(itemsize)
		# # 	# kernelargs.append(data)
		# # 	# for ax in range(devary.ndim):
		# # 	# 	kernelargs.append(c_intp(devary.shape[ax]))
		# # 	# for ax in range(devary.ndim):
		# # 	# 	kernelargs.append(c_intp(devary.strides[ax]))

# # # def device_1darray(shape, dtype = float32):
# # 	# return Cu1DArray(shape, dtype)

# # # def device_2darray(shape, dtype = float32):
# # 	# return Cu2DArray(shape, dtype)

# # # def device_3darray(shape, dtype = float32):
# # 	# return Cu3DArray(shape, dtype)

cubuffer_t = CuBufferType()
cuarray_t = CuArrayType()

preparer.add_mapping(CuBuffer, cubuffer_t)
preparer.add_mapping(CuArray, cuarray_t)
preparer.add_mapping(CuNDArray, CuNDArrayType)
