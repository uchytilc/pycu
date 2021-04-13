# from pycu.core.py_allocator import py_malloc, py_free
from pycu.driver import (mem_alloc, mem_free, memcpy_dtoh, memcpy_dtoh_async, memcpy_htod, memcpy_htod_async,
						mem_alloc_host, mem_free_host, mem_host_register, mem_host_unregister) #cuMemsetD8, cuMemsetD8Async, cuMemsetD16, cuMemsetD16Async, cuMemsetD32, cuMemsetD32Async

from pycu.driver.core import ArgPreparer, unsupported_type

import numpy as np
import ctypes
import weakref
import warnings

#https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api

	#CUresult cuMemGetInfo ( size_t* free, size_t* total )
		#A device class should probably have this

	#CUresult cuDeviceGetByPCIBusId ( CUdevice* dev, const char* pciBusId )
	#CUresult cuDeviceGetPCIBusId ( char* pciBusId, int  len, CUdevice dev )
	#CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels )
	#CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray )
	#CUresult cuMipmappedArrayGetLevel ( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level )

#https://jhui.github.io/2017/03/06/CUDA/
#https://stackoverflow.com/questions/7651450/how-to-create-page-locked-memory-from-a-existing-numpy-array-in-pycuda
#https://stackoverflow.com/questions/33247262/the-corresponding-ctypes-type-of-a-numpy-dtype

# self._handle = handle = mem_alloc_host(self.nbytes)
# # self.handle = handle = mem_host_alloc(self.nbytes, flags)
# weakref.finalize(self, mem_free_host, handle)
# # self.handle = mem_host_get_device_pointer(self.host_handle) #flags

def get_handle(obj):
	#TO DO
		#change get_handle to use/get a memory veiw of the input memory
			#this allows for any input memory, not just numpy arrays or pycu arrays

	if isinstance(obj, np.ndarray):
		handle = obj.ctypes.get_data()
		# ctype = np.ctypeslib.as_ctypes_type(self.dtype)
		# dst = h_ary.ctypes.data_as(ctypes.POINTER(ctype))
	else:
		handle = obj.handle
	return handle

class CuBufferPtr:
	def __init__(self, handle):
		self.handle = handle

	def __repr__(self):
		return f"CuBufferPtr() <{self.handle}>"

	def copy_to_host(self, dst, nbytes, offset = 0, stream = 0):
		#offset is in bytes

		args = [get_handle(dst), self.handle, nbytes] #get_handle(dst) + offset
		memcpy = memcpy_dtoh
		if stream:
			args.append(stream.handle)
			memcpy = memcpy_dtoh_async

		memcpy(*args)

		return dst

	def copy_from_host(self, src, nbytes, offset = 0, stream = 0):
		#offset is in bytes

		args = [self.handle, get_handle(src), nbytes] #get_handle(src) + offset

		memcpy = memcpy_htod
		if stream:
			args.append(stream.handle)
			memcpy = memcpy_htod_async

		memcpy(*args)

	# def memset(self, value, size, itemsize, stream = 0): #offset = 0
		# # if not isinstance(value, np.generic):
		# 	# value = np.result_type(value)(value)
		# 	# #Need to down cast all 64 bit numbers to 32 bit numbers

		# if itemsize == 1:
		# 	memset = cuMemsetD8
		# 	if stream:
		# 		memset = cuMemsetD8Async
		# 	value = x.view(np.uint8)
		# elif itemsize == 2:
		# 	memset = cuMemsetD16
		# 	if stream:
		# 		memset = cuMemsetD16Async
		# 	value = x.view(np.uint16)
		# elif itemsize == 4:
		# 	memset = cuMemsetD32
		# 	if stream:
		# 		memset = cuMemsetD32Async
		# 	value = x.view(np.uint32)
		# else:
		# 	raise ValueError('data type is to large')

		# args = [self.handle, value, size] #self.handle + offset
		# if stream:
		# 	args.append(stream.handle)

		# memset(*args)

class CuBuffer(CuBufferPtr):
	def __init__(self, size, dtype):

		self.dtype = np.dtype(dtype)
		self.size = size

		handle = mem_alloc(self.nbytes)
		weakref.finalize(self, mem_free, handle)

		super().__init__(handle)

	def __repr__(self):
		return f"CuBuffer({self.size}, {self.dtype}) <{self.handle}>"

	@property
	def itemsize(self):
		return self.dtype.itemsize

	@property
	def nbytes(self):
		return self.itemsize*self.size

	def astype(self, dtype):
		self.cast(dtype)

	def cast(self, dtype):
		dtype = np.dtype(dtype)

		if self.dtype != dtype:
			if self.size%dtype.itemsize == 0:
				self.size = self.nbytes//dtype.itemsize
				self.dtype = dtype
			else:
				raise ValueError('dtype does not fit buffer itemsize')

	def copy_to_host(self, dst = None, nbytes = 0, offset = 0, stream = 0):
		if not nbytes:
			nbytes = self.nbytes

		# if nbytes%self.itemsize:
			# raise ValueError('')

		#create host array if one isn't passed in (this array will own the cpu data)
		if dst is None:
			dst = np.empty(nbytes//self.itemsize, dtype = self.dtype)

		# if nbytes > min(dst.nbytes, self.nbytes):
			# raise ValueError('')

		return super().copy_to_host(dst, nbytes, offset, stream)

	def copy_from_host(self, src, nbytes = 0, offset = 0, stream = 0):
		if not nbytes:
			nbytes = self.nbytes

		super().copy_from_host(src, nbytes, offset, stream)

	def memset(self, value, size = 0, stream = 0):
		if not size:
			size = self.size

		super().memset(self.dtype.type(value), size, self.itemsize, stream)

	# def to_device_array(self, shape, dtype = np.float32, strides = None, order = 'C'):
		# #to_cundarray()
		# pass

def _is_contiguous(shape, strides, itemsize):
	for shape, stride in zip(shape, strides):
		if shape > 1 and stride != 0:
			if itemsize != stride:
				return False
			itemsize *= shape
	return True

def is_c_contiguous(shape, strides, itemsize):
	return _is_contiguous(reversed(shape), reversed(strides), itemsize)

def is_f_contiguous(shape, strides, itemsize):
	return _is_contiguous(shape, strides, itemsize)

def is_contiguous(shape, strides, itemsize):
	contiguous = is_c_contiguous(shape, strides, itemsize)
	if not contiguous:
		contiguous = is_f_contiguous(shape, strides, itemsize)
	return contiguous

def check_contiguous(shape, strides, itemsize):
	if not is_contiguous(shape, strides, itemsize):
		raise ValueError('Not contiguous')

def generate_strides(shape, itemsize, order = 'C'):
	if order == 'C':
		shape = reversed(shape)

	strides = []
	stride = itemsize
	for dim in shape:
		strides.insert(0, stride)
		stride *= dim

	if order == 'F':
		strides = reversed(strides)

	return tuple(strides)

class CuNDArray:
	# #for numba interop
	# __cuda_memory__ = True
	# __cuda_ndarray__ = True

	def __init__(self, shape, dtype = np.float32, strides = None, order = 'C'):
		if not isinstance(shape, (tuple, list, np.ndarray)):
			shape = (shape,)
		if strides and not isinstance(strides, (tuple, list, np.ndarray)):
			strides = (strides,)

		self.shape = shape
		self.order = order
		self.dtype = np.dtype(dtype) #type.as_numpy() if isinstance(dtype, pycu.types.Type) else np.dtype(dtype)
		self.strides = generate_strides(self.shape, self.itemsize, self.order) if not strides else strides

		if self.size <= 0:
			raise ValueError('CuNDArray size must be larger than 0.')

		if not is_contiguous(self.shape, self.strides, self.itemsize):
			raise ValueError('The shape and stride do match')

		self._buff = CuBuffer(self.size, self.dtype)

	def __iter__(self):
		return iter([self])

	@property
	def __cuda_array_interface__(self):
		return {'shape': tuple(self.shape),
				'strides': None if is_contiguous(self) else tuple(self.strides),
				'data': (self.handle, False),
				'typestr': self.dtype.name,
				'version': 2}

	@property
	def _pycu_type_(self):
		import pycu #.core.types
		return pycu.core.cuarray(self.ndim, pycu.core.dtype(self.dtype))

	@property
	def itemsize(self):
		return self.dtype.itemsize

	@property
	def handle(self):
		return self._buff.handle
	
	@property
	def nbytes(self):
		return self._buff.nbytes

	@property
	def size(self):
		return np.prod(self.shape)
	
	@property
	def ndim(self):
		return len(self.shape)

	def copy_from_host(self, src, stream = 0, nbytes = 0, offset = 0):
		self._buff.copy_from_host(src, nbytes, offset, stream)

	def copy_to_host(self, dst = None, nbytes = 0, offset = 0, stream = 0):
		# dst = self._buff.copy_to_host(dst, nbytes, offset, stream)
		# #if isinstance(h_ary, np.ndarray):
		# 	# h_ary = h_ary.reshape(self.shape)
		# return dst
		return self._buff.copy_to_host(dst, nbytes, offset, stream)

	def memset(self, value, size, stream = 0):
		self._buff.memset(value, size, stream)

	# def get_address_range(self):
	# 	#CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )
	# 	pass

	# def copy_from_array(self, a_array):
	# 	#CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount )
	# 	pass

	# def copy_from_peer(self, p_ary):
	# 	#CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount )
	# 	#CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream )
	# 	pass

	# def copy_from_device(self, d_ary, stream = 0):
	# 	pass
	# 	#CUresult cuMemcpyDtoD ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount )
	# 	#CUresult cuMemcpyDtoDAsync ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )




# class DevicePitched:
	# pass
	# #CUresult cuMemAllocPitch ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes )
	# #CUresult cuMemFree ( CUdeviceptr dptr )
	# #CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )

	#CUresult cuMemsetD2D16 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height )
	#CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream )
	#CUresult cuMemsetD2D32 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height )
	#CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream )
	#CUresult cuMemsetD2D8 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height )
	#CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream )


# class DeviceArray:
	# #NOT like a Numba device array

	# def copy_from_array(self):
	# 	#CUresult cuMemcpyAtoA ( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount )
	# 	pass


	# def copy_from_host(self):
	# 	#CUresult cuMemcpyHtoA ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount )
	# 	#CUresult cuMemcpyHtoAAsync ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream )
	# 	pass

	# def copy_from_device(self):
	# 	#CUresult cuMemcpyDtoA ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount )
	# 	pass

	# def copy_to_host(self):
	# 	#CUresult cuMemcpyAtoH ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount )
	# 	#CUresult cuMemcpyAtoHAsync ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream )
	# 	pass


class Cu1DArray: #(DeviceArray)
	def __init__(self, width, format, numchannels):
		#height = 0
		pass
	#CUDA_ARRAY_DESCRIPTOR
		# CUarray_format Format
		# size_t  Height
		# unsigned int  NumChannels
		# size_t  Width

	#CUresult cuArrayCreate ( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray )
	#CUresult cuArrayDestroy ( CUarray hArray )
	#CUresult cuArrayGetDescriptor ( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray )

class Cu2DArray: #(DeviceArray):
	def __init__(self, width, height, format, numchannels):
		pass

	#CUDA_ARRAY_DESCRIPTOR
		# CUarray_format Format
		# size_t  Height
		# unsigned int  NumChannels
		# size_t  Width

	#CUresult cuArrayCreate ( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray )
	#CUresult cuArrayDestroy ( CUarray hArray )
	#CUresult cuArrayGetDescriptor ( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray )

	#CUresult cuMemcpy2D ( const CUDA_MEMCPY2D* pCopy )
	#CUresult cuMemcpy2DAsync ( const CUDA_MEMCPY2D* pCopy, CUstream hStream )
	#CUresult cuMemcpy2DUnaligned ( const CUDA_MEMCPY2D* pCopy )

class Cu3DArray:
	def __init__(self, width, height, depth, format, numchannels, flags):
		pass
	#CUDA_ARRAY3D_DESCRIPTOR
		# size_t  Depth
		# size_t  Height
		# size_t  Width
		# unsigned int  NumChannels
		# unsigned int  Flags
		# CUarray_format Format

	#CUresult cuArray3DCreate ( CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray )
	#CUresult cuArray3DGetDescriptor ( CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray )

	#CUresult cuMemcpy3D ( const CUDA_MEMCPY3D* pCopy )
	#CUresult cuMemcpy3DAsync ( const CUDA_MEMCPY3D* pCopy, CUstream hStream )
	#CUresult cuMemcpy3DPeer ( const CUDA_MEMCPY3D_PEER* pCopy )
	#CUresult cuMemcpy3DPeerAsync ( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream )



class PinnedBufferPtr:
	def __init__(self, handle):
		self.handle = handle

	# def __repr__(self):
		# return f"PinnedBuffer({self.size}, {self.dtype}) <{self._handle}>"

	def memmove(self, dst, nbytes, offset = 0):
		handle = get_handle(dst)

		ctypes.memmove(dst, self.handle, nbytes) #self.handle + offset

		return dst

	def get_flags(self):
		#CUresult cuMemHostGetFlags ( unsigned int* pFlags, void* p )
		pass

#Host pinner array
class PinnedBuffer(PinnedBufferPtr):
	def __init__(self, src, flags = 0): #nbytes = 0, offset = 0

		#keep reference to src buffer to prevent it from being garbage collected as long as pinned memory exists
		self.src = src
		handle = get_handle(src)

		mem_host_register(handle, self.nbytes, flags)
		weakref.finalize(self, mem_host_unregister, handle)

		super().__init__(handle)

	@property
	def size(self):
		return self.src.size

	@property
	def dtype(self):
		return self.src.dtype

	@property
	def itemsize(self):
		return self.src.itemsize

	@property
	def nbytes(self):
		return self.src.nbytes

	def memmove(self, dst = None, nbytes = 0, offset = 0):
		if not nbytes:
			nbytes = self.nbytes

		if dst is None:
			dst = np.empty(nbytes//self.itemsize, dtype = self.dtype)

		super().memmove(dst, nbytes, offset)

		return dst

	def __getitem__(self, item):
		#TO DO
			#check if/when this returns copies
				#the copies won't be pinned and this can lead to trouble
		return self.src.__getitem__(item)

	def __setitem__(self, key, value):
		self.src.__setitem__(key, value)

	def __add__(self, other):
		return self.src.__add__(other)

	def __sub__(self, other):
		return self.src.__sub__(other)

	def __mul__(self, other):
		return self.src.__mul__(other)

	def __floordiv__(self, other):
		return self.src.__floordiv__(other)

	def __truediv__(self, other):
		return self.src.__truediv__(other)

	def __mod__(self, other):
		return self.src.__mod__(other)

	def __pow__(self, other):
		return self.src.__pow__(other)

	def __lshift__(self, other):
		return self.src.__lshift__(other)

	def __rshift__(self, other):
		return self.src.__rshift__(other)

	def __and__(self, other):
		return self.src.__and__(other)

	def __xor__(self, other):
		return self.src.__xor__(other)

	def __or__(self, other):
		return self.src.__or__(other)

	def __iadd__(self, other):
		return self.src.__iadd__(other)

	def __isub__(self, other):
		return self.src.__isub__(other)

	def __imul__(self, other):
		return self.src.__imul__(other)

	def __idiv__(self, other):
		return self.src.__idiv__(other)

	def __ifloordiv__(self, other):
		return self.src.__ifloordiv__(other)

	def __imod__(self, other):
		return self.src.__imod__(other)

	def __ipow__(self, other):
		return self.src.__ipow__(other)

	def __ilshift__(self, other):
		return self.src.__ilshift__(other)

	def __irshift__(self, other):
		return self.src.__irshift__(other)

	def __iand__(self, other):
		return self.src.__iand__(other)

	def __ixor__(self, other):
		return self.src.__ixor__(other)

	def __ior__(self, other):
		return self.src.__ior__(other)

	def __neg__(self):
		return self.src.__neg__()

	def __pos__(self):
		return self.src.__pos__()

	def __abs__(self):
		return self.src.__abs__()




class CuNDArrayManaged:
	pass
	#CUresult cuMemAllocManaged ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags )
	#CUresult cuMemFree ( CUdeviceptr dptr )
	#CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )

	#CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount )
	#CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream )


class IPCHandle:
	pass
	#CUresult cuIpcCloseMemHandle ( CUdeviceptr dptr )
	#CUresult cuIpcGetEventHandle ( CUipcEventHandle* pHandle, CUevent event )
	#CUresult cuIpcGetMemHandle ( CUipcMemHandle* pHandle, CUdeviceptr dptr )
	#CUresult cuIpcOpenEventHandle ( CUevent* phEvent, CUipcEventHandle handle )
	#CUresult cuIpcOpenMemHandle ( CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags )




# import numpy as np

# #buffers
# #numba array/numpy array




# # meminfo = ctypes.c_void_p(0)
# # parent = ctypes.c_void_p(0)
# # nitems = c_intp(devary.size)
# # itemsize = c_intp(devary.dtype.itemsize)
# # data = ctypes.c_void_p(driver.device_pointer(devary))
# # for ax in range(arr.ndim):
# # 	args.append(c_intp(arr.shape[ax]))
# # for ax in range(arr.ndim):
# # 	args.append(c_intp(arr.strides[ax]))


# # template<typename T, size_t D>
# # struct NumbaArray{
# # 	void* meminfo;
# # 	void* parent;
# # 	ssize_t size
# # 	ssize_t itemsize
# # 	T* data;
# # 	ssize_t shape[D]
# # 	ssize_t strides[D]
# # }

# # def unpack_nbarray():
# # 	pass


#     # def __init__(self, shape, dtype=float, memptr=None, strides=None, order='C'):
#     #     cdef Py_ssize_t x, itemsize
#     #     cdef tuple s = internal.get_size(shape)
#     #     del shape

#     #     cdef int order_char = (b'C' if order is None else internal._normalize_order(order))

#     #     # `strides` is prioritized over `order`, but invalid `order` should be
#     #     # checked even if `strides` is given.
#     #     if order_char != b'C' and order_char != b'F':
#     #         raise ValueError('order not understood. order=%s' % order)

#     #     # Check for erroneous shape
#     #     self._shape.reserve(len(s))
#     #     for x in s:
#     #         if x < 0:
#     #             raise ValueError('Negative dimensions are not allowed')
#     #         self._shape.push_back(x)
#     #     del s

#     #     # dtype
#     #     self.dtype, itemsize = _dtype.get_dtype_with_itemsize(dtype)

#     #     # Store shape and strides
#     #     if strides is not None:
#     #         if memptr is None:
#     #             raise ValueError('memptr is required if strides is given.')
#     #         self._set_shape_and_strides(self._shape, strides, True, True)
#     #     elif order_char == b'C':
#     #         self._set_contiguous_strides(itemsize, True)
#     #     elif order_char == b'F':
#     #         self._set_contiguous_strides(itemsize, False)
#     #     else:
#     #         assert False

#     #     # data
#     #     if memptr is None:
#     #         self.data = memory.alloc(self.size * itemsize)
#     #         self._index_32_bits = (self.size * itemsize) <= (1 << 31)
#     #     else:
#     #         self.data = memptr
#     #         bound = cupy.core._memory_range.get_bound(self)
#     #         self._index_32_bits = bound[1] - bound[0] <= (1 << 31)




# class DeviceArray:
# 	def __init__(self, size, dtype = c_float):
# 		self.size   = size
# 		self.shape  = shape
# 		self.dtype  = as_ctype(dtype)
# 		self.nbytes = sizeof(dtype)*size

# 	def fill(self, value):
# 		pass
# 		#call kernel to fill array with given value

# 	def as_numpy(self):
# 		pass
# 		# print(ary.ctypes.data_as(ctypes.POINTER(ctype[ary.dtype.name])))

# 	    # @devices.require_context
# 	    # def copy_to_host(self, ary=None, stream=0):
# 	    #     """Copy ``self`` to ``ary`` or create a new Numpy ndarray
# 	    #     if ``ary`` is ``None``.
# 	    #     If a CUDA ``stream`` is given, then the transfer will be made
# 	    #     asynchronously as part as the given stream.  Otherwise, the transfer is
# 	    #     synchronous: the function returns after the copy is finished.
# 	    #     Always returns the host array.
# 	    #     Example::
# 	    #         import numpy as np
# 	    #         from numba import cuda
# 	    #         arr = np.arange(1000)
# 	    #         d_arr = cuda.to_device(arr)
# 	    #         my_kernel[100, 100](d_arr)
# 	    #         result_array = d_arr.copy_to_host()
# 	    #     """
# 	    #     if any(s < 0 for s in self.strides):
# 	    #         msg = 'D->H copy not implemented for negative strides: {}'
# 	    #         raise NotImplementedError(msg.format(self.strides))
# 	    #     assert self.alloc_size >= 0, "Negative memory size"
# 	    #     stream = self._default_stream(stream)
# 	    #     if ary is None:
# 	    #         hostary = np.empty(shape=self.alloc_size, dtype=np.byte)
# 	    #     else:
# 	    #         check_array_compatibility(self, ary)
# 	    #         hostary = ary

# 	    #     if self.alloc_size != 0:
# 	    #         _driver.device_to_host(hostary, self, self.alloc_size, stream=stream)

# 	    #     if ary is None:
# 	    #         if self.size == 0:
# 	    #             hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
# 	    #                                  buffer=hostary)
# 	    #         else:
# 	    #             hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
# 	    #                                  strides=self.strides, buffer=hostary)
# 	    #     return hostary




#     # @property
#     # def __cuda_array_interface__(self):
#     #     if self.device_ctypes_pointer.value is not None:
#     #         ptr = self.device_ctypes_pointer.value
#     #     else:
#     #         ptr = 0

#     #     return {
#     #         'shape': tuple(self.shape),
#     #         'strides': None if is_contiguous(self) else tuple(self.strides),
#     #         'data': (ptr, False),
#     #         'typestr': self.dtype.str,
#     #         'version': 2,
#     #     }






# def host_to_device(val, stream = None):
# 	if isinstance(val, np.ndarray):
# 		return ary_to_device(val, stream)



# def ary_to_device(ary, stream = None):
# 	pass

# def device_array(*args, **kwargs): #dtype = np.float64
# 	#args and kwargs are input to numpy array
# 		#np.array(*args, **kwargs)
# 	pass

# def to_host(val, stream = None):
# 	pass
# 	# if isinstance(val, CUDA_ARRAY):
# 	# 	return ary_to_host(val, stream)

# def ary_to_host(d_ary, stream = None):
# 	pass





# def _is_contiguous(size, shape, strides):
# 	for shape, stride in zip(shape, strides):
# 		if shape > 1 and stride != 0:
# 			if size != stride:
# 				return False
# 			size *= shape
# 	return True

# def is_c_contiguous(ary):
# 	return _is_contiguous(ary.dtype.itemsize, reversed(ary.shape), reversed(ary.strides))

# def is_f_contiguous(ary):
# 	return _is_contiguous(ary.dtype.itemsize, ary.shape, ary.strides)

# def is_contiguous(ary):
# 	contiguous = is_c_contiguous()
# 	if not contiguous:
# 		contiguous = is_f_contiguous()
# 	return contiguous

# 	# core = array_core(ary)
# 	# if not any([core.flags['C_CONTIGUOUS'], core.flags['F_CONTIGUOUS']]):
# 	# 	raise ValueError('NOT CONTIGUOUS')

# 	# def array_core(ary):
# 		# """
# 		# Extract the repeated core of a broadcast array.
# 		# Broadcast arrays are by definition non-contiguous due to repeated
# 		# dimensions, i.e., dimensions with stride 0. In order to ascertain memory
# 		# contiguity and copy the underlying data from such arrays, we must create
# 		# a view without the repeated dimensions.
# 		# """
# 		# if not ary.strides:
# 		# 	return ary
# 		# core_index = []
# 		# for stride in ary.strides:
# 		# 	core_index.append(0 if stride == 0 else slice(None))
# 		# return ary[tuple(core_index)]

# def check_contiguous(ary):
# 	if not is_contiguous(ary):
# 		raise ValueError('NOT CONTIGUOUS')
# 		# errmsg_contiguous_buffer = ("Array contains non-contiguous buffer and cannot "
# 		#							 "be transferred as a single memory region. Please "
# 		#							 "ensure contiguous buffer with numpy "
# 		#							 ".ascontiguousarray()")






# # ary = np.zeros((10,10), dtype = np.float32)
# # # ary.strides = (8,80)
# # # print(ary.flags, is_f_contiguous(ary), is_c_contiguous(ary))
# # # ary.strides = (80,8)
# # # print(ary.flags, is_f_contiguous(ary), is_c_contiguous(ary))

# # import ctypes

# # ctype = {'float32':ctypes.c_float}


# # #Need mapping from numpy dtypes to ctypes

# # print(ary.ctypes.data_as(ctypes.POINTER(ctype[ary.dtype.name])))



# # ary = ary.reshape((5,1,20))
# # ary = ary[2:5]
# # print(ary)

# # is_contiguous(ary)







# def copy_to_device(self, ary, stream=0):
# 	"""Copy `ary` to `self`.
# 	If `ary` is a CUDA memory, perform a device-to-device transfer.
# 	Otherwise, perform a a host-to-device transfer.
# 	"""
# 	if ary.size == 0:
# 		# Nothing to do
# 		return

# 	sentry_contiguous(self)
# 	stream = self._default_stream(stream)

# 	self_core, ary_core = array_core(self), array_core(ary)
# 	if _driver.is_device_memory(ary):
# 		sentry_contiguous(ary)
# 		check_array_compatibility(self_core, ary_core)
# 		_driver.device_to_device(self, ary, self.alloc_size, stream=stream)
# 	else:
# 		# Ensure same contiguity. Only makes a host-side copy if necessary
# 		# (i.e., in order to materialize a writable strided view)
# 		ary_core = np.array(
# 			ary_core,
# 			order='C' if self_core.flags['C_CONTIGUOUS'] else 'F',
# 			subok=True,
# 			copy=not ary_core.flags['WRITEABLE'])
# 		check_array_compatibility(self_core, ary_core)
# 		_driver.host_to_device(self, ary_core, self.alloc_size, stream=stream)

# def host_to_device(dst, src, size, stream=0):
# 	"""
# 	NOTE: The underlying data pointer from the host data buffer is used and
# 	it should not be changed until the operation which can be asynchronous
# 	completes.
# 	"""
# 	varargs = []

# 	if stream:
# 		assert isinstance(stream, Stream)
# 		fn = driver.cuMemcpyHtoDAsync
# 		varargs.append(stream.handle)
# 	else:
# 		fn = driver.cuMemcpyHtoD

# 	fn(device_pointer(dst), host_pointer(src, readonly=True), size, *varargs)


# def host_pointer(obj, readonly=False):
#     """Get host pointer from an obj.
#     If `readonly` is False, the buffer must be writable.
#     NOTE: The underlying data pointer from the host data buffer is used and
#     it should not be changed until the operation which can be asynchronous
#     completes.
#     """
#     if isinstance(obj, (int, long)):
#         return obj

#     forcewritable = False
#     if not readonly:
#         forcewritable = isinstance(obj, np.void) or _is_datetime_dtype(obj)

#     obj = _workaround_for_datetime(obj)
#     return mviewbuf.memoryview_get_buffer(obj, forcewritable, readonly)







# def copy_to_host(self, ary=None, stream=0):
# 	"""Copy ``self`` to ``ary`` or create a new Numpy ndarray
# 	if ``ary`` is ``None``.
# 	If a CUDA ``stream`` is given, then the transfer will be made
# 	asynchronously as part as the given stream.  Otherwise, the transfer is
# 	synchronous: the function returns after the copy is finished.
# 	Always returns the host array.
# 	Example::
# 		import numpy as np
# 		from numba import cuda
# 		arr = np.arange(1000)
# 		d_arr = cuda.to_device(arr)
# 		my_kernel[100, 100](d_arr)
# 		result_array = d_arr.copy_to_host()
# 	"""
# 	if any(s < 0 for s in self.strides):
# 		msg = 'D->H copy not implemented for negative strides: {}'
# 		raise NotImplementedError(msg.format(self.strides))
# 	assert self.alloc_size >= 0, "Negative memory size"
# 	stream = self._default_stream(stream)
# 	if ary is None:
# 		hostary = np.empty(shape=self.alloc_size, dtype=np.byte)
# 	else:
# 		check_array_compatibility(self, ary)
# 		hostary = ary

# 	if self.alloc_size != 0:
# 		_driver.device_to_host(hostary, self, self.alloc_size, stream=stream)

# 	if ary is None:
# 		if self.size == 0:
# 			hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
# 								 buffer=hostary)
# 		else:
# 			hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
# 								 strides=self.strides, buffer=hostary)
# 	return hostary

# def device_to_host(dst, src, size, stream=0):
# 	"""
# 	NOTE: The underlying data pointer from the host data buffer is used and
# 	it should not be changed until the operation which can be asynchronous
# 	completes.
# 	"""
# 	varargs = []

# 	if stream:
# 		assert isinstance(stream, Stream)
# 		fn = driver.cuMemcpyDtoHAsync
# 		varargs.append(stream.handle)
# 	else:
# 		fn = driver.cuMemcpyDtoH

# 	fn(host_pointer(dst), device_pointer(src), size, *varargs)




















	# def copy_to_host_dtype(self, dtype = None, stream = 0, nbytes = 0, offset = 0):
		# if not nbytes:
		# 	nbytes = self.nbytes

		# _dtype = self.dtype
		# if dtype:
		# 	self.cast(dtype)

		# h_ary = np.empty(nbytes//self.itemsize, dtype = dtype)

		# args = [h_ary.ctypes.get_data(), self.handle, nbytes]
		# memcpy = memcpy_dtoh
		# if stream:
		# 	args.append(stream.handle)
		# 	memcpy = memcpy_dtoh_async

		# memcpy(*args)

		# if dtype:
		# 	self.cast(_dtype)

		# return h_ary

	# def copy_to_host(self, h_ary = None, stream = 0, nbytes = 0, offset = 0):
		# if not nbytes:
		# 	nbytes = self.nbytes

		# #create host array if one isn't passed in (this array will own the cpu data)
		# if h_ary is None:
		# 	h_ary = np.empty(self.size, dtype = self.dtype)
		# 	#self.size -> nbytes//self.dtype.itemsize

		# if isinstance(h_ary, np.ndarray):
		# 	dst = h_ary.ctypes.get_data()
		# else:
		# 	dst = h_ary.handle

		# args = [dst, self.handle, nbytes] #self.handle + offset (offset needs to be in bytes)
		# memcpy = memcpy_dtoh
		# if stream:
		# 	args.append(stream.handle)
		# 	memcpy = memcpy_dtoh_async

		# memcpy(*args)

		# return h_ary








def to_device(ndarray, stream = 0):
	order = "F" if ndarray.flags.fnc else 'C'
	d_ndarray = CuNDArray(ndarray.shape, ndarray.dtype, ndarray.strides, order)
	d_ndarray.copy_from_host(ndarray, stream = stream)
	return d_ndarray

def to_host(d_ary, stream = 0, nbytes = 0, offset = 0):
	return d_ary.copy_to_host(stream = stream, nbytes = nbytes, offset = offset)

def copy_to_device(*args, **kwargs):
	return to_device(*args, **kwargs)

def copy_to_host(*args, **kwargs):
	return to_host(*args, **kwargs)

def to_device_buffer(ndarray, stream = 0):
	d_buff = CuBuffer(ndarray.size, dtype = ndarray.dtype)
	d_buff.copy_from_host(ndarray, stream = stream)
	return d_buff

def device_buffer(size, dtype = np.float32):
	return CuBuffer(size, dtype)

def device_array(shape, dtype = np.float32):
	return CuNDArray(shape, dtype)

def device_ndarray(shape, dtype = np.float32):
	return CuNDArray(shape, dtype)

def pinned_buffer(size, dtype, flags = 0):
	return PinnedBuffer(np.empty(size, dtype = dtype), flags = flags)

def pin_host_array(h_ary, flags = 0): #nbytes = 0, flags = 0, offset = 0
	return PinnedBuffer(h_ary, flags = flags)



def _cubuffer_type(typ, arg):
	return arg.handle

def _cundarray_type(typ, arg):
	pass
	# fields = [("meminfo", pointer(int8)), #pointer(int8) #pointer(void)
	# 		  ("parent", pointer(int8)), #pointer(int8) #pointer(void)
	# 		  ("nitems", ssize_t),
	# 		  ("itemsize", ssize_t),
	# 		  ("data", pointer(dtype)), #pointer(float32) #pointer(void)
	# 		  ("shape", ShapeStride(ndim)), #Pointer(ShapeStride(ndim)) #Array(ndim, ssize_t)
	# 		  ("strides", ShapeStride(ndim))] #Pointer(ShapeStride(ndim)) #Array(ndim, ssize_t)

	# class ShapeStride(Tuple):
		# def __init__(self, ndim):
		# 	self.ndim = ndim

		# 	####################
		# 	members = ['x','y','z']
		# 	members = members[:ndim]
		# 	if ndim > 3:
		# 		members = generate_member_name(ndim)
		# 	####################

		# 	fields = [(members[n], ssize_t) for n in range(ndim)]

		# 	super().__init__(fields)

		# def _construct_ctypes_values(self, fields, shape_stride):
		# 	if not isinstance(shape_stride, (list, tuple, np.ndarray)):
		# 		raise ValueError('Not supported')

		# 	return {field[0]:val for field, val in zip(self.fields, shape_stride)}

ArgPreparer.add_preparer(CuBufferPtr, _cubuffer_type)
ArgPreparer.add_preparer(CuBuffer, _cubuffer_type)
ArgPreparer.add_preparer(CuNDArray, _cundarray_type)
# ArgPreparer.push_type(Cu1DArray, None)
# ArgPreparer.push_type(Cu2DArray, None)
# ArgPreparer.push_type(Cu3DArray, None)








# def device_1darray(shape, dtype = float32):
	# return Cu1DArray(shape, dtype)

# def device_2darray(shape, dtype = float32):
	# return Cu2DArray(shape, dtype)

# def device_3darray(shape, dtype = float32):
	# return Cu3DArray(shape, dtype)

