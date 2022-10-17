# from pycu.core.py_allocator import py_malloc, py_free
from pycu.driver import (mem_alloc, mem_free, memcpy_dtoh, memcpy_dtoh_async, memcpy_htod, memcpy_htod_async,
						 mem_alloc_host, mem_free_host, mem_host_register, mem_host_unregister,
						 memset_D16, memset_D16_async,
						 memset_D32, memset_D32_async,
						 memset_D8, memset_D8_async,
						 CUdeviceptr)

import numpy as np
import ctypes
import weakref
import warnings

def get_handle(obj):
	# _CData = ctypes._SimpleCData.__mro__[-2]

	#TO DO
		#change get_handle to use/get a memory veiw of the input memory
			#this allows for any input memory, not just numpy arrays, ctypes objects, or pycu arrays

	if isinstance(obj, np.ndarray):
		handle = obj.ctypes.get_data()
		# ctype = np.ctypeslib.as_ctypes_type(self.dtype)
		# dst = h_ary.ctypes.data_as(ctypes.POINTER(ctype))
	elif isinstance(obj, (ctypes._SimpleCData, ctypes._Pointer,
						  ctypes.Array, ctypes.Structure, ctypes.Union)): #isinstance(obj, _CData)
		handle = ctypes.addressof(obj)
	else:
		try:
			handle = obj.handle
		except AttributeError:
			print(f"A handle to the device memory owned by {obj} could not be found.")

	return handle

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

class CuBuffer:
	def __init__(self, nbytes, * , handle = None, auto_free = True):
		if handle is None:
			handle = mem_alloc(nbytes)
		# else:
			# handle = CUdeviceptr(handle)

		if auto_free:
			self._finalizer = weakref.finalize(self, mem_free, handle)

		self.nbytes = nbytes
		self.handle = handle
		self.auto_free = auto_free

	def __repr__(self):
		return f"CuBuffer({self.nbytes}) <{int(self)}>"

	def __len__(self):
		return self.nbytes

	def __int__(self):
		return self.handle.value

	def __index__(self):
		return int(self)

	@property
	def __cuda_array_interface__(self):
		dtype = np.dtype(np.uint8)
		return {'shape': (self.nbytes,),
				'strides': (dtype.itemsize,),
				'typestr': dtype.name,
				'descr': dtype.descr,
				'data': (self.handle.value, False),
				'version': 3}

	def copy_to_host(self, dst, nbytes = np.uint64(-1), offset = 0, stream = 0):
		nbytes = min(nbytes, self.nbytes)
		args = [get_handle(dst), self.handle, nbytes] # + ctypes.c_uint64(offset)
		memcpy = memcpy_dtoh
		if stream:
			args.append(stream.handle)
			memcpy = memcpy_dtoh_async

		memcpy(*args)

		return dst

		#offset must be a multiple of the itemsize

	def copy_from_host(self, src, nbytes = np.uint64(-1), offset = 0, stream = 0):
		nbytes = min(nbytes, self.nbytes)
		args = [self.handle, get_handle(src), nbytes] # + ctypes.c_uint64(offset)

		memcpy = memcpy_htod
		if stream:
			args.append(stream.handle)
			memcpy = memcpy_htod_async

		memcpy(*args)

	def _memset(self, memset, memset_async, value, size, offset, stream):
		args = [self.handle, value, size] # + ctypes.c_uint64(offset)

		if stream:
			args.append(stream.handle)
			memset = memset_async
		memset(*args)

		# # # if not isinstance(value, np.generic):
		# # 	# value = np.result_type(value)(value)
		# # 	# #Need to down cast all 64 bit numbers to 32 bit numbers

		# if itemsize == 1:
		# 	memset = cuMemsetD8
		# 	if stream:
		# 		memset = cuMemsetD8Async
		# 	value = value.view(np.uint8)
		# elif itemsize == 2:
		# 	memset = cuMemsetD16
		# 	if stream:
		# 		memset = cuMemsetD16Async
		# 	value = value.view(np.uint16)
		# elif itemsize == 4:
		# 	memset = cuMemsetD32
		# 	if stream:
		# 		memset = cuMemsetD32Async
		# 	value = value.view(np.uint32)
		# # else:
		# # 	raise ValueError('data type is to large')

		# # args = [self.handle, value, size] #self.handle + offset
		# # if stream:
		# # 	args.append(stream.handle)

		# # memset(*args)

	def memset(self, value, size = np.uint64(-1), offset = 0, stream = 0):
		self.memsetD8(value, size, offset, stream)

	def memsetD8(self, value, size = np.uint64(-1), offset = 0, stream = 0):
		size = min(self.nbytes, size)
		self._memset(memset_D8, memset_D8_async, value, size, offset, stream)

	def memsetD16(self, value, size = np.uint64(-1), offset = 0, stream = 0):
		size = min(self.nbytes//2, size)
		self._memset(memset_D16, memset_D16_async, value, size, offset, stream)

	def memsetD32(self, value, size = np.uint64(-1), offset = 0, stream = 0):
		size = min(self.nbytes//4, size)
		self._memset(memset_D32, memset_D32_async, value, size, offset, stream)


class CuArray(CuBuffer):
	def __init__(self, size, dtype, * , handle = None, auto_free = True):
		self.size = size
		self.dtype = np.dtype(dtype)

		super().__init__(self.dtype.itemsize * self.size, handle = handle, auto_free = auto_free)

	def __repr__(self):
		return f"CuArray({self.size}, {self.dtype}) <{self.handle}>"

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		if isinstance(idx, slice):
			start = idx.start if idx.start != None else 0
			stop = idx.stop if idx.stop != None else self.size - 1
			size = stop - start
			handle = CUdeviceptr(self.handle.value + start * self.dtype.itemsize)
		else:
			size = 1
			handle = CUdeviceptr(self.handle.value + idx * self.dtype.itemsize)

		return CuArray(size, self.dtype, handle = handle, auto_free = False)

	@property
	def __cuda_array_interface__(self):
		return {'shape': (self.size,),
				'strides': (self.dtype.itemsize,),
				'typestr': self.dtype.name,
				'descr': self.dtype.descr,
				'data': (self.handle.value, False),
				'version': 3}

	@property
	def itemsize(self):
		return self.dtype.itemsize

	def view(self, dtype):
		dtype = np.dtype(dtype)

		if self.dtype != dtype:
			if self.size%dtype.itemsize == 0:
				self.size = self.nbytes//dtype.itemsize
				self.dtype = dtype
			else:
				raise ValueError('dtype does not fit buffer size')
		# if self.dtype != dtype:
		# 	dtype = np.dtype(dtype)
		# 	if self.size%dtype.itemsize == 0:
		# 		return CuArray(self.nbytes//dtype.itemsize, dtype, handle = self.handle, auto_free = False)
		# 	else:
		# 		raise ValueError('dtype does not fit buffer size')
		# else:
		# 	return self

	def copy_to_host(self, dst = None, nbytes = np.uint64(-1), offset = 0, stream = 0):
		nbytes = min(nbytes, self.nbytes)

		if nbytes%self.itemsize:
			raise ValueError('')

		if offset%self.itemsize:
			raise ValueError('')

		#create host array if one isn't passed in (this array will own the cpu data)
		if dst is None:
			dst = np.empty(nbytes//self.itemsize, dtype = self.dtype)

		return super().copy_to_host(dst, nbytes, offset, stream)

	def copy_from_host(self, src, nbytes = np.uint64(-1), offset = 0, stream = 0):
		if nbytes%self.itemsize:
			raise ValueError('')

		if offset%self.itemsize:
			raise ValueError('')

		return super().copy_from_host(src, nbytes, offset, stream)

	def memset(self, value, size = np.uint64(-1), stream = 0):
		size = min(size, self.size)

		memset = super().memset
		if self.itemsize == 1:
			memset = super().memsetD8
		elif self.itemsize == 2:
			memset = super().memsetD16
		elif self.itemsize == 4:
			memset = super().memsetD32
		else:
			memset = super().memsetD8
			size = size*self.itemsize

		memset(value, size, stream)

	# def as_device_ndarray(self, shape): # dtype = np.float32, strides = None, order = 'C'
		# finalizer.detach() #prevents the memory from being garbage collected when transfering ownership of the pointer
		# 	#need to save the finalizer so it can be referenced
		# pass

class CuNDArray(CuArray):
	# #for numba interop
	# __cuda_memory__ = True
	# __cuda_ndarray__ = True

	def __init__(self, shape, dtype = np.float32, strides = None, order = 'C', * , handle = None, auto_free = True):
		if not isinstance(shape, (tuple, list, np.ndarray)):
			shape = (shape,)
		if strides and not isinstance(strides, (tuple, list, np.ndarray)):
			strides = (strides,)

		super().__init__(np.prod(shape), dtype, handle = handle, auto_free = auto_free)

		self.shape = shape
		self.order = order
		self.strides = generate_strides(self.shape, self.itemsize, self.order) if not strides else strides

		if self.size <= 0:
			raise ValueError('CuNDArray size must be non-zero.')

		if not is_contiguous(self.shape, self.strides, self.itemsize):
			raise ValueError(f'The shape and stride provided to {self} are not compatible')

	def __repr__(self):
		return f"CuNDArray({self.shape}, {self.dtype}) <{self.handle}>"

	def __len__(self):
		return self.size

	@property
	def __cuda_array_interface__(self):
		return {'shape': tuple(self.shape),
				'strides': None if is_contiguous(self.shape, self.strides, self.itemsize) else tuple(self.strides),
				'typestr': self.dtype.name,
				'descr': self.dtype.descr,
				'data': (self.handle, False),
				'version': 3}

	@property
	def device_pointer(self):
		return self.handle

	@property
	def ndim(self):
		return len(self.shape)

	def copy_from_host(self, src, stream = 0): #nbytes = np.uint64(-1), offset = 0
		super().copy_from_host(src, stream = stream)

	def copy_to_host(self, dst = None, stream = 0): #nbytes = np.uint64(-1), offset = 0
		dst = super().copy_to_host(dest, stream = stream)
		dst = dst.reshape(self.shape)
		return dst

	def as_device_array(self, replace = True):
		#replace transfers ownership of the pointer to the CuArray if auto_free was set to True
		auto_free = self.auto_free and replace
		if auto_free:
			#prevents the memory from being garbage collected when transfering ownership of the pointer
			self._finalizer.detach()
		return CuArray(self.size, self.dtype, handle = self.handle, auto_free = auto_free)

	# # def memset(self, value, size, stream = 0):
	# # 	self.memset(value, size, stream)




	# # def get_address_range(self):
	# # 	#CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )
	# # 	pass

	# # def copy_from_array(self, a_array):
	# # 	#CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount )
	# # 	pass

	# # def copy_from_peer(self, p_ary):
	# # 	#CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount )
	# # 	#CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream )
	# # 	pass

	# # def copy_from_device(self, d_ary, stream = 0):
	# # 	pass
	# # 	#CUresult cuMemcpyDtoD ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount )
	# # 	#CUresult cuMemcpyDtoDAsync ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )




#CUresult cuMemGetInfo ( size_t* free, size_t* total )
	#A device class should probably have this

#CUresult cuDeviceGetByPCIBusId ( CUdevice* dev, const char* pciBusId )
#CUresult cuDeviceGetPCIBusId ( char* pciBusId, int  len, CUdevice dev )
#CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels )
#CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray )
#CUresult cuMipmappedArrayGetLevel ( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level )


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

	def __repr__(self):
		return f"PinnedBufferPtr() <{self._handle}>"

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

	def __repr__(self):
		return f"PinnedBuffer({self.size}, {self.dtype}) <{self._handle}>"

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









def to_device_ndarray(ndarray, stream = 0):
	order = "F" if ndarray.flags.fnc else 'C'
	d_ndarray = CuNDArray(ndarray.shape, ndarray.dtype, ndarray.strides, order)
	d_ndarray.copy_from_host(ndarray, stream = stream)
	return d_ndarray

def to_device_array(ary, stream = 0, auto_free = True):
	d_ary = CuArray(ary.size, ary.dtype, auto_free = auto_free)
	d_ary.copy_from_host(ary, stream = stream)
	return d_ary

def to_host(d_ary, stream = 0, nbytes = 0, offset = 0):
	return d_ary.copy_to_host(stream = stream, nbytes = nbytes, offset = offset)

def copy_to_ndarray(*args, **kwargs):
	return to_device_ndarray(*args, **kwargs)

def copy_to_array(*args, **kwargs):
	return to_device_array(*args, **kwargs)

def copy_to_host(*args, **kwargs):
	return to_host(*args, **kwargs)

def device_buffer(nbytes, auto_free = True):
	return CuBuffer(nbytes, auto_free = auto_free)

def device_array(size, dtype = np.float32, auto_free = True):
	return CuArray(size, dtype, auto_free = auto_free)

def device_ndarray(shape, dtype = np.float32, auto_free = True):
	return CuNDArray(shape, dtype, auto_free = auto_free)

def pinned_buffer(size, dtype, flags = 0):
	return PinnedBuffer(np.empty(size, dtype = dtype), flags = flags)

def pin_host_array(h_ary, flags = 0): #nbytes = 0, flags = 0, offset = 0
	return PinnedBuffer(h_ary, flags = flags)

#buff
#ary
#ary1d
#ary2d
#ary3d
#arynd
