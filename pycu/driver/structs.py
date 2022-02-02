from .enums import *
from .typedefs import *
from .defines import CU_IPC_HANDLE_SIZE

from ctypes import *


class CUuuid_st(Structure):
	_fields_ = [("bytes", c_char*16)]
CUuuid = CUuuid_st


class CUDA_ARRAY3D_DESCRIPTOR_st(Structure):
	_fields_ = [("Width", c_size_t),
				("Height", c_size_t),
				("Depth", c_size_t),
				("Format", CUarray_format),
				("NumChannels", c_uint),
				("Flags", c_uint)]
CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_st


class CUDA_ARRAY_DESCRIPTOR_st(Structure):
	_fields_ = [("Width", c_size_t),
				("Height", c_size_t),
				("Format", CUarray_format),
				("NumChannels", c_uint)]
CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_st


class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(Structure):
	_fields_ = [("offset", c_ulonglong),
				("size", c_ulonglong),
				("flags", c_uint),
				("reserved", c_uint*16)]
CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st

class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(Structure):
	class handle(Union):
		class win32(Structure):
			_fields_ = [("handle", c_void_p),
						("name", c_void_p)]

		_fields_ = [("fd", c_int),
					("win32", win32)]

	_fields_ = [("type", CUexternalMemoryHandleType),
				("handle", handle),
				("size", c_ulonglong),
				("flags", c_uint),
				("reserved", c_uint*16)]			
CUDA_EXTERNAL_MEMORY_HANDLE_DESC = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st


class CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st(Structure):
	_fields_ = [("offset", c_ulonglong),
				("arrayDesc", CUDA_ARRAY3D_DESCRIPTOR),
				("numLevels", c_uint),
				("reserved", c_uint*16),]
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st


class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(Structure):
	class handle(Union):
		class win32(Structure):
			_fields_ = [("handle", c_void_p),
						("name", c_void_p)]

		_fields_ = [("fd", c_int),
					("win32", win32)]

	_fields_ = [("type", CUexternalSemaphoreHandleType),
				("handle", handle),
				("flags", c_uint),
				("reserved", c_uint*16)]
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st


class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(Structure):
	class params(Structure):
		class fence(Structure):
			_fields_ = [("value", c_ulonglong)]

		_fields_ = [("fence", fence),
					("reserved", c_uint*16),]

	_fields_ = [("params", params),
				("flags", c_uint),
				("reserved", c_uint*16)]
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st


class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(Structure):
	class params(Structure):
		class fence(Structure):
			_fields_ = [("value", c_ulonglong)]

		_fields_ = [("fence", fence),
					("reserved", c_uint*16),]

	_fields_ = [("params", params),
				("flags", c_uint),
				("reserved", c_uint*16)]
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st


class CUDA_HOST_NODE_PARAMS_st(Structure):
	_fields_ = [("fn", CUhostFn),
				("userData", c_void_p)]
CUDA_HOST_NODE_PARAMS = CUDA_HOST_NODE_PARAMS_st


class CUDA_KERNEL_NODE_PARAMS_st(Structure):
	_fields_ = [("func", CUfunction),
				("gridDimX", c_uint),
				("gridDimY", c_uint),
				("gridDimZ", c_uint),
				("blockDimX", c_uint),
				("blockDimY", c_uint),
				("blockDimZ", c_uint),
				("sharedMemBytes", c_uint),
				("kernelParams", POINTER(c_void_p)),
				("extra", POINTER(c_void_p))]
CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_st


class CUDA_LAUNCH_PARAMS_st(Structure):
	_fields_ = [("function", CUfunction),
				("gridDimX", c_uint),
				("gridDimY", c_uint),
				("gridDimZ", c_uint),
				("blockDimX", c_uint),
				("blockDimY", c_uint),
				("blockDimZ", c_uint),
				("sharedMemBytes", c_uint),
				("hStream", CUstream),
				("kernelParams", POINTER(c_void_p))]
CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_st


class CUDA_MEMCPY2D_st(Structure):
	_fields_ = [("srcXInBytes", c_size_t),
				("srcY", c_size_t),
				("srcMemoryType", CUmemorytype),
				("srcHost", c_void_p),
				("srcDevice", CUdeviceptr),
				("srcArray", CUarray),
				("srcPitch",c_size_t),
				("dstXInBytes", c_size_t),
				("dstY", c_size_t),
				("dstMemoryType", CUmemorytype),
				("dstHost", c_void_p),
				("dstDevice", CUdeviceptr),
				("dstArray", CUarray),
				("dstPitch", c_size_t),
				("WidthInBytes", c_size_t),
				("Height", c_size_t)]
CUDA_MEMCPY2D = CUDA_MEMCPY2D_st


class CUDA_MEMCPY3D_st(Structure):
	_fields_ = [("srcXInBytes", c_size_t),
				("srcY", c_size_t),
				("srcZ", c_size_t),
				("srcLOD", c_size_t),
				("srcMemoryType", CUmemorytype),
				("srcHost", c_void_p),
				("srcDevice", CUdeviceptr),
				("srcArray", CUarray),
				("reserved0", c_void_p),
				("srcPitch", c_size_t),
				("srcHeight", c_size_t),
				("dstXInBytes", c_size_t),
				("dstY", c_size_t),
				("dstZ", c_size_t),
				("dstLOD", c_size_t),
				("dstMemoryType", CUmemorytype),
				("dstHost", c_void_p),
				("dstDevice", CUdeviceptr),
				("dstArray", CUarray),
				("reserved1", c_void_p),
				("dstPitch", c_size_t),
				("dstHeight", c_size_t),
				("WidthInBytes", c_size_t),
				("Height", c_size_t),
				("Depth", c_size_t)]
CUDA_MEMCPY3D = CUDA_MEMCPY3D_st


class CUDA_MEMCPY3D_PEER_st(Structure):
	_fields_ = [("srcXInBytes", c_size_t),
				("srcY", c_size_t),
				("srcZ", c_size_t),
				("srcLOD", c_size_t),
				("srcMemoryType", CUmemorytype),
				("srcHost", c_void_p),
				("srcDevice", CUdeviceptr),
				("srcArray", CUarray),
				("srcContext", CUcontext),
				("srcPitch", c_size_t),
				("srcHeight", c_size_t),
				("dstXInBytes", c_size_t),
				("dstY", c_size_t),
				("dstZ", c_size_t),
				("dstLOD", c_size_t),
				("dstMemoryType", CUmemorytype),
				("dstHost", c_void_p),
				("dstDevice", CUdeviceptr),
				("dstArray", CUarray),
				("dstContext", CUcontext),
				("dstPitch", c_size_t),
				("dstHeight", c_size_t),
				("WidthInBytes", c_size_t),
				("Height", c_size_t),
				("Depth", c_size_t)]
CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_st


class CUDA_MEMSET_NODE_PARAMS_st(Structure):
	_fields_ = [("dst", CUdeviceptr),
				("pitch", c_size_t),
				("value", c_uint),
				("elementSize", c_uint),
				("width", c_size_t),
				("height", c_size_t)]
CUDA_MEMSET_NODE_PARAMS = CUDA_MEMSET_NODE_PARAMS_st


class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st(Structure):
	_fields_ = [("p2pToken", c_ulonglong),
				("vaSpaceToken", c_uint)]
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st


class CUDA_RESOURCE_DESC_st(Structure):
	class res(Union):
		class array(Structure):
			_fields_ = [("hArray", CUarray)]
		class mipmap(Structure):
			_fields_ = [("hMipmappedArray", CUmipmappedArray)]
		class linear(Structure):
			_fields_ = [("devPtr", CUdeviceptr),
						("format", CUarray_format),
						("numChannels", c_uint),
						("sizeInBytes", c_size_t),]
		class pitch2D(Structure):
			_fields_ = [("devPtr", CUdeviceptr),
						("format", CUarray_format),
						("numChannels", c_uint),
						("width", c_size_t),
						("height", c_size_t),
						("pitchInBytes", c_size_t)]
		class reserved(Structure):
			_fields_ = [("reserved", c_int*32)]

		_fields_ = [("array", array),
					("mipmap", mipmap),
					("linear", linear),
					("pitch2D", pitch2D),
					("reserved", reserved)]

	_fields_ = [("resType", CUresourcetype),
				("res", res),
				("flags", c_uint)]
CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_st


class CUDA_RESOURCE_VIEW_DESC_st(Structure):
	_fields_ = [("format", CUresourceViewFormat),
				("width", c_size_t),
				("height", c_size_t),
				("depth", c_size_t),
				("firstMipmapLevel", c_uint),
				("lastMipmapLevel", c_uint),
				("firstLayer", c_uint),
				("lastLayer", c_uint),
				("reserved", c_uint*16)]
CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_st


class CUDA_TEXTURE_DESC_st(Structure):
	_fields_ = [("addressMode", CUaddress_mode*3),
				("filterMode", CUfilter_mode),
				("flags", c_uint),
				("maxAnisotropy", c_uint),
				("mipmapFilterMode", CUfilter_mode),
				("mipmapLevelBias", c_float),
				("minMipmapLevelClamp", c_float),
				("maxMipmapLevelClamp", c_float),
				("borderColor", c_float*4),
				("reserved", c_int*12)]
CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_st


class CUdevprop_st(Structure):
	_fields_ = [("maxThreadsPerBlock", c_int),
				("maxThreadsDim", c_int*3),
				("maxGridSize", c_int*3),
				("sharedMemPerBlock", c_int),
				("totalConstantMemory", c_int),
				("SIMDWidth", c_int),
				("memPitch", c_int),
				("regsPerBlock", c_int),
				("clockRate", c_int),
				("textureAlign", c_int)]
CUdevprop = CUdevprop_st


class CUipcEventHandle_st(Structure):
	_fields_ = [("reserved", c_char*CU_IPC_HANDLE_SIZE.value)]
CUipcEventHandle = CUipcEventHandle_st


class CUipcMemHandle_st(Structure):
	_fields_ = [("reserved", c_char*CU_IPC_HANDLE_SIZE.value)]
CUipcMemHandle = CUipcMemHandle_st


# class CUeglFrame(Structure):
# 	_fields_ = [("", ),]


class CUaccessPolicyWindow_st(Structure):
	_fields_ = [("base_ptr", c_void_p),
				("num_bytes", c_size_t),
				("hitRatio", c_float),
				("hitProp", CUaccessProperty),
				("missProp", CUaccessProperty)]
CUaccessPolicyWindow = CUaccessPolicyWindow_st


# class CUkernelNodeAttrValue(Structure):
	# _fields_ = [("", )]


# class CUmemAccessDesc(Structure):
	# _fields_ = [("", )]


# class CUmemAllocationProp(Structure):
	# _fields_ = [("", )]


# class CUmemLocation(Structure):
	# _fields_ = [("", )]


class CUstreamAttrValue_union(Union):
	_fields_ = [("accessPolicyWindow", CUaccessPolicyWindow),
				("syncPolicy", CUsynchronizationPolicy)]
CUstreamAttrValue = CUstreamAttrValue_union

class CUstreamBatchMemOpParams_union(Union):
	class waitValue(Structure):
		class union(Union):
			_fields_ = [("value", cuuint32_t),
						("value64", cuuint64_t)]

		_anonymous_ = ("union",)
		_fields_ = [("operation", CUstreamBatchMemOpType),
					("address", CUdeviceptr),
					("union", union),
					("flags", c_uint),
					("alias", CUdeviceptr)]

	class writeValue(Structure):
		class union(Union):
			_fields_ = [("value", cuuint32_t),
						("value64", cuuint64_t)]

		_anonymous_ = ("union",)
		_fields_ = [("operation", CUstreamBatchMemOpType),
					("address", CUdeviceptr),
					("union", union),
					("flags", c_uint),
					("alias", CUdeviceptr)]

	class flushRemoteWrites(Structure):
		_fields_ = [("operation", CUstreamBatchMemOpType),
					("flags", c_uint)]

	_fields_ = [("operation", CUstreamBatchMemOpType),
				("waitValue", waitValue),
				("writeValue", writeValue),
				("flushRemoteWrites", flushRemoteWrites),
				("pad", cuuint64_t*6)]
CUstreamBatchMemOpParams = CUstreamBatchMemOpParams_union






