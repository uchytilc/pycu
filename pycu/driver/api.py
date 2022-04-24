from .typedefs import *
from .structs import *
from .enums import *

from ctypes import *

# def filter_api(api, filter = []):
# 	for func in filter:
# 		api.pop(func, None)

#format of prototypes: (return, arg1, arg2, ..., argN)

def API_PROTOTYPES_08000():
	api = {
		#ERROR HANDLING
			"cuGetErrorName": (CUresult, CUresult, POINTER(c_char_p)),
			"cuGetErrorString": (CUresult, CUresult, POINTER(c_char_p)),
		#INITILIZATION
			"cuInit": (CUresult, c_uint),
		#VERSION MANAGEMENT
			"cuDriverGetVersion": (CUresult, POINTER(c_int)),
		#DEVICE MANAGEMENT
			"cuDeviceGet": (CUresult, POINTER(CUdevice), c_int),
			"cuDeviceGetAttribute": (CUresult, POINTER(c_int), CUdevice_attribute, CUdevice),
			"cuDeviceGetCount": (CUresult, POINTER(c_int)),
			"cuDeviceGetName": (CUresult, c_char_p, c_int, CUdevice),
			"cuDeviceTotalMem_v2": (CUresult, POINTER(c_size_t), CUdevice),
			#DEPRECATED
			"cuDeviceComputeCapability":(CUresult, POINTER(c_int), POINTER(c_int), CUdevice),
			"cuDeviceGetProperties":(CUresult, POINTER(CUdevprop), CUdevice),
		#PRIMARY CONTEXT MANAGEMENT
			"cuDevicePrimaryCtxGetState":(CUresult, CUdevice, POINTER(c_uint), POINTER(c_int)),
			"cuDevicePrimaryCtxRelease":(CUresult, CUdevice),
			"cuDevicePrimaryCtxReset":(CUresult, CUdevice),
			"cuDevicePrimaryCtxRetain":(CUresult, POINTER(CUcontext), CUdevice),
			"cuDevicePrimaryCtxSetFlags":(CUresult, CUdevice, c_uint),
		#CONTEXT MANAGEMENT
			"cuCtxCreate_v2":(CUresult, POINTER(CUcontext), c_uint, CUdevice),
			"cuCtxDestroy_v2":(CUresult, CUcontext),
			"cuCtxGetApiVersion":(CUresult, CUcontext, POINTER(c_uint)),
			"cuCtxGetCacheConfig":(CUresult, POINTER(CUfunc_cache)),
			"cuCtxGetCurrent":(CUresult, POINTER(CUcontext)),
			"cuCtxGetDevice":(CUresult, POINTER(CUdevice)),
			"cuCtxGetFlags":(CUresult, POINTER(c_uint)),
			"cuCtxGetLimit":(CUresult, POINTER(c_size_t), CUlimit),
			"cuCtxGetSharedMemConfig":(CUresult, POINTER(CUsharedconfig)),
			"cuCtxGetStreamPriorityRange":(CUresult, POINTER(c_int), POINTER(c_int)),
			"cuCtxPopCurrent_v2":(CUresult, POINTER(CUcontext)),
			"cuCtxPushCurrent_v2":(CUresult, POINTER(CUcontext)),
			"cuCtxSetCacheConfig":(CUresult, CUfunc_cache),
			"cuCtxSetCurrent":(CUresult, CUcontext),
			"cuCtxSetLimit":(CUresult, CUlimit, c_size_t),
			"cuCtxSetSharedMemConfig":(CUresult, CUsharedconfig),
			"cuCtxSynchronize":(CUresult, ),
			############
			#DEPRECATED#
			############
			"cuCtxAttach": (CUresult, POINTER(CUcontext), c_uint),
			"cuCtxDetach": (CUresult, CUcontext),
		#MODULE MANAGEMENT
			"cuLinkAddData_v2":(CUresult, CUlinkState, CUjitInputType, c_void_p, c_size_t, c_char_p, c_uint, POINTER(CUjit_option), POINTER(c_void_p)),
			"cuLinkAddFile_v2":(CUresult, CUlinkState, CUjitInputType, c_char_p, c_uint, POINTER(CUjit_option), POINTER(c_void_p)),
			"cuLinkComplete":(CUresult, CUlinkState, POINTER(c_void_p), POINTER(c_size_t)),
			"cuLinkCreate_v2":(CUresult, c_uint, POINTER(CUjit_option), POINTER(c_void_p), POINTER(CUlinkState)),
			"cuLinkDestroy":(CUresult, CUlinkState),
			"cuModuleGetFunction":(CUresult, POINTER(CUfunction), CUmodule, c_char_p),
			"cuModuleGetGlobal_v2":(CUresult, POINTER(CUdeviceptr), POINTER(c_size_t), CUmodule, c_char_p),
			"cuModuleGetSurfRef":(CUresult, POINTER(CUsurfref), CUmodule, c_char_p),
			"cuModuleGetTexRef":(CUresult, POINTER(CUtexref), CUmodule, c_char_p),
			"cuModuleLoad":(CUresult, POINTER(CUmodule), c_char_p),
			"cuModuleLoadData":(CUresult, POINTER(CUmodule), c_void_p),
			"cuModuleLoadDataEx":(CUresult, POINTER(CUmodule), c_void_p, c_uint, POINTER(CUjit_option), POINTER(c_void_p)),
			"cuModuleLoadFatBinary":(CUresult, POINTER(CUmodule), c_void_p),
			"cuModuleUnload":(CUresult, CUmodule),
		#MEMORY MANAGEMENT
			"cuArray3DCreate_v2":(CUresult, POINTER(CUarray), POINTER(CUDA_ARRAY3D_DESCRIPTOR)),
			"cuArray3DGetDescriptor_v2":(CUresult, POINTER(CUDA_ARRAY3D_DESCRIPTOR), CUarray),
			"cuArrayCreate_v2":(CUresult, POINTER(CUarray), POINTER(CUDA_ARRAY_DESCRIPTOR)),
			"cuArrayDestroy":(CUresult, CUarray),
			"cuArrayGetDescriptor_v2":(CUresult, POINTER(CUDA_ARRAY_DESCRIPTOR), CUarray),
			"cuDeviceGetByPCIBusId":(CUresult, POINTER(CUdevice), c_char_p),
			"cuDeviceGetPCIBusId":(CUresult, c_char_p, c_int, CUdevice),
			"cuIpcCloseMemHandle":(CUresult, CUdeviceptr),
			"cuIpcGetEventHandle":(CUresult, POINTER(CUipcEventHandle), CUevent),
			"cuIpcGetMemHandle":(CUresult, POINTER(CUipcMemHandle), CUdeviceptr),
			"cuIpcOpenEventHandle":(CUresult, POINTER(CUevent), CUipcEventHandle),
			"cuIpcOpenMemHandle":(CUresult, POINTER(CUdeviceptr), CUipcMemHandle, c_uint),
			"cuMemAlloc_v2":(CUresult, POINTER(CUdeviceptr), c_size_t),
			"cuMemAllocHost_v2":(CUresult, POINTER(c_void_p), c_size_t),
			"cuMemAllocManaged":(CUresult, POINTER(CUdeviceptr), c_size_t, c_uint),
			"cuMemAllocPitch_v2":(CUresult, POINTER(CUdeviceptr), POINTER(c_size_t), c_size_t, c_size_t, c_uint),
			"cuMemFree_v2":(CUresult, CUdeviceptr),
			"cuMemFreeHost":(CUresult, c_void_p),
			"cuMemGetAddressRange_v2":(CUresult, POINTER(CUdeviceptr), POINTER(c_size_t), CUdeviceptr),
			"cuMemGetInfo_v2":(CUresult, POINTER(c_size_t), POINTER(c_size_t)),
			"cuMemHostAlloc":(CUresult, POINTER(c_void_p), c_size_t, c_uint),
			"cuMemHostGetDevicePointer_v2":(CUresult, POINTER(CUdeviceptr), c_void_p, c_uint),
			"cuMemHostGetFlags":(CUresult, POINTER(c_uint), c_void_p),
			"cuMemHostRegister_v2":(CUresult, c_void_p, c_size_t, c_uint),
			"cuMemHostUnregister":(CUresult, c_void_p),
			"cuMemcpy":(CUresult, CUdeviceptr, CUdeviceptr, c_size_t),
			"cuMemcpy2D_v2":(CUresult, POINTER(CUDA_MEMCPY2D)),
			"cuMemcpy2DAsync_v2":(CUresult, POINTER(CUDA_MEMCPY2D), CUstream),
			"cuMemcpy2DUnaligned_v2":(CUresult, POINTER(CUDA_MEMCPY2D)),
			"cuMemcpy3D_v2":(CUresult, POINTER(CUDA_MEMCPY3D)),
			"cuMemcpy3DAsync_v2":(CUresult, POINTER(CUDA_MEMCPY3D), CUstream),
			"cuMemcpy3DPeer":(CUresult, POINTER(CUDA_MEMCPY3D_PEER)),
			"cuMemcpy3DPeerAsync":(CUresult, POINTER(CUDA_MEMCPY3D_PEER), CUstream),
			"cuMemcpyAsync":(CUresult, CUdeviceptr, CUdeviceptr, c_size_t, CUstream),
			"cuMemcpyAtoA_v2":(CUresult, CUarray, c_size_t, CUarray, c_size_t, c_size_t),
			"cuMemcpyAtoD_v2":(CUresult, CUdeviceptr, CUarray, c_size_t, c_size_t),
			"cuMemcpyAtoH_v2":(CUresult, c_void_p, CUarray, c_size_t, c_size_t),
			"cuMemcpyAtoHAsync_v2":(CUresult, c_void_p, CUarray, c_size_t, c_size_t, CUstream),
			"cuMemcpyDtoA_v2":(CUresult, CUarray, c_size_t, CUdeviceptr, c_size_t),
			"cuMemcpyDtoD_v2":(CUresult, CUdeviceptr, CUdeviceptr, c_size_t),
			"cuMemcpyDtoDAsync_v2":(CUresult, CUdeviceptr, CUdeviceptr, c_size_t, CUstream),
			"cuMemcpyDtoH_v2":(CUresult, c_void_p, CUdeviceptr, c_size_t),
			"cuMemcpyDtoHAsync_v2":(CUresult, c_void_p, CUdeviceptr, c_size_t, CUstream),
			"cuMemcpyHtoA_v2":(CUresult, CUarray, c_size_t, c_void_p, c_size_t),
			"cuMemcpyHtoAAsync_v2":(CUresult, CUarray, c_size_t, c_void_p, c_size_t, CUstream),
			"cuMemcpyHtoD_v2":(CUresult, CUdeviceptr, c_void_p, c_size_t),
			"cuMemcpyHtoDAsync_v2":(CUresult, CUdeviceptr, c_void_p, c_size_t, CUstream),
			"cuMemcpyPeer":(CUresult, CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, c_size_t),
			"cuMemcpyPeerAsync":(CUresult, CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, c_size_t, CUstream),
			"cuMemsetD16_v2":(CUresult, CUdeviceptr, c_uint16, c_size_t),
			"cuMemsetD16Async":(CUresult, CUdeviceptr, c_uint16, c_size_t, CUstream),
			# "cuMemsetD2D16":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t),
			# "cuMemsetD2D16Async":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t, CUstream),
			# "cuMemsetD2D32":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t),
			# "cuMemsetD2D32Async":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t, CUstream),
			# "cuMemsetD2D8":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t),
			# "cuMemsetD2D8Async":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t, CUstream),
			"cuMemsetD32_v2":(CUresult, CUdeviceptr, c_uint32, c_size_t),
			"cuMemsetD32Async":(CUresult, CUdeviceptr, c_uint32, c_size_t, CUstream),
			"cuMemsetD8_v2":(CUresult, CUdeviceptr, c_uint8, c_size_t),
			"cuMemsetD8Async":(CUresult, CUdeviceptr, c_uint8, c_size_t, CUstream),
			"cuMipmappedArrayCreate":(CUresult, POINTER(CUmipmappedArray), POINTER(CUDA_ARRAY3D_DESCRIPTOR), c_uint),
			"cuMipmappedArrayDestroy":(CUresult, CUmipmappedArray),
			"cuMipmappedArrayGetLevel":(CUresult, POINTER(CUarray), CUmipmappedArray, c_uint),
		#UNIFIED ADDRESSING
			"cuMemAdvise": (CUresult, CUdeviceptr, c_size_t, CUmem_advise, CUdevice),
			"cuMemPrefetchAsync": (CUresult, CUdeviceptr, c_size_t, CUdevice, CUstream),
			"cuMemRangeGetAttribute": (CUresult, c_void_p, c_size_t, CUmem_range_attribute, ),
			"cuMemRangeGetAttributes": (CUresult, POINTER(c_void_p), POINTER(c_size_t), POINTER(CUmem_range_attribute), c_size_t, CUdeviceptr, c_size_t),
			"cuPointerGetAttribute": (CUresult, c_void_p, CUpointer_attribute, CUdeviceptr),
			"cuPointerGetAttributes": (CUresult, c_uint, POINTER(CUpointer_attribute), POINTER(c_void_p), CUdeviceptr),
			"cuPointerSetAttribute": (CUresult, c_void_p, CUpointer_attribute, CUdeviceptr),
		#STREAM MANAGEMENT
			"cuStreamAddCallback": (CUresult, CUstream, CUstreamCallback, c_void_p, c_uint),
			"cuStreamAttachMemAsync": (CUresult, CUstream, CUdeviceptr, c_size_t, c_uint),
			"cuStreamCreate": (CUresult, POINTER(CUstream), c_uint),
			"cuStreamCreateWithPriority": (CUresult, POINTER(CUstream), c_uint, c_int),
			"cuStreamDestroy_v2": (CUresult, CUstream),
			"cuStreamGetFlags": (CUresult, CUstream, POINTER(c_uint)),
			"cuStreamGetPriority": (CUresult, CUstream, POINTER(c_int)),
			"cuStreamQuery": (CUresult, CUstream),
			"cuStreamSynchronize": (CUresult, CUstream),
			"cuStreamWaitEvent": (CUresult, CUstream, CUevent, c_uint),
		#EVENT MANAGEMENT
			"cuEventCreate": (CUresult, POINTER(CUevent), c_uint),
			"cuEventDestroy_v2": (CUresult, CUevent),
			"cuEventElapsedTime": (CUresult, POINTER(c_float), CUevent, CUevent),
			"cuEventQuery": (CUresult, CUevent),
			"cuEventRecord": (CUresult, CUevent, CUstream),
			"cuEventSynchronize": (CUresult, CUevent),
			"cuStreamBatchMemOp": (CUresult, CUstream, c_uint, POINTER(CUstreamBatchMemOpParams), c_uint),
			"cuStreamWaitValue32": (CUresult, CUstream, CUdeviceptr, cuuint32_t, c_uint),
			"cuStreamWriteValue32": (CUresult, CUstream, CUdeviceptr, cuuint64_t, c_uint),
		#EXECUTION CONTROL
			"cuFuncGetAttribute": (CUresult, POINTER(c_int), CUfunction_attribute, CUfunction),
			"cuFuncSetCacheConfig": (CUresult, CUfunction, CUfunc_cache),
			"cuFuncSetSharedMemConfig": (CUresult, CUfunction, CUsharedconfig),
			"cuLaunchKernel": (CUresult, CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, POINTER(c_void_p), POINTER(c_void_p)),
			############
			#DEPRECATED#
			############
			"cuFuncSetBlockShape": (CUresult, CUfunction, c_int, c_int, c_int),
			"cuFuncSetSharedSize": (CUresult, CUfunction, c_uint),
			"cuLaunch": (CUresult, CUfunction),
			"cuLaunchGrid": (CUresult, CUfunction, c_int, c_int),
			"cuLaunchGridAsync": (CUresult, CUfunction, c_int, c_int, CUstream),
			"cuParamSetSize": (CUresult, CUfunction, c_uint),
			"cuParamSetTexRef": (CUresult, CUfunction, c_int, CUtexref),
			"cuParamSetf": (CUresult, CUfunction, c_int, c_float),
			"cuParamSeti": (CUresult, CUfunction, c_int, c_uint),
			"cuParamSetv": (CUresult, CUfunction, c_int, c_void_p, c_uint),
		#OCCUPANCY
			# # CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize )
			# "": (CUresult, ),
			# # CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags )
			# "": (CUresult, ),
			# # CUresult cuOccupancyMaxPotentialBlockSize ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit )
			# "": (CUresult, ),
			# # CUresult cuOccupancyMaxPotentialBlockSizeWithFlags ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit, unsigned int  flags )
			# "": (CUresult, ),
		#TEXTURE REFERENCE MANAGEMENT
			# CUresult cuTexRefGetAddress ( CUdeviceptr* pdptr, CUtexref hTexRef )
			# Gets the address associated with a texture reference.
			# CUresult cuTexRefGetAddressMode ( CUaddress_mode* pam, CUtexref hTexRef, int  dim )
			# Gets the addressing mode used by a texture reference.
			# CUresult cuTexRefGetArray ( CUarray* phArray, CUtexref hTexRef )
			# Gets the array bound to a texture reference.
			# CUresult cuTexRefGetBorderColor ( float* pBorderColor, CUtexref hTexRef )
			# Gets the border color used by a texture reference.
			# CUresult cuTexRefGetFilterMode ( CUfilter_mode* pfm, CUtexref hTexRef )
			# Gets the filter-mode used by a texture reference.
			# CUresult cuTexRefGetFlags ( unsigned int* pFlags, CUtexref hTexRef )
			# Gets the flags used by a texture reference.
			# CUresult cuTexRefGetFormat ( CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef )
			# Gets the format used by a texture reference.
			# CUresult cuTexRefGetMaxAnisotropy ( int* pmaxAniso, CUtexref hTexRef )
			# Gets the maximum anisotropy for a texture reference.
			# CUresult cuTexRefGetMipmapFilterMode ( CUfilter_mode* pfm, CUtexref hTexRef )
			# Gets the mipmap filtering mode for a texture reference.
			# CUresult cuTexRefGetMipmapLevelBias ( float* pbias, CUtexref hTexRef )
			# Gets the mipmap level bias for a texture reference.
			# CUresult cuTexRefGetMipmapLevelClamp ( float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef )
			# Gets the min/max mipmap level clamps for a texture reference.
			# CUresult cuTexRefGetMipmappedArray ( CUmipmappedArray* phMipmappedArray, CUtexref hTexRef )
			# Gets the mipmapped array bound to a texture reference.
			# CUresult cuTexRefSetAddress ( size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes )
			# Binds an address as a texture reference.
			# CUresult cuTexRefSetAddress2D ( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch )
			# Binds an address as a 2D texture reference.
			# CUresult cuTexRefSetAddressMode ( CUtexref hTexRef, int  dim, CUaddress_mode am )
			# Sets the addressing mode for a texture reference.
			# CUresult cuTexRefSetArray ( CUtexref hTexRef, CUarray hArray, unsigned int  Flags )
			# Binds an array as a texture reference.
			# CUresult cuTexRefSetBorderColor ( CUtexref hTexRef, float* pBorderColor )
			# Sets the border color for a texture reference.
			# CUresult cuTexRefSetFilterMode ( CUtexref hTexRef, CUfilter_mode fm )
			# Sets the filtering mode for a texture reference.
			# CUresult cuTexRefSetFlags ( CUtexref hTexRef, unsigned int  Flags )
			# Sets the flags for a texture reference.
			# CUresult cuTexRefSetFormat ( CUtexref hTexRef, CUarray_format fmt, int  NumPackedComponents )
			# Sets the format for a texture reference.
			# CUresult cuTexRefSetMaxAnisotropy ( CUtexref hTexRef, unsigned int  maxAniso )
			# Sets the maximum anisotropy for a texture reference.
			# CUresult cuTexRefSetMipmapFilterMode ( CUtexref hTexRef, CUfilter_mode fm )
			# Sets the mipmap filtering mode for a texture reference.
			# CUresult cuTexRefSetMipmapLevelBias ( CUtexref hTexRef, float  bias )
			# Sets the mipmap level bias for a texture reference.
			# CUresult cuTexRefSetMipmapLevelClamp ( CUtexref hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp )
			# Sets the mipmap min/max mipmap level clamps for a texture reference.
			# CUresult cuTexRefSetMipmappedArray ( CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int  Flags )
			# Binds a mipmapped array to a texture reference.
			############
			#DEPRECATED#
			############
			# CUresult cuTexRefCreate ( CUtexref* pTexRef )
			# Creates a texture reference.
			# CUresult cuTexRefDestroy ( CUtexref hTexRef )
			# Destroys a texture reference.
		#SURFACE REFERENCE MANAGEMENT
			# CUresult cuSurfRefGetArray ( CUarray* phArray, CUsurfref hSurfRef )
			# Passes back the CUDA array bound to a surface reference.
			# CUresult cuSurfRefSetArray ( CUsurfref hSurfRef, CUarray hArray, unsigned int  Flags )
			# Sets the CUDA array for a surface reference.
		#TEXTURE OBJECT MANAGEMENT
			# # CUresult cuTexObjectCreate ( CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc )
			# "": (CUresult, ),
			# # CUresult cuTexObjectDestroy ( CUtexObject texObject )
			# "": (CUresult, ),
			# # CUresult cuTexObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject )
			# "": (CUresult, ),
			# # CUresult cuTexObjectGetResourceViewDesc ( CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject )
			# "": (CUresult, ),
			# # CUresult cuTexObjectGetTextureDesc ( CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject )
			# "": (CUresult, ),
		#SURFACE OBJECT MANAGEMENT
			# # CUresult cuSurfObjectCreate ( CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc )
			# "": (CUresult, ),
			# # CUresult cuSurfObjectDestroy ( CUsurfObject surfObject )
			# "": (CUresult, ),
			# # CUresult cuSurfObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject )
			# "": (CUresult, ),
		#PEER CONTEXT MEMORY ACCESS
			# # CUresult cuCtxDisablePeerAccess ( CUcontext peerContext )
			# "": (CUresult, ),
			# # CUresult cuCtxEnablePeerAccess ( CUcontext peerContext, unsigned int  Flags )
			# "": (CUresult, ),
			# # CUresult cuDeviceCanAccessPeer ( int* canAccessPeer, CUdevice dev, CUdevice peerDev )
			# "": (CUresult, ),
			# # CUresult cuDeviceGetP2PAttribute ( int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice )
			# "": (CUresult, ),
		#GRAPHICS INTEROPERABILITY
			"cuGraphicsMapResources": (CUresult, c_uint, POINTER(CUgraphicsResource), CUstream),
			"cuGraphicsResourceGetMappedMipmappedArray": (CUresult, POINTER(CUmipmappedArray), CUgraphicsResource),
			"cuGraphicsResourceGetMappedPointer_v2": (CUresult, POINTER(CUdeviceptr), POINTER(c_size_t), CUgraphicsResource),
			"cuGraphicsResourceSetMapFlags_v2": (CUresult, CUgraphicsResource, c_uint),
			"cuGraphicsSubResourceGetMappedArray": (CUresult, POINTER(CUarray), CUgraphicsResource, c_uint, c_uint),
			"cuGraphicsUnmapResources": (CUresult, c_uint, POINTER(CUgraphicsResource), CUstream),
			"cuGraphicsUnregisterResource": (CUresult, CUgraphicsResource),
		#PROFILER CONTROL
			# CUresult cuProfilerInitialize ( const char* configFile, const char* outputFile, CUoutput_mode outputMode )
			# Initialize the profiling.
			# CUresult cuProfilerStart ( void )
			# Enable profiling.
			# CUresult cuProfilerStop ( void )
			# Disable profiling.
		#OPENGL INTEROPERABILITY
			"cuGLGetDevices_v2": (CUresult, POINTER(c_uint), POINTER(CUdevice), c_uint, CUGLDeviceList),
			"cuGraphicsGLRegisterBuffer": (CUresult, POINTER(CUgraphicsResource), GLuint, c_uint),
			"cuGraphicsGLRegisterImage": (CUresult, POINTER(CUgraphicsResource), GLuint, GLenum, c_uint),
			# # CUresult cuWGLGetDevice ( CUdevice* pDevice, HGPUNV hGpu )
			# "cuWGLGetDevice": (CUresult, ),
			############
			#DEPRECATED#
			############
			# CUresult cuGLCtxCreate ( CUcontext* pCtx, unsigned int  Flags, CUdevice device )
			# Create a CUDA context for interoperability with OpenGL.
			# CUresult cuGLInit ( void )
			# Initializes OpenGL interoperability.
			# CUresult cuGLMapBufferObject ( CUdeviceptr* dptr, size_t* size, GLuint buffer )
			# Maps an OpenGL buffer object.
			# CUresult cuGLMapBufferObjectAsync ( CUdeviceptr* dptr, size_t* size, GLuint buffer, CUstream hStream )
			# Maps an OpenGL buffer object.
			# CUresult cuGLRegisterBufferObject ( GLuint buffer )
			# Registers an OpenGL buffer object.
			# CUresult cuGLSetBufferObjectMapFlags ( GLuint buffer, unsigned int  Flags )
			# Set the map flags for an OpenGL buffer object.
			# CUresult cuGLUnmapBufferObject ( GLuint buffer )
			# Unmaps an OpenGL buffer object.
			# CUresult cuGLUnmapBufferObjectAsync ( GLuint buffer, CUstream hStream )
			# Unmaps an OpenGL buffer object.
			# CUresult cuGLUnregisterBufferObject ( GLuint buffer )
			# Unregister an OpenGL buffer object.
		#VDPAU Interoperability
			# CUresult cuGraphicsVDPAURegisterOutputSurface ( CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int  flags )
			# Registers a VDPAU VdpOutputSurface object.
			# CUresult cuGraphicsVDPAURegisterVideoSurface ( CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int  flags )
			# Registers a VDPAU VdpVideoSurface object.
			# CUresult cuVDPAUCtxCreate ( CUcontext* pCtx, unsigned int  flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress )
			# Create a CUDA context for interoperability with VDPAU.
			# CUresult cuVDPAUGetDevice ( CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress )
			# Gets the CUDA device associated with a VDPAU device.
		#EGL Interoperability
			# CUresult cuEGLStreamConsumerAcquireFrame ( CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int  timeout )
			# Acquire an image frame from the EGLStream with CUDA as a consumer.
			# CUresult cuEGLStreamConsumerConnect ( CUeglStreamConnection* conn, EGLStreamKHR stream )
			# Connect CUDA to EGLStream as a consumer.
			# CUresult cuEGLStreamConsumerConnectWithFlags ( CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int  flags )
			# Connect CUDA to EGLStream as a consumer with given flags.
			# CUresult cuEGLStreamConsumerDisconnect ( CUeglStreamConnection* conn )
			# Disconnect CUDA as a consumer to EGLStream .
			# CUresult cuEGLStreamConsumerReleaseFrame ( CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream )
			# Releases the last frame acquired from the EGLStream.
			# CUresult cuEGLStreamProducerConnect ( CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height )
			# Connect CUDA to EGLStream as a producer.
			# CUresult cuEGLStreamProducerDisconnect ( CUeglStreamConnection* conn )
			# Disconnect CUDA as a producer to EGLStream .
			# CUresult cuEGLStreamProducerPresentFrame ( CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream )
			# Present a CUDA eglFrame to the EGLStream with CUDA as a producer.
			# CUresult cuEGLStreamProducerReturnFrame ( CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream )
			# Return the CUDA eglFrame to the EGLStream released by the consumer.
			# CUresult cuGraphicsEGLRegisterImage ( CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int  flags )
			# Registers an EGL image.
			# CUresult cuGraphicsResourceGetMappedEglFrame ( CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int  index, unsigned int  mipLevel )
			# Get an eglFrame through which to access a registered EGL graphics resource.
	}

	alias = {
		#DEVICE MANAGEMENT
			 "cuDeviceTotalMem":"cuDeviceTotalMem_v2",
		#CONTEXT MANAGEMENT
			 "cuCtxCreate":"cuCtxCreate_v2",
			 "cuCtxDestroy":"cuCtxDestroy_v2",
			 "cuCtxPopCurrent":"cuCtxPopCurrent_v2",
			 "cuCtxPushCurrent":"cuCtxPushCurrent_v2",
		#MODULE MANAGEMENT
			 "cuLinkAddData":"cuLinkAddData_v2",
			 "cuLinkAddFile":"cuLinkAddFile_v2",
			 "cuLinkCreate":"cuLinkCreate_v2",
			 "cuModuleGetGlobal":"cuModuleGetGlobal_v2",
		#MEMORY MANAGEMENT
			 "cuArray3DCreate":"cuArray3DCreate_v2",
			 "cuArray3DGetDescriptor":"cuArray3DGetDescriptor_v2",
			 "cuArrayCreate":"cuArrayCreate_v2",
			 "cuArrayGetDescriptor":"cuArrayGetDescriptor_v2",
			 "cuMemGetInfo":"cuMemGetInfo_v2",
			 "cuMemAlloc":"cuMemAlloc_v2",
			 "cuMemAllocHost":"cuMemAllocHost_v2",
			 "cuMemAllocPitch":"cuMemAllocPitch_v2",
			 "cuMemFree":"cuMemFree_v2",
			 "cuMemGetAddressRange":"cuMemGetAddressRange_v2",
			 "cuMemHostGetDevicePointer":"cuMemHostGetDevicePointer_v2",
			 "cuMemHostRegister":"cuMemHostRegister_v2",
			 "cuMemcpy2D":"cuMemcpy2D_v2",
			 "cuMemcpy2DAsync":"cuMemcpy2DAsync_v2",
			 "cuMemcpy2DUnaligned":"cuMemcpy2DUnaligned_v2",
			 "cuMemcpy3D":"cuMemcpy3D_v2",
			 "cuMemcpy3DAsync":"cuMemcpy3DAsync_v2",
			 "cuMemcpyAtoA":"cuMemcpyAtoA_v2",
			 "cuMemcpyAtoD":"cuMemcpyAtoD_v2",
			 "cuMemcpyAtoH":"cuMemcpyAtoH_v2",
			 "cuMemcpyAtoHAsync":"cuMemcpyAtoHAsync_v2",
			 "cuMemcpyDtoA":"cuMemcpyDtoA_v2",
			 "cuMemcpyDtoD":"cuMemcpyDtoD_v2",
			 "cuMemcpyDtoDAsync":"cuMemcpyDtoDAsync_v2",
			 "cuMemcpyDtoH":"cuMemcpyDtoH_v2",
			 "cuMemcpyDtoHAsync":"cuMemcpyDtoHAsync_v2",
			 "cuMemcpyHtoA":"cuMemcpyHtoA_v2",
			 "cuMemcpyHtoAAsync":"cuMemcpyHtoAAsync_v2",
			 "cuMemcpyHtoD":"cuMemcpyHtoD_v2",
			 "cuMemcpyHtoDAsync":"cuMemcpyHtoDAsync_v2",
			 "cuMemsetD16":"cuMemsetD16_v2",
			 "cuMemsetD32":"cuMemsetD32_v2",
			 "cuMemsetD8":"cuMemsetD8_v2",
			 # "cuMemsetD2D8":"cuMemsetD2D8_v2",
			 # "cuMemsetD2D16":"cuMemsetD2D16_v2",
			 # "cuMemsetD2D32":"cuMemsetD2D32_v2",
		#STREAM MANAGEMENT
			 "cuStreamDestroy":"cuStreamDestroy_v2",
		#EVENT MANAGEMENT
			 "cuEventDestroy":"cuEventDestroy_v2",
		#TEXTURE REFERENCE MANAGEMENT
			 # "cuTexRefSetAddress":"cuTexRefSetAddress_v2",
			 # "cuTexRefSetAddress2D":"cuTexRefSetAddress2D_v2",
			 # "cuTexRefGetAddress":"cuTexRefGetAddress_v2",
		#GRAPHICS INTEROPERABILITY
			 "cuGraphicsResourceGetMappedPointer":"cuGraphicsResourceGetMappedPointer_v2",
			 "cuGraphicsResourceSetMapFlags":"cuGraphicsResourceSetMapFlags_v2",
		#OPENGL INTEROPERABILITY
			 "cuGLGetDevices":"cuGLGetDevices_v2",
		#VDPAU Interoperability
			 # "cuVDPAUCtxCreate":"cuVDPAUCtxCreate_v2",
			}

	return api, alias

def API_PROTOTYPES_09000():
	api = {
		#EVENT MANAGEMENT
			"cuStreamWaitValue64":(CUresult, CUstream, CUdeviceptr, cuuint64_t, c_uint),
			"cuStreamWriteValue64": (CUresult, CUstream, CUdeviceptr, cuuint64_t, c_uint),
		#EXECUTION CONTROL
			"cuFuncSetAttribute": (CUresult, CUfunction, CUfunction_attribute, c_int),
			"cuLaunchCooperativeKernel": (CUresult, CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, POINTER(c_void_p)),
			"cuLaunchCooperativeKernelMultiDevice": (CUresult, POINTER(CUDA_LAUNCH_PARAMS), c_uint, c_uint)
	}

	alias = {}

	api_08000, alias_08000 = API_PROTOTYPES_08000()

	api.update(api_08000)
	alias.update(alias_08000)

	return api, alias

def API_PROTOTYPES_09010():
	api = {}

	alias = {}

	api_09000, alias_09000 = API_PROTOTYPES_09000()

	api.update(api_09000)
	alias.update(alias_09000)

	return api, alias

def API_PROTOTYPES_09020():
	api = {
		#DEVICE MANAGEMENT
			"cuDeviceGetUuid": (CUresult, POINTER(CUuuid), CUdevice),
		#STREAM MANAGEMENT	
			"cuStreamGetCtx": (CUresult, CUstream, POINTER(CUcontext)),
		#Following were moved from EVENT MANAGMENT to STREAM MEMORY OPERATIONS
			# cuStreamBatchMemOp
			# cuStreamWaitValue32
			# cuStreamWaitValue64
			# cuStreamWriteValue32
			# cuStreamWriteValue64
	}
	
	alias = {}

	api_09010, alias_09010 = API_PROTOTYPES_09010()

	api.update(api_09010)
	alias.update(alias_09010)

	return api, alias

def API_PROTOTYPES_10000():
	api = {
		#DEVICE MANAGEMENT
			"cuDeviceGetLuid": (CUresult, c_char_p, POINTER(c_uint), CUdevice),
		#STREAM MANAGEMENT
			"cuStreamBeginCapture": (CUresult, CUstream, CUstreamCaptureMode),
			"cuStreamEndCapture": (CUresult, CUstream, POINTER(CUgraph)),
			"cuStreamIsCapturing": (CUresult, CUstream, POINTER(CUstreamCaptureStatus)),
		#EXTERNAL RESOURCE INTEROPERABILITY
			"cuDestroyExternalMemory": (CUresult, CUexternalMemory),
			"cuDestroyExternalSemaphore": (CUresult, CUexternalSemaphore),
			"cuExternalMemoryGetMappedBuffer": (CUresult, POINTER(CUdeviceptr), CUexternalMemory, POINTER(CUDA_EXTERNAL_MEMORY_BUFFER_DESC)),
			"cuExternalMemoryGetMappedMipmappedArray": (CUresult, POINTER(CUmipmappedArray), CUexternalMemory, POINTER(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC)),
			"cuImportExternalMemory": (CUresult, POINTER(CUexternalMemory), POINTER(CUDA_EXTERNAL_MEMORY_HANDLE_DESC)),
			"cuImportExternalSemaphore": (CUresult, POINTER(CUexternalSemaphore), POINTER(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC)),
			"cuSignalExternalSemaphoresAsync": (CUresult, POINTER(CUexternalSemaphore), POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), c_uint, CUstream),
			"cuWaitExternalSemaphoresAsync": (CUresult, POINTER(CUexternalSemaphore), POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS), c_uint, CUstream),
		#EXECUTION CONTROL
			"cuLaunchHostFunc": (CUresult, CUstream, CUhostFn, py_object),
		#GRAPH MANAGEMENT
			# cuGraphAddChildGraphNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph )
			# cuGraphAddDependencies ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t numDependencies )
			# cuGraphAddEmptyNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies )
			# cuGraphAddHostNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams )
			# cuGraphAddKernelNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams )
			# cuGraphAddMemcpyNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx )
			# cuGraphAddMemsetNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx )
			# cuGraphChildGraphNodeGetGraph ( CUgraphNode hNode, CUgraph* phGraph )
			# cuGraphClone ( CUgraph* phGraphClone, CUgraph originalGraph )
			# cuGraphCreate ( CUgraph* phGraph, unsigned int  flags )
			# cuGraphDestroy ( CUgraph hGraph )
			# cuGraphDestroyNode ( CUgraphNode hNode )
			# cuGraphExecDestroy ( CUgraphExec hGraphExec )
			# cuGraphGetEdges ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges )
			# cuGraphGetNodes ( CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes )
			# cuGraphGetRootNodes ( CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes )
			# cuGraphHostNodeGetParams ( CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams )
			# cuGraphHostNodeSetParams ( CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams )
			# cuGraphInstantiate ( CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize )
			# cuGraphKernelNodeGetParams ( CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams )
			# cuGraphKernelNodeSetParams ( CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams )
			# cuGraphLaunch ( CUgraphExec hGraphExec, CUstream hStream )
			# cuGraphMemcpyNodeGetParams ( CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams )
			# cuGraphMemcpyNodeSetParams ( CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams )
			# cuGraphMemsetNodeGetParams ( CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams )
			# cuGraphMemsetNodeSetParams ( CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams )
			# cuGraphNodeFindInClone ( CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph )
			# cuGraphNodeGetDependencies ( CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies )
			# cuGraphNodeGetDependentNodes ( CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes )
			# cuGraphNodeGetType ( CUgraphNode hNode, CUgraphNodeType* type )
			# cuGraphRemoveDependencies ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t numDependencies )
	}

	alias = {}

	api_09020, alias_09020 = API_PROTOTYPES_09020()

	api.update(api_09020)
	alias.update(alias_09020)

	return api, alias

def API_PROTOTYPES_10010():
	api = {
		#STREAM MANAGEMENT
			"cuStreamBeginCapture_v2":(CUresult, CUstream, CUstreamCaptureMode),
			"cuStreamGetCaptureInfo": (CUresult, CUstream, POINTER(CUstreamCaptureStatus), POINTER(cuuint64_t)),
			"cuThreadExchangeStreamCaptureMode": (CUresult, POINTER(CUstreamCaptureMode)),
		#GRAPH MANAGEMENT
			#cuGraphExecKernelNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams )
		#TEXTURE REFERENCE MANAGEMENT is now DEPRECATED
		#SURFACE REFERENCE MANAGEMENT is now DEPRECATED
	}

	alias = {"cuStreamBeginCapture":"cuStreamBeginCapture_v2"}

	api_10000, alias_10000 = API_PROTOTYPES_10000()

	api.update(api_10000)
	alias.update(alias_10000)

	return api, alias

def API_PROTOTYPES_10020():
	api = {
		#DEVICE MANAGEMENT
			"cuDeviceGetNvSciSyncAttributes": (c_void_p, CUdevice, c_int),
		#VIRTUAL MEMORY MANAGEMENT
			# cuMemAddressFree ( CUdeviceptr ptr, size_t size )
			# cuMemAddressReserve ( CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags )
			# cuMemCreate ( CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags )
			# cuMemExportToShareableHandle ( void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags )
			# cuMemGetAccess ( unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr )
			# cuMemGetAllocationGranularity ( size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option )
			# cuMemGetAllocationPropertiesFromHandle ( CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle )
			# cuMemImportFromShareableHandle ( CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType )
			# cuMemMap ( CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags )
			# cuMemRelease ( CUmemGenericAllocationHandle handle )
			# cuMemSetAccess ( CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count )
			# cuMemUnmap ( CUdeviceptr ptr, size_t size )
		#GRAPH MANAGEMENT
			# cuGraphExecHostNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams )
			# cuGraphExecMemcpyNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx )
			# cuGraphExecMemsetNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx )
	}

	alias = {}

	api_10010, alias_10010 = API_PROTOTYPES_10010()

	api.update(api_10010)
	alias.update(alias_10010)

	return api, alias

def API_PROTOTYPES_11000():
	api = {
		#CONTEXT MANAGEMENT
			"cuCtxResetPersistingL2Cache":(CUresult,),
		#PRIMARY CONTEXT MANAGEMENT
			"cuDevicePrimaryCtxRelease_v2":(CUresult, CUdevice),
			"cuDevicePrimaryCtxSetFlags_v2":(CUresult, CUdevice, c_uint),
			"cuDevicePrimaryCtxReset_v2":(CUresult, CUdevice),
		#MEMORY MANAGEMENT
			"cuIpcOpenMemHandle_v2":(CUresult, POINTER(CUdeviceptr), CUipcMemHandle, c_uint),
		#VIRTUAL MEMORY MANAGEMENT
			# cuMemRetainAllocationHandle ( CUmemGenericAllocationHandle* handle, void* addr )
		#STREAM MANAGEMENT
			"cuStreamCopyAttributes": (CUresult, CUstream, CUstream),
			"cuStreamSetAttribute": (CUresult, CUstream, CUstreamAttrID, POINTER(CUstreamAttrID)),
			"cuStreamGetAttribute": (CUresult, CUstream, CUstreamAttrID, POINTER(CUstreamAttrValue)),
		#GRAPH MANAGEMENT
			# cuGraphKernelNodeCopyAttributes ( CUgraphNode dst, CUgraphNode src )
			# cuGraphKernelNodeGetAttribute ( CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out )
			# cuGraphKernelNodeSetAttribute ( CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value )
			# cuGraphInstantiate_v2 ( CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize )
				# "cuGraphInstantiate_v2":
		#OCCUPANCY
			# cuOccupancyAvailableDynamicSMemPerBlock ( size_t* dynamicSmemSize, CUfunction func, int  numBlocks, int  blockSize )

		#PROFILER CONTROL
			#cuProfilerInitialize is now DEPRECATED
	}

	alias = {
			"cuDevicePrimaryCtxRelease_v2":"cuDevicePrimaryCtxRelease",
			"cuDevicePrimaryCtxSetFlags_v2":"cuDevicePrimaryCtxSetFlags",
			"cuDevicePrimaryCtxReset_v2":"cuDevicePrimaryCtxReset",
			"cuIpcOpenMemHandle_v2":"cuIpcOpenMemHandle",
			# "cuGraphInstantiate":"cuGraphInstantiate_v2"
	}

	api_10020, alias_10020 = API_PROTOTYPES_10020()

	api.update(api_10020)
	alias.update(alias_10020)

	return api, alias

def API_PROTOTYPES_11010():
	api = {
		#DEVICE MANAGEMENT
			# cuDeviceGetTexture1DLinearMaxWidth ( size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev )

		# Memory Managment
			# cuArrayGetSparseProperties ( CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array )
			# cuMipmappedArrayGetSparseProperties ( CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap )

		# Virtual Memory Management (introduced in 10.2)
			# cuMemMapArrayAsync ( CUarrayMapInfo* mapInfoList, unsigned int  count, CUstream hStream )


		# Event Management
			# cuEventRecordWithFlags ( CUevent hEvent, CUstream hStream, unsigned int  flags )

		# Graph Management
			# cuGraphAddEventRecordNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event )
			# cuGraphAddEventWaitNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event )
			# cuGraphEventRecordNodeGetEvent ( CUgraphNode hNode, CUevent* event_out )
			# cuGraphEventRecordNodeSetEvent ( CUgraphNode hNode, CUevent event )
			# cuGraphEventWaitNodeGetEvent ( CUgraphNode hNode, CUevent* event_out )
			# cuGraphEventWaitNodeSetEvent ( CUgraphNode hNode, CUevent event )
			# cuGraphExecChildGraphNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph )
			# cuGraphExecEventRecordNodeSetEvent ( CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event )
			# cuGraphExecEventWaitNodeSetEvent ( CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event )
			# cuGraphUpload ( CUgraphExec hGraphExec, CUstream hStream )

		# Direct3D X Interoperability
			# 9/10/11 removed
	}

	alias = {}

	api_11000, alias_11000 = API_PROTOTYPES_11000()

	api.update(api_11000)
	alias.update(alias_11000)

	return api, alias

def API_PROTOTYPES_11011():
	api = {}

	alias = {}

	api_11010, alias_11010 = API_PROTOTYPES_11010()

	api.update(api_11010)
	alias.update(alias_11010)

	return api, alias

def API_PROTOTYPES_11020():
	api = {
		#DEVICE MANAGEMENT
			"cuDeviceGetDefaultMemPool": (CUresult, POINTER(CUmemoryPool), CUdevice),
			"cuDeviceGetMemPool":(CUresult, POINTER(CUmemoryPool), CUdevice),
			# cuDeviceSetMemPool ( CUdevice dev, CUmemoryPool pool )
		# Memory Managment
			# cuArrayGetPlane ( CUarray* pPlaneArray, CUarray hArray, unsigned int  planeIdx )
		# Stream Ordered Memory Allocator (introduced in 11.2)
			# cuMemAllocAsync ( CUdeviceptr* dptr, size_t bytesize, CUstream hStream )
			# Allocates memory with stream ordered semantics.
			# cuMemAllocFromPoolAsync ( CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream )
			# Allocates memory from a specified pool with stream ordered semantics.
			# cuMemFreeAsync ( CUdeviceptr dptr, CUstream hStream )
			# Frees memory with stream ordered semantics.
			# cuMemPoolCreate ( CUmemoryPool* pool, const CUmemPoolProps* poolProps )
			# Creates a memory pool.
			# cuMemPoolDestroy ( CUmemoryPool pool )
			# Destroys the specified memory pool.
			# cuMemPoolExportPointer ( CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr )
			# Export data to share a memory pool allocation between processes.
			# cuMemPoolExportToShareableHandle ( void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags )
			# Exports a memory pool to the requested handle type.
			# cuMemPoolGetAccess ( CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location )
			# Returns the accessibility of a pool from a device.
			# cuMemPoolGetAttribute ( CUmemoryPool pool, CUmemPool_attribute attr, void* value )
			# Gets attributes of a memory pool.
			# cuMemPoolImportFromShareableHandle ( CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags )
			# imports a memory pool from a shared handle.
			# cuMemPoolImportPointer ( CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData )
			# Import a memory pool allocation from another process.
			# cuMemPoolSetAccess ( CUmemoryPool pool, const CUmemAccessDesc* map, size_t count )
			# Controls visibility of pools between devices.
			# cuMemPoolSetAttribute ( CUmemoryPool pool, CUmemPool_attribute attr, void* value )
			# Sets attributes of a memory pool.
			# cuMemPoolTrimTo ( CUmemoryPool pool, size_t minBytesToKeep )
			# Tries to release memory back to the OS.
		# Graph Management
			# cuGraphAddExternalSemaphoresSignalNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams )
			# Creates an external semaphore signal node and adds it to a graph.
			# cuGraphAddExternalSemaphoresWaitNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams )
			# Creates an external semaphore wait node and adds it to a graph.
			# cuGraphExecExternalSemaphoresSignalNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams )
			# Sets the parameters for an external semaphore signal node in the given graphExec.
			# cuGraphExecExternalSemaphoresWaitNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams )
			# Sets the parameters for an external semaphore wait node in the given graphExec.
			# cuGraphExternalSemaphoresSignalNodeGetParams ( CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out )
			# Returns an external semaphore signal node's parameters.
			# cuGraphExternalSemaphoresSignalNodeSetParams ( CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams )
			# Sets an external semaphore signal node's parameters.
			# cuGraphExternalSemaphoresWaitNodeGetParams ( CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out )
			# Returns an external semaphore wait node's parameters.
			# cuGraphExternalSemaphoresWaitNodeSetParams ( CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams )
			# Sets an external semaphore wait node's parameters.
	}

	alias = {}

	api_11011, alias_11011 = API_PROTOTYPES_11011()

	api.update(api_11011)
	alias.update(alias_11011)

	return api, alias

# https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/index.html
# https://docs.nvidia.com/cuda/archive/11.2.1/cuda-driver-api/index.html
# https://docs.nvidia.com/cuda/archive/11.2.2/cuda-driver-api/index.html
# https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/index.html


#TO DO
	#implement prototypes beyond here


def API_PROTOTYPES_11021():
	api = {}

	alias = {}

	api_11020, alias_11020 = API_PROTOTYPES_11020()

	api.update(api_11020)
	alias.update(alias_11020)

	return api, alias

def API_PROTOTYPES_11022():
	api = {}

	alias = {}

	api_11021, alias_11021 = API_PROTOTYPES_11021()

	api.update(api_11021)
	alias.update(alias_11021)

	return api, alias

def API_PROTOTYPES_11030():
	api = {
		# "cuStreamGetCaptureInfo_v2":(),
		# 	#note: does not alias cuStreamGetCaptureInfo
	}

	alias = {}

	api_11022, alias_11022 = API_PROTOTYPES_11022()

	api.update(api_11022)
	alias.update(alias_11022)

	return api, alias

def API_PROTOTYPES_11031():
	api = {}

	alias = {}

	api_11030, alias_11030 = API_PROTOTYPES_11030()

	api.update(api_11030)
	alias.update(alias_11030)

	return api, alias

def API_PROTOTYPES_11040():
	api = {
		# "cuDeviceGetUuid_v2":(),
		# 	#note: does not alias cuDeviceGetUuid
		# "cuCtxCreate_v3":(),
		# 	#note: does not alias cuCtxCreate
	}

	alias = {}

	api_11031, alias_11031 = API_PROTOTYPES_11031()

	api.update(api_11031)
	alias.update(alias_11031)

	return api, alias

def API_PROTOTYPES_11041():
	api = {}

	alias = {}

	api_11040, alias_11040 = API_PROTOTYPES_11040()

	api.update(api_11040)
	alias.update(alias_11040)

	return api, alias

def API_PROTOTYPES_11042():
	api = {}

	alias = {}

	api_11041, alias_11041 = API_PROTOTYPES_11041()

	api.update(api_11041)
	alias.update(alias_11041)

	return api, alias

def API_PROTOTYPES_11043():
	api = {}

	alias = {}

	api_11042, alias_11042 = API_PROTOTYPES_11042()

	api.update(api_11042)
	alias.update(alias_11042)

	return api, alias

def API_PROTOTYPES_11044():
	api = {}

	alias = {}

	api_11043, alias_11043 = API_PROTOTYPES_11043()

	api.update(api_11043)
	alias.update(alias_11043)

	return api, alias

def API_PROTOTYPES_11050():
	api = {}

	alias = {}

	api_11044, alias_11044 = API_PROTOTYPES_11044()

	api.update(api_11044)
	alias.update(alias_11044)

	return api, alias

def API_PROTOTYPES_11051():
	api = {}

	alias = {}

	api_11050, alias_11050 = API_PROTOTYPES_11050()

	api.update(api_11050)
	alias.update(alias_11050)

	return api, alias

def API_PROTOTYPES_11052():
	api = {}

	alias = {}

	api_11051, alias_11051 = API_PROTOTYPES_11051()

	api.update(api_11051)
	alias.update(alias_11051)

	return api, alias

def API_PROTOTYPES_11060():
	api = {}

	alias = {}

	api_11052, alias_11052 = API_PROTOTYPES_11052()

	api.update(api_11052)
	alias.update(alias_11052)

	return api, alias

def API_PROTOTYPES_11061():
	api = {}

	alias = {}

	api_11060, alias_11060 = API_PROTOTYPES_11060()

	api.update(api_11060)
	alias.update(alias_11060)

	return api, alias

#note: used to determine which API_PROTOTYPE to load. Calling cuDriverGetVersion does not require initializing the driver
VERSION_PROTOTYPE = ("cuDriverGetVersion", (CUresult, POINTER(c_int)))

API_PROTOTYPES = { 8000:API_PROTOTYPES_08000,
				   9000:API_PROTOTYPES_09000,
				   9010:API_PROTOTYPES_09010,
				   9020:API_PROTOTYPES_09020,
				  10000:API_PROTOTYPES_10000,
				  10010:API_PROTOTYPES_10010,
				  10020:API_PROTOTYPES_10020,
				  11010:API_PROTOTYPES_11010,
				  11011:API_PROTOTYPES_11011,
				  11020:API_PROTOTYPES_11020,
				  11021:API_PROTOTYPES_11021,
				  11022:API_PROTOTYPES_11022,
				  11030:API_PROTOTYPES_11030,
				  11031:API_PROTOTYPES_11031,
				  11040:API_PROTOTYPES_11040,
				  11041:API_PROTOTYPES_11041,
				  11042:API_PROTOTYPES_11042,
				  11043:API_PROTOTYPES_11043,
				  11044:API_PROTOTYPES_11044,
				  11050:API_PROTOTYPES_11050,
				  11051:API_PROTOTYPES_11051,
				  11052:API_PROTOTYPES_11052,
				  11060:API_PROTOTYPES_11060,
				  11061:API_PROTOTYPES_11061}
