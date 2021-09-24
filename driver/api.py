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
			"cuDeviceTotalMem": (CUresult, POINTER(c_size_t), CUdevice),
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
			"cuCtxCreate":(CUresult, POINTER(CUcontext), c_uint, CUdevice),
			"cuCtxDestroy":(CUresult, CUcontext),
			"cuCtxGetApiVersion":(CUresult, CUcontext, POINTER(c_uint)),
			"cuCtxGetCacheConfig":(CUresult, POINTER(CUfunc_cache)),
			"cuCtxGetCurrent":(CUresult, POINTER(CUcontext)),
			"cuCtxGetDevice":(CUresult, POINTER(CUdevice)),
			"cuCtxGetFlags":(CUresult, POINTER(c_uint)),
			"cuCtxGetLimit":(CUresult, POINTER(c_size_t), CUlimit),
			"cuCtxGetSharedMemConfig":(CUresult, POINTER(CUsharedconfig)),
			"cuCtxGetStreamPriorityRange":(CUresult, POINTER(c_int), POINTER(c_int)),
			"cuCtxPopCurrent":(CUresult, POINTER(CUcontext)),
			"cuCtxPushCurrent":(CUresult, POINTER(CUcontext)),
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
			"cuLinkAddData":(CUresult, CUlinkState, CUjitInputType, c_void_p, c_size_t, c_char_p, c_uint, POINTER(CUjit_option), POINTER(c_void_p)),
			"cuLinkAddFile":(CUresult, CUlinkState, CUjitInputType, c_char_p, c_uint, POINTER(CUjit_option), POINTER(c_void_p)),
			"cuLinkComplete":(CUresult, CUlinkState, POINTER(c_void_p), POINTER(c_size_t)),
			"cuLinkCreate":(CUresult, c_uint, POINTER(CUjit_option), POINTER(c_void_p), POINTER(CUlinkState)),
			"cuLinkDestroy":(CUresult, CUlinkState),
			"cuModuleGetFunction":(CUresult, POINTER(CUfunction), CUmodule, c_char_p),
			"cuModuleGetGlobal":(CUresult, POINTER(CUdeviceptr), POINTER(c_size_t), CUmodule, c_char_p),
			"cuModuleGetSurfRef":(CUresult, POINTER(CUsurfref), CUmodule, c_char_p),
			"cuModuleGetTexRef":(CUresult, POINTER(CUtexref), CUmodule, c_char_p),
			"cuModuleLoad":(CUresult, POINTER(CUmodule), c_char_p),
			"cuModuleLoadData":(CUresult, POINTER(CUmodule), c_void_p),
			"cuModuleLoadDataEx":(CUresult, POINTER(CUmodule), c_void_p, c_uint, POINTER(CUjit_option), POINTER(c_void_p)),
			"cuModuleLoadFatBinary":(CUresult, POINTER(CUmodule), c_void_p),
			"cuModuleUnload":(CUresult, CUmodule),
		#MEMORY MANAGEMENT
			"cuArray3DCreate":(CUresult, POINTER(CUarray), POINTER(CUDA_ARRAY3D_DESCRIPTOR)),
			"cuArray3DGetDescriptor":(CUresult, POINTER(CUDA_ARRAY3D_DESCRIPTOR), CUarray),
			"cuArrayCreate":(CUresult, POINTER(CUarray), POINTER(CUDA_ARRAY_DESCRIPTOR)),
			"cuArrayDestroy":(CUresult, CUarray),
			"cuArrayGetDescriptor":(CUresult, POINTER(CUDA_ARRAY_DESCRIPTOR), CUarray),
			"cuDeviceGetByPCIBusId":(CUresult, POINTER(CUdevice), c_char_p),
			"cuDeviceGetPCIBusId":(CUresult, c_char_p, c_int, CUdevice),
			"cuIpcCloseMemHandle":(CUresult, CUdeviceptr),
			"cuIpcGetEventHandle":(CUresult, POINTER(CUipcEventHandle), CUevent),
			"cuIpcGetMemHandle":(CUresult, POINTER(CUipcMemHandle), CUdeviceptr),
			"cuIpcOpenEventHandle":(CUresult, POINTER(CUevent), CUipcEventHandle),
			"cuIpcOpenMemHandle":(CUresult, POINTER(CUdeviceptr), CUipcMemHandle, c_uint),
			"cuMemAlloc":(CUresult, POINTER(CUdeviceptr), c_size_t),
			"cuMemAllocHost":(CUresult, POINTER(c_void_p), c_size_t),
			"cuMemAllocManaged":(CUresult, POINTER(CUdeviceptr), c_size_t, c_uint),
			"cuMemAllocPitch":(CUresult, POINTER(CUdeviceptr), POINTER(c_size_t), c_size_t, c_size_t, c_uint),
			"cuMemFree":(CUresult, CUdeviceptr),
			"cuMemFreeHost":(CUresult, c_void_p),
			"cuMemGetAddressRange":(CUresult, POINTER(CUdeviceptr), POINTER(c_size_t), CUdeviceptr),
			"cuMemGetInfo":(CUresult, POINTER(c_size_t), POINTER(c_size_t)),
			"cuMemHostAlloc":(CUresult, POINTER(c_void_p), c_size_t, c_uint),
			"cuMemHostGetDevicePointer":(CUresult, POINTER(CUdeviceptr), c_void_p, c_uint),
			"cuMemHostGetFlags":(CUresult, POINTER(c_uint), c_void_p),
			"cuMemHostRegister":(CUresult, c_void_p, c_size_t, c_uint),
			"cuMemHostUnregister":(CUresult, c_void_p),
			"cuMemcpy":(CUresult, CUdeviceptr, CUdeviceptr, c_size_t),
			"cuMemcpy2D":(CUresult, POINTER(CUDA_MEMCPY2D)),
			"cuMemcpy2DAsync":(CUresult, POINTER(CUDA_MEMCPY2D), CUstream),
			"cuMemcpy2DUnaligned":(CUresult, POINTER(CUDA_MEMCPY2D)),
			"cuMemcpy3D":(CUresult, POINTER(CUDA_MEMCPY3D)),
			"cuMemcpy3DAsync":(CUresult, POINTER(CUDA_MEMCPY3D), CUstream),
			"cuMemcpy3DPeer":(CUresult, POINTER(CUDA_MEMCPY3D_PEER)),
			"cuMemcpy3DPeerAsync":(CUresult, POINTER(CUDA_MEMCPY3D_PEER), CUstream),
			"cuMemcpyAsync":(CUresult, CUdeviceptr, CUdeviceptr, c_size_t, CUstream),
			"cuMemcpyAtoA":(CUresult, CUarray, c_size_t, CUarray, c_size_t, c_size_t),
			"cuMemcpyAtoD":(CUresult, CUdeviceptr, CUarray, c_size_t, c_size_t),
			"cuMemcpyAtoH":(CUresult, c_void_p, CUarray, c_size_t, c_size_t),
			"cuMemcpyAtoHAsync":(CUresult, c_void_p, CUarray, c_size_t, c_size_t, CUstream),
			"cuMemcpyDtoA":(CUresult, CUarray, c_size_t, CUdeviceptr, c_size_t),
			"cuMemcpyDtoD":(CUresult, CUdeviceptr, CUdeviceptr, c_size_t),
			"cuMemcpyDtoDAsync":(CUresult, CUdeviceptr, CUdeviceptr, c_size_t, CUstream),
			"cuMemcpyDtoH":(CUresult, c_void_p, CUdeviceptr, c_size_t),
			"cuMemcpyDtoHAsync":(CUresult, c_void_p, CUdeviceptr, c_size_t, CUstream),
			"cuMemcpyHtoA":(CUresult, CUarray, c_size_t, c_void_p, c_size_t),
			"cuMemcpyHtoAAsync":(CUresult, CUarray, c_size_t, c_void_p, c_size_t, CUstream),
			"cuMemcpyHtoD":(CUresult, CUdeviceptr, c_void_p, c_size_t),
			"cuMemcpyHtoDAsync":(CUresult, CUdeviceptr, c_void_p, c_size_t, CUstream),
			"cuMemcpyPeer":(CUresult, CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, c_size_t),
			"cuMemcpyPeerAsync":(CUresult, CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, c_size_t, CUstream),
			"cuMemsetD16":(CUresult, CUdeviceptr, c_uint, c_size_t),
			"cuMemsetD16Async":(CUresult, CUdeviceptr, c_uint, c_size_t, CUstream),
			"cuMemsetD2D16":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t),
			"cuMemsetD2D16Async":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t, CUstream),
			"cuMemsetD2D32":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t),
			"cuMemsetD2D32Async":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t, CUstream),
			"cuMemsetD2D8":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t),
			"cuMemsetD2D8Async":(CUresult, CUdeviceptr, c_size_t, c_uint, c_size_t, c_size_t, CUstream),
			"cuMemsetD32":(CUresult, CUdeviceptr, c_uint, c_size_t),
			"cuMemsetD32Async":(CUresult, CUdeviceptr, c_uint, c_size_t, CUstream),
			"cuMemsetD8":(CUresult, CUdeviceptr, c_uint, c_size_t),
			"cuMemsetD8Async":(CUresult, CUdeviceptr, c_uint, c_size_t, CUstream),
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
			"cuStreamDestroy": (CUresult, CUstream),
			"cuStreamGetFlags": (CUresult, CUstream, POINTER(c_uint)),
			"cuStreamGetPriority": (CUresult, CUstream, POINTER(c_int)),
			"cuStreamQuery": (CUresult, CUstream),
			"cuStreamSynchronize": (CUresult, CUstream),
			"cuStreamWaitEvent": (CUresult, CUstream, CUevent, c_uint),
		#EVENT MANAGEMENT
			"cuEventCreate": (CUresult, POINTER(CUevent), c_uint),
			"cuEventDestroy": (CUresult, CUevent),
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
			"cuGraphicsResourceGetMappedPointer": (CUresult, POINTER(CUdeviceptr), POINTER(c_size_t), CUgraphicsResource),
			"cuGraphicsResourceSetMapFlags": (CUresult, CUgraphicsResource, c_uint),
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
			"cuGLGetDevices": (CUresult, POINTER(c_uint), POINTER(CUdevice), c_uint, CUGLDeviceList),
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

	return api

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

	api.update(API_PROTOTYPES_08000())

	return api

def API_PROTOTYPES_09010():
	api = {}

	api.update(API_PROTOTYPES_09000())

	return api

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
	
	api.update(API_PROTOTYPES_09010())

	return api

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

	api.update(API_PROTOTYPES_09020())

	return api

def API_PROTOTYPES_10010():
	api = {
		#STREAM MANAGEMENT
			"cuStreamGetCaptureInfo": (CUresult, CUstream, POINTER(CUstreamCaptureStatus), POINTER(cuuint64_t)),
			"cuThreadExchangeStreamCaptureMode": (CUresult, POINTER(CUstreamCaptureMode)),
		#GRAPH MANAGEMENT
			#cuGraphExecKernelNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams )
		#TEXTURE REFERENCE MANAGEMENT is now DEPRECATED
		#SURFACE REFERENCE MANAGEMENT is now DEPRECATED
	}

	api.update(API_PROTOTYPES_10000())

	return api

def API_PROTOTYPES_10020():
	api = {
		#DEVICE MANAGEMENT
			"cuDeviceGetNvSciSyncAttributes": (c_void_p, CUdevice, c_int),
		# Virtual Memory Management
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
		# Graph Management
			# cuGraphExecHostNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams )
			# cuGraphExecMemcpyNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx )
			# cuGraphExecMemsetNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx )
	}

	api.update(API_PROTOTYPES_10010())

	return api

def API_PROTOTYPES_11000():
	api = {
		#CONTEXT MANAGEMENT
			"cuCtxResetPersistingL2Cache":(CUresult,),
		#Virtual Memory Management
			# cuMemRetainAllocationHandle ( CUmemGenericAllocationHandle* handle, void* addr )
		#STREAM MANAGEMENT
			"cuStreamCopyAttributes": (CUresult, CUstream, CUstream),
			"cuStreamSetAttribute": (CUresult, CUstream, CUstreamAttrID, POINTER(CUstreamAttrID)),
			"cuStreamGetAttribute": (CUresult, CUstream, CUstreamAttrID, POINTER(CUstreamAttrValue)),
		# Graph Management
			# cuGraphKernelNodeCopyAttributes ( CUgraphNode dst, CUgraphNode src )
			# cuGraphKernelNodeGetAttribute ( CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out )
			# cuGraphKernelNodeSetAttribute ( CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value )
		# Occupancy
			# cuOccupancyAvailableDynamicSMemPerBlock ( size_t* dynamicSmemSize, CUfunction func, int  numBlocks, int  blockSize )

		# Profiler Control
			#cuProfilerInitialize is now DEPRECATED
	}

	api.update(API_PROTOTYPES_10020())

	return api

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

	api.update(API_PROTOTYPES_11000())

	return api

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

	api.update(API_PROTOTYPES_11010())

	return api


# https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/index.html
# https://docs.nvidia.com/cuda/archive/11.2.1/cuda-driver-api/index.html
# https://docs.nvidia.com/cuda/archive/11.2.2/cuda-driver-api/index.html
# https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/index.html


def API_PROTOTYPES_11021():
	#TO DO
		#implement prototypes

	api = {}

	api.update(API_PROTOTYPES_11020())

	return api

def API_PROTOTYPES_11022():
	#TO DO
		#implement prototypes

	api = {}

	api.update(API_PROTOTYPES_11021())

	return api

def API_PROTOTYPES_11030():
	#TO DO
		#implement prototypes

	api = {}

	api.update(API_PROTOTYPES_11022())

	return api

def API_PROTOTYPES_11040():
	#TO DO
		#implement prototypes

	api = {}

	api.update(API_PROTOTYPES_11030())

	return api

#note: used to determine which API_PROTOTYPE to load. Calling cuDriverGetVersion does not require initializing the driver
VERSION_PROTOTYPE = ("cuDriverGetVersion", (CUresult, POINTER(c_int)))

API_PROTOTYPES = { 8000:API_PROTOTYPES_08000,
				   9000:API_PROTOTYPES_09000,
				   9010:API_PROTOTYPES_09010,
				   9020:API_PROTOTYPES_09020,
				  10000:API_PROTOTYPES_10000,
				  10010:API_PROTOTYPES_10010,
				  10020:API_PROTOTYPES_10020,
				  11000:API_PROTOTYPES_11000,
				  11010:API_PROTOTYPES_11010,
				  11020:API_PROTOTYPES_11020,
				  11021:API_PROTOTYPES_11021,
				  11022:API_PROTOTYPES_11022,
				  11030:API_PROTOTYPES_11030,
				  11040:API_PROTOTYPES_11040}
