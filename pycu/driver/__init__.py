from .driver import Driver
from .structs import *
from .enums import *
from .defines import *
from .typedefs import *
from .error import check_driver_error

from ctypes import byref, c_char_p, string_at

driver = Driver()

# # #add driver functions to module namespace (allows driver functions to be called directly. example: cuInit() instead of cuInit())
# def __getattr__(name):
# 	if name in API_PROTOTYPES:
# 		return getattr(driver, name)
# # 	# elif name == '__all__':
# # 	# 	return list(globals().keys())
# # 	# 	# return list(globals().values())
# # 	raise ImportError(f'cannot import name "{name}" from {__name__} ({__file__})')

#note: add driver functions to global namespace of module so you can call functions as cuInit() instead of driver.cuInit().
for name, cfunc in vars(driver).items():
	globals()[name] = cfunc

def check_driver_error_verbose(err, msg):
	check_driver_error(err, "%s: %s" % (msg, get_error_string(err)))

def check_driver_error_either(err, msg):
	if True: #pycu.driver.error.VERBOSE_DRIVER_ERRORS
		check_driver_error_verbose(err, msg)
	else:
		check_driver_error(err, msg)

#ERROR HANDLING
def get_error_name(error):
	name = c_char_p()
	err = cuGetErrorName(error, byref(name))
	check_driver_error(err, "cuGetErrorName error")

	return name.value.decode('utf8')

def get_error_string(error):
	string = c_char_p()
	err = cuGetErrorString(error, byref(string))
	check_driver_error(err, "cuGetErrorString error")

	return string.value.decode('utf8')

#INITILIZATION
def init(flags = None):
	err = cuInit(0) #flags
	check_driver_error(err, "cuInit error")

#VERSION MANAGEMENT
def driver_get_version():
	version = c_int()
	err = cuDriverGetVersion(byref(version))
	check_driver_error_either(err, "cuDriverGetVersion error")

	return version.value

#DEVICE MANAGEMENT
def device_get(ordinal):
	device = CUdevice()
	err = cuDeviceGet(byref(device), ordinal)
	check_driver_error_either(err, "cuDeviceGet error")

	return device.value

def device_get_attribute(attrib, dev):
	pi = c_int()
	err = cuDeviceGetAttribute(byref(pi), attrib, dev)
	check_driver_error_either(err, "cuDeviceGetAttribute error")

	return pi.value

def device_get_count():
	count = c_int()
	err = cuDeviceGetCount(byref(count))
	check_driver_error_either(err, "cuDeviceGetCount error")

	return count.value

# def device_get_luid(dev):
	# luid = c_char()
	# nodemask = c_uint()
	# err = cuDeviceGetLuid(byref(luid), byref(nodemask), dev)
	# check_driver_error(err, "cuDeviceGetLuid error")

	# return luid, nodemask

def device_get_name(dev, length = 20):
	name = c_char_p(addressof((c_char * length)()))
	err = cuDeviceGetName(name, length, dev)
	check_driver_error_either(err, "cuDeviceGetName error")

	return name.value.decode('utf8')

def device_get_nvscisync_attributes(dev, flags):
	attributes = c_void_p()
	err = cuDeviceGetNvSciSyncAttributes(attributes, dev, flags)
	check_driver_error_either(err, "cuDeviceGetNvSciSyncAttributes error")

	return attributes

def device_get_uuid(dev):
	uuid = CUuuid()
	err = cuDeviceGetUuid(byref(uuid), dev)
	check_driver_error_either(err, "cuDeviceGetUuid error")

	return uuid

def device_total_mem(dev):
	byte = c_size_t()
	err = cuDeviceTotalMem(byref(byte), dev)
	check_driver_error_either(err, "cuDeviceTotalMem error")

	return byte.value

def device_compute_capability(dev):
	major = c_int()
	minor = c_int()
	err = cuDeviceComputeCapability(byref(major), byref(minor), dev)
	check_driver_error_either(err, "cuDeviceComputeCapability error")

	return major.value, minor.value

def device_get_properties(dev):
	prop = CUdevprop()
	err = cuDeviceGetProperties(byref(prop), dev)
	check_driver_error_either(err, "cuDeviceGetProperties error")

	return prop

#PRIMARY CONTEXT MANAGEMENT
def device_primary_ctx_get_state(dev):
	flags = c_uint()
	active = c_int()
	err = cuDevicePrimaryCtxGetState(dev, byref(flags), byref(active))
	check_driver_error_either(err, "PrimaryCtxGetState error")

	return flags.value, active.value

def device_primary_ctx_release(dev):
	err = cuDevicePrimaryCtxRelease(dev)
	check_driver_error_either(err, "cuDevicePrimaryCtxRelease error")

def device_primary_ctx_reset(dev):
	err = cuDevicePrimaryCtxReset()
	check_driver_error_either(err, "cuDevicePrimaryCtxReset error")

def device_primary_ctx_retain(dev):
	ctx = CUcontext()
	err = cuDevicePrimaryCtxRetain(byref(ctx), dev)
	check_driver_error_either(err, "cuDevicePrimaryCtxRetain error")

	return ctx

def device_primary_ctx_set_flags_(dev, flags):
	err = cuDevicePrimaryCtxSetFlags(dev, flags)
	check_driver_error_either(err, "cuDevicePrimaryCtxSetFlags error")

#CONTEXT MANAGEMENT
def ctx_create(dev, flags = 0):
	ctx = CUcontext()
	flags = c_uint(flags)
	err = cuCtxCreate(byref(ctx), flags, dev)
	check_driver_error_either(err, "cuCtxCreate error")

	return ctx

def ctx_destroy(ctx):
	err = cuCtxDestroy(ctx)
	check_driver_error_either(err, "cuCtxDestroy error")

def ctx_get_api_version(ctx):
	version = c_uint()
	err = cuCtxGetApiVersion(ctx, byref(version))
	check_driver_error_either(err, "cuCtxGetApiVersion error")

	return version.value

def ctx_get_cache_config():
	pconfig = CUfunc_cache()
	err = cuCtxGetCacheConfig(byref(pconfig))
	check_driver_error_either(err, "cuCtxGetCacheConfig error")

	return pconfig

def ctx_get_current():
	ctx = CUcontext()
	err = cuCtxGetCurrent(byref(ctx))
	check_driver_error_either(err, "cuCtxGetCurrent error")

	return ctx

def ctx_get_device():
	device = CUdevice()
	err = cuCtxGetDevice(byref(device))
	check_driver_error_either(err, "cuCtxGetDevice error")

	return device.value

def ctx_get_flags():
	flags = c_uint()
	err = cuCtxGetFlags(byref(flags))
	check_driver_error_either(err, "cuCtxGetFlags error")

	return flags.value

def ctx_get_limit(limit):
	pvalue = c_size_t()
	err = cuCtxGetLimit(byref(pvalue), limit)
	check_driver_error_either(err, "cuCtxGetLimit error")

	return pvalue.value

def ctx_get_shared_mem_config():
	pConfig = CUsharedconfig()
	err = cuCtxGetSharedMemConfig(byref(pConfig))
	check_driver_error_either(err, "cuCtxGetSharedMemConfig error")

	return pConfig.value

def ctx_get_stream_priority_range():
	least    = c_int()
	greatest = c_int()
	err = cuCtxGetStreamPriorityRange(byref(least), byref(greatest))
	check_driver_error_either(err, "cuCtxGetStreamPriorityRange error")

	return least.value, greatest.value

def ctx_pop_current():
	ctx = CUcontext()
	err = cuCtxPopCurrent(byref(ctx))
	check_driver_error_either(err, "cuCtxPopCurrent error")

	return ctx

def ctx_push_current(ctx):
	err = cuCtxPushCurrent(ctx)
	check_driver_error_either(err, "cuCtxPushCurrent error")

def ctx_reset_persisting_l2_cache():
	err = cuCtxResetPersistingL2Cache()
	check_driver_error_either(err, "cuCtxResetPersistingL2Cache error")

def ctx_set_cache_config(config):
	err = cuCtxSetCacheConfig(config)
	check_driver_error_either(err, "cuCtxSetCacheConfig error")

def ctx_set_current(ctx):
	err = cuCtxSetCurrent(ctx)
	check_driver_error_either(err, "cuCtxSetCurrent error")

def ctx_set_limit(limit, value):
	err = cuCtxSetLimit(limit, value)
	check_driver_error_either(err, "cuCtxSetLimit error")

def ctx_set_shared_mem_config(config):
	err = cuCtxSetSharedMemConfig(config)
	check_driver_error_either(err, "cuCtxSetSharedMemConfig error")

def ctx_synchronize():
	err = cuCtxSynchronize()
	check_driver_error_either(err, "cuCtxSynchronize error")

# def ctx_attach(ctx, flags):
	# err = cuCtxAttach(byref(ctx), flags)
	# check_driver_error_either(err, "cuCtxAttach error")

# def ctx_detach(ctx):
	# err = cuCtxDetach(ctx)
	# check_driver_error_either(err, "cuCtxDetach error")

# def _prepare_linker_options(options):
# 	numopts = len(options)
# 	optkeys = (CUjit_option * len(options))(*list(options.keys()))
# 	optvals = (c_void_p * len(options))(*list(options.values()))

# 	return numopts, optkeys, optvals

#MODULE MANAGEMENT
def link_add_data(linker, jittype, data, size, name, optkeys = None, optvals = None):
	err = cuLinkAddData(linker, jittype, data, size, name, len(optkeys) if optkeys else 0, optkeys, optvals)
	check_driver_error_either(err, "cuLinkAddData error")

def link_add_file(linker, jittype, path, optkeys = None, optvals = None):
	err = cuLinkAddFile(linker, jittype, path, len(optkeys) if optkeys else 0, optkeys, optvals)
	check_driver_error_either(err, "cuLinkAddFile error")

def link_complete(linker):
	cubin = c_void_p()
	size = c_size_t()
	err = cuLinkComplete(linker, byref(cubin), byref(size))
	check_driver_error_either(err, "cuLinkComplete error")

	return cubin, size.value

def _get_log_buffer(optkeys, optvals):
	buff = None
	size = 0
	if optkeys and optvals:
		for n, opt in enumerate(optkeys):
			if opt == CU_JIT_ERROR_LOG_BUFFER:
				buff = optvals[n]
			elif opt == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
				size = optvals[n]
	return buff, size

def link_create(optkeys = None, optvals = None):
	#note: if log buffer is present grab it and add it to error message if an error occurs. Need to grab before cuLinkCreate is called
	buff, size = _get_log_buffer(optkeys, optvals)
	linker = CUlinkState()
	err = cuLinkCreate(len(optkeys) if optkeys else 0, optkeys, optvals, byref(linker))

	msg = "cuLinkCreate error"
	if size:
		log = string_at(c_char_p(buff), size)
		msg += f": \n{log.decode('utf8')}"
	check_driver_error_either(err, msg)

	return linker

def link_destroy(linker):
	err = cuLinkDestroy(linker)
	check_driver_error_either(err, "cuLinkDestroy error")

def module_get_function(module, entry):
	kernel = CUfunction()
	err = cuModuleGetFunction(byref(kernel), module, entry)
	check_driver_error_either(err, "cuModuleGetFunction error")

	return kernel

def module_get_global(module, name):
	ptr  = CUdeviceptr()
	byte = c_size_t()
	err = cuModuleGetGlobal(byref(ptr), byref(byte), module, name)
	check_driver_error_either(err, "cuModuleGetGlobal error")

	return ptr, byte.value

def module_get_surf_ref(module, name):
	psurf = CUsurfref()
	err = cuModuleGetSurfRef(byref(psurf), module, byref(name))
	check_driver_error_either(err, "cuModuleGetSurfRef error")

	return psurf

def module_get_tex_ref(module, name):
	ptex = CUtexref()
	err = cuModuleGetSurfRef(byref(ptex), module, byref(name))
	check_driver_error_either(err, "cuModuleGetSurfRef error")

	return ptex

def module_load(module, path):
	err = cuModuleLoad(byref(module), byref(path))
	check_driver_error_either(err, "cuModuleLoad error")

	return module

def module_load_data(image):
	module = CUmodule()
	err = cuModuleLoadData(module, image)
	check_driver_error_either(err, "cuModuleLoadData error")

	return module

def module_load_data_ex(image, optkeys = None, optvals = None):
	module = CUmodule()
	err = cuModuleLoadDataEx(byref(module), image, len(optkeys) if optkeys else 0, optkeys, optvals)
	# msg = "cuModuleLoadDataEx error"
	# if CU_JIT_ERROR_LOG_BUFFER in options:
	# 	log = string_at(c_char_p(options[CU_JIT_ERROR_LOG_BUFFER]), options[CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES].value)
	# 	msg += f": \n{log.decode('utf8')}"
	# check_driver_error_either(err, msg)

	return module

def module_load_fat_binary(fatcubin):
	module = CUmodule()
	err = cuModuleLoadFatBinary(module, fatcubin)
	check_driver_error_either(err, "cuModuleLoadFatBinary error")

	return module

def module_unload(module):
	err = cuModuleUnload(module)
	check_driver_error_either(err, "cuModuleUnload error")

#MEMORY MANAGEMENT
#CUresult cuArray3DCreate ( CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray )
#CUresult cuArray3DGetDescriptor ( CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray )
#CUresult cuArrayCreate ( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray )
#CUresult cuArrayDestroy ( CUarray hArray )
#CUresult cuArrayGetDescriptor ( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray )
#CUresult cuDeviceGetByPCIBusId ( CUdevice* dev, const char* pciBusId )
#CUresult cuDeviceGetPCIBusId ( char* pciBusId, int  len, CUdevice dev )
#CUresult cuIpcCloseMemHandle ( CUdeviceptr dptr )
#CUresult cuIpcGetEventHandle ( CUipcEventHandle* pHandle, CUevent event )
#CUresult cuIpcGetMemHandle ( CUipcMemHandle* pHandle, CUdeviceptr dptr )
#CUresult cuIpcOpenEventHandle ( CUevent* phEvent, CUipcEventHandle handle )
#CUresult cuIpcOpenMemHandle ( CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags )

def mem_alloc(bytesize):
	dptr = CUdeviceptr()
	err = cuMemAlloc(byref(dptr), c_size_t(bytesize))
	check_driver_error_either(err, "cuMemAlloc error")

	return dptr

def mem_alloc_host(bytesize):
	hptr = c_void_p()
	err = cuMemAllocHost(byref(hptr), c_size_t(bytesize))
	check_driver_error_either(err, "cuMemAllocHost error")

	return hptr

#CUresult cuMemAllocManaged ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags )
#CUresult cuMemAllocPitch ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes )

def mem_free(dptr):
	err = cuMemFree(dptr)
	check_driver_error_either(err, "cuMemFree error")

def mem_free_host(hptr):
	err = cuMemFreeHost(hptr)
	check_driver_error_either(err, "cuMemFreeHost error")

#CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )

def mem_get_info():
	free = c_size_t() 
	total = c_size_t()
	err = cuMemGetInfo(byref(free), byref(total))

	return free.value, total.value

#CUresult cuMemHostAlloc ( void** pp, size_t bytesize, unsigned int  Flags )

def mem_host_get_device_pointer(hptr, flags = 0):
	dptr = CUdeviceptr()
	err = cuMemHostGetDevicePointer(dptr, hptr, flags)
	check_driver_error_either(err, "cuMemHostGetDevicePointer error")

def mem_host_get_flags(hptr):
	flags = c_uint()
	err = cuMemHostGetFlags(byref(flags), hptr)
	check_driver_error_either(err, "cuMemHostGetFlags error")

	return flags.value

def mem_host_register(hptr, bytesize, flags = 0):
	err = cuMemHostRegister(hptr, bytesize, flags)
	check_driver_error_either(err, "cuMemHostRegister error")

def mem_host_unregister(hptr):
	err = cuMemHostUnregister(hptr)
	check_driver_error_either(err, "cuMemHostUnregister error")

#CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount )
#CUresult cuMemcpy2D ( const CUDA_MEMCPY2D* pCopy )
#CUresult cuMemcpy2DAsync ( const CUDA_MEMCPY2D* pCopy, CUstream hStream )
#CUresult cuMemcpy2DUnaligned ( const CUDA_MEMCPY2D* pCopy )
#CUresult cuMemcpy3D ( const CUDA_MEMCPY3D* pCopy )
#CUresult cuMemcpy3DAsync ( const CUDA_MEMCPY3D* pCopy, CUstream hStream )
#CUresult cuMemcpy3DPeer ( const CUDA_MEMCPY3D_PEER* pCopy )
#CUresult cuMemcpy3DPeerAsync ( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream )
#CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream )
#CUresult cuMemcpyAtoA ( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount )
#CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount )
#CUresult cuMemcpyAtoH ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount )
#CUresult cuMemcpyAtoHAsync ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream )
#CUresult cuMemcpyDtoA ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount )
#CUresult cuMemcpyDtoD ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount )
#CUresult cuMemcpyDtoDAsync ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )

def memcpy_dtoh(h_dst, d_src, nbytes):
	err = cuMemcpyDtoH(h_dst, d_src, nbytes)
	check_driver_error_either(err, "cuMemcpyDtoH error")

def memcpy_dtoh_async(h_dst, d_src, nbytes, stream):
	err = cuMemcpyDtoHAsync(h_dst, d_src, nbytes, stream)
	check_driver_error_either(err, "cuMemcpyDtoHAsync error")

#CUresult cuMemcpyHtoA ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount )
#CUresult cuMemcpyHtoAAsync ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream )

def memcpy_htod(d_dst, h_src, nbytes):
	#CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount )
	err = cuMemcpyHtoD(d_dst, h_src, nbytes)
	check_driver_error_either(err, "cuMemcpyHtoD error")

def memcpy_htod_async(d_dst, h_src, nbytes, stream):
	#CUresult cuMemcpyHtoDAsync ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream )
	err = cuMemcpyHtoDAsync(d_dst, h_src, nbytes, stream)
	check_driver_error_either(err, "cuMemcpyHtoDAsync error")

#CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount )
#CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream )

def memset_D16(d_dst, value, n):
	err = cuMemsetD16(d_dst, value, n) 
	check_driver_error_either(err, "cuMemsetD16 error")

def memset_D16_async(d_dst, value, n, stream):
	err = cuMemsetD16Async(d_dst, value, n, stream)
	check_driver_error_either(err, "cuMemsetD16Async error")

#CUresult cuMemsetD2D16 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height )
#CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream )
#CUresult cuMemsetD2D32 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height )
#CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream )
#CUresult cuMemsetD2D8 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height )
#CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream )

def memset_D32(d_dst, value, n):
	err = cuMemsetD32(d_dst, value, n) 
	check_driver_error_either(err, "cuMemsetD32 error")

def memset_D32_async(d_dst, value, n, stream):
	err = cuMemsetD32Async(d_dst, value, n, stream) 
	check_driver_error_either(err, "cuMemsetD32Async error")

def memset_D8(d_dst, value, n):
	err = cuMemsetD8(d_dst, value, n) 
	check_driver_error_either(err, "cuMemsetD8 error")

def memset_D8_async(d_dst, value, n, stream):
	err = cuMemsetD8Async(d_dst, value, n, stream) #CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream
	check_driver_error_either(err, "cuMemsetD8Async error")




#CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels )
#CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray )
#CUresult cuMipmappedArrayGetLevel ( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level )




#VIRTUAL MEMORY MANAGEMENT

#UNIFIED ADDRESSING
	# CUresult cuMemAdvise ( CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device )
	# CUresult cuMemPrefetchAsync ( CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream )
	# CUresult cuMemRangeGetAttribute ( void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count )
	# CUresult cuMemRangeGetAttributes ( void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count )
	# CUresult cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr )
	# CUresult cuPointerGetAttributes ( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr )
	# CUresult cuPointerSetAttribute ( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr )

#STREAM MANAGEMENT
def stream_add_callback(stream, callback, data, flags = 0):
	err = cuStreamAddCallback(stream, callback, data, flags)
	check_driver_error_either(err, "cuStreamAddCallback error")

def stream_attach_mem_async(stream, dptr, length, flags):
	err = cuStreamAttachMemAsync(stream, dptr, length, flags)
	check_driver_error_either(err, "cuStreamAttachMemAsync error")

def stream_begin_capture(stream, mode):
	err = cuStreamBeginCapture(stream, mode)
	check_driver_error_either(err, "cuStreamBeginCapture error")

def stream_copy_attributes(dst, src):
	err = cuStreamCopyAttributes(dst, src)
	check_driver_error_either(err, "cuStreamCopyAttributes error")

def stream_create(flags = 0):
	stream = CUstream()
	err = cuStreamCreate(byref(stream), flags)
	check_driver_error_either(err, "cuStreamCreate error")

	return stream

def stream_create_with_priority(flags, priority):
	stream = CUstream()
	err = cuStreamCreateWithPriority(stream, flags, priority)
	check_driver_error_either(err, "cuStreamCreateWithPriority error")

	return stream

def stream_destroy(stream):
	err = cuStreamDestroy(stream)
	check_driver_error_either(err, " error")

def stream_end_capture(stream, graph):
	err = cuStreamEndCapture(stream, byref(graph))
	check_driver_error_either(err, "cuStreamEndCapture error")

def stream_get_attribute(stream, attr):
	value = CUstreamAttrValue()
	err = cuStreamGetAttribute(stream, attr, byref(value))
	check_driver_error_either(err, "cuStreamGetAttribute error")

	return value.value

def stream_get_capture_info(stream):
	status = CUstreamCaptureStatus()
	id = cuuint64_t()
	err = cuStreamGetCaptureInfo(stream, byref(status), byref(id))
	check_driver_error_either(err, "cuStreamGetCaptureInfo error")

	return status.value, id.value

def stream_get_ctx(stream):
	ctx = CUcontext()
	err = cuStreamGetCtx(stream, byref(ctx))
	check_driver_error_either(err, "cuStreamGetCtx error")

	return ctx

def stream_get_flags(stream):
	flags = c_uint()
	err = cuStreamGetFlags(stream, byref(flags))
	check_driver_error_either(err, "cuStreamGetFlags error")

	return flags.value

def stream_get_priority(stream):
	priority = c_int()
	err = cuStreamGetPriority(stream, byref(priority))
	check_driver_error_either(err, "cuStreamGetPriority error")

	return priority.value

def stream_is_capturing(stream):
	status = CUstreamCaptureStatus()
	err = cuStreamIsCapturing(stream, byref(CUstreamCaptureStatus))
	check_driver_error_either(err, "cuStreamIsCapturing error")

	return status.value

def stream_query(stream):
	err = cuStreamQuery(stream)
	if err and err != CUDA_ERROR_NOT_READY:
		check_driver_error_either(err, "cuStreamQuery error")
	return err

def stream_set_attributes(stream, attr, value):
	err = cuStreamSetAttribute(stream, attr, byref(value))
	check_driver_error_either(err, "cuStreamSetAttribute error")

def stream_synchronize(stream):
	err = cuStreamSynchronize(stream)
	check_driver_error_either(err, "cuStreamSynchronize error")

def stream_wait_event(stream, event, flags = 0):
	err = cuStreamWaitEvent(stream, event, flags)
	check_driver_error_either(err, "cuStreamWaitEvent error")

def thread_exchange_stream_capture_mode(mode):
	err = cuThreadExchangeStreamCaptureMode(byref(mode))
	check_driver_error_either(err, "cuThreadExchangeStreamCaptureMode error")

#EVENT MANAGEMENT
def event_create(flags = 0):
	event = CUevent()
	err = cuEventCreate(byref(event), flags)
	check_driver_error_either(err, "cuEventCreate error")

	return event

def event_destroy(event):
	err = cuEventDestroy(event)
	check_driver_error_either(err, "cuEventDestroy error")

def event_ellapsed_time(start, end):
	time = c_float()
	err = cuEventElapsedTime(byref(time), start, end)
	check_driver_error_either(err, "cuEventElapsedTime error")

	return time.value

def event_query(event):
	err = cuEventQuery(event)
	if err and err != CUDA_ERROR_NOT_READY:
		check_driver_error_either(err, "cuStreamQuery error")
	return err

def event_record(event, stream = 0):
	err = cuEventRecord(event, stream)
	check_driver_error_either(err, "cuEventRecord error")

def event_synchronize(event):
	err = cuEventSynchronize(event)
	check_driver_error_either(err, "cuEventSynchronize error")

#EXTERNAL RESOURCE INTEROPERABILITY
def destroy_external_memory(extmem):
	err = cuDestroyExternalMemory(extmem)
	check_driver_error_either(err, "cuDestroyExternalMemory error")

def destroy_external_semaphore(extsem):
	err = cuDestroyExternalSemaphore(extsem)
	check_driver_error_either(err, "cuDestroyExternalSemaphore error")

def external_memory_get_mapped_buffer(extmem, bufferdesc):
	devptr = CUdeviceptr()
	err = cuExternalMemoryGetMappedBuffer(byref(devptr), extmem, byref(bufferdesc))
	check_driver_error_either(err, "cuExternalMemoryGetMappedBuffer error")

	return devptr

def external_memory_get_mapped_mipmapped_array(extmem, mipmapdesc):
	mipmap = CUmipmappedArray()
	err = cuExternalMemoryGetMappedMipmappedArray(byref(mipmap), extmem, byref(mipmapdesc))
	check_driver_error_either(err, "cuExternalMemoryGetMappedMipmappedArray error")

	return mipmap

def import_external_memory(memdesc):
	extmem = CUexternalMemory()
	err = cuImportExternalMemory(byref(extmem), byref(memdesc))
	check_driver_error_either(err, "cuImportExternalMemory error")

	return extmem

def import_external_semaphore(semdesc):
	extsem = CUexternalSemaphore()
	err = cuImportExternalSemaphore(byref(extsem), byref(semdesc))
	check_driver_error_either(err, "cuImportExternalSemaphore error")

	return extsem

# def signal_external_semaphore_async():
	# # CUresult cuSignalExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream )
	# err = cuSignalExternalSemaphoresAsync()
	# check_driver_error_either(err, "cuSignalExternalSemaphoresAsync error")

	# return

# def wait_external_semaphore_async():
	# # CUresult cuWaitExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream )
	# err = cuWaitExternalSemaphoresAsync
	# check_driver_error_either(err, "cuWaitExternalSemaphoresAsync error")

	# return

#STREAM MEMORY OPERATIONS
def stream_batch_mem_op(stream, count, parray, flags = 0):
	err = cuStreamBatchMemOp(stream, count, byref(parray), flags)
	check_driver_error_either(err, "cuStreamBatchMemOp error")

def stream_wait_value_32(stream, addr, value, flags = 0):
	err = cuStreamWaitValue32(stream, addr, value, flags)
	check_driver_error_either(err, "cuStreamWaitValue32 error")

def stream_wait_value_64(stream, addr, value, flags = 0):
	err = cuStreamWaitValue64(stream, addr, value, flags)
	check_driver_error_either(err, "cuStreamWaitValue64 error")

def stream_write_value_32(stream, addr, value, flags = 0):
	err = cuStreamWriteValue32(stream, addr, value, flags)
	check_driver_error_either(err, "cuStreamWriteValue32 error")

def stream_write_value_64(stream, addr, value, flags = 0):
	err = cuStreamWriteValue64(stream, addr, value, flags)
	check_driver_error_either(err, "cuStreamWriteValue64 error")

#EXECUTION CONTROL
def func_get_attribute(func, attrib):
	pi = c_int()
	err = cuFuncGetAttribute(byref(pi), attrib, func)
	check_driver_error_either(err, "cuFuncGetAttribute error")

	return pi.value

def func_set_attribute(func, attrib, value):
	err = cuFuncSetAttribute(func, attrib, value)
	check_driver_error_either(err, "cuFuncSetAttribute error")

def func_set_cache_config(func, config = 0):
	err = cuFuncSetCacheConfig(func, config)
	check_driver_error_either(err, "cuFuncSetCacheConfig error")

def func_set_shared_mem_config(func, config = 0):
	err = cuFuncSetSharedMemConfig(func, config)
	check_driver_error_either(err, "cuFuncSetSharedMemConfig error")

def launch_cooperative_kernel(f, griddim, blockdim, params, shmem = 0, stream = 0):
	griddimx, griddimy, griddimz = griddim
	blockdimx, blockdimy, blockdimz = blockdim
	params = (c_void_p * len(params))(*[addressof(c_void_p(p)) for p in params])
	# params = (c_void_p * len(params))(*[addressof(p) for p in params])

	err = cuLaunchCooperativeKernel(f, griddimx, griddimy, griddimz, blockdimx, blockdimy, blockdimz, shmem, stream, params)
	check_driver_error_either(err, "cuLaunchCooperativeKernel error")

def launch_cooperative_kernel_multi_device(launchparams, numdevs, flags):
	err = cuLaunchCooperativeKernelMultiDevice(byref(launchparams), numdevs, flags)
	check_driver_error_either(err, "cuLaunchCooperativeKernelMultiDevice error")

def launch_host_func(stream, fn, data):
	err = cuLaunchHostFunc(stream, fn, data)
	check_driver_error_either(err, "cuLaunchHostFunc error")

def launch_kernel(f, griddim, blockdim, params, shmem = 0, stream = 0, extra = None):
	griddimx, griddimy, griddimz = griddim
	blockdimx, blockdimy, blockdimz = blockdim
	params = (c_void_p * len(params))(*[addressof(p) for p in params])
	#extra

	err = cuLaunchKernel(f, griddimx, griddimy, griddimz, blockdimx, blockdimy, blockdimz, shmem, stream, params, extra)
	check_driver_error_either(err, "cuLaunchKernel error")

# def func_set_block_shape(f, blockdim):
	# x,y,z = blockdim
	# err = cuFuncSetBlockShape(f, x, y, z)
	# check_driver_error_either(err, "cuFuncSetBlockShape error")

# def func_set_shared_size(f, byte):
	# err = cuFuncSetSharedSize(f, byte)
	# check_driver_error_either(err, "cuFuncSetSharedSize error")

# def launch(f):
	# err = cuLaunch(f)
	# check_driver_error_either(err, "cuLaunch error")

# def launch_grid(f, gridsize):
	# w, h = gridsize
	# err = cuLaunchGrid(f, w, h)
	# check_driver_error_either(err, "cuLaunchGrid error")

# def launch_grid_async(f, gridsize, stream):
	# w, h = gridsize

	# err = cuLaunchGridAsync(f, w, h, stream)
	# check_driver_error_either(err, "cuLaunchGridAsync error")

# def param_set_size(f, byte):
	# err = cuParamSetSize(f, byte)
	# check_driver_error_either(err, "cuParamSetSize error")

# def param_set_tex_ref(f, texunit, texref):
	# err = cuParamSetTexRef(f, texunit, texref)
	# check_driver_error_either(err, "cuParamSetTexRef error")

# def param_set_f(f, offset, value):
	# err = cuParamSetf(f, offset, value)
	# check_driver_error_either(err, "cuParamSetf error")

# def param_set_i(f, offset, value):
	# err = cuParamSeti(f, offset, value)
	# check_driver_error_either(err, "cuParamSeti error")

# def param_set_v(f, offset, ptr, byte):
	# err = cuParamSetv(f, offset, ptr, byte)
	# check_driver_error_either(err, "cuParamSetv error")

#GRAPH MANAGEMENT

#OCCUPANCY

#TEXTURE OBJECT MANAGEMENT

#SURFACE OBJECT MANAGEMENT

#PEER CONTEXT MEMORY ACCESS

#GRAPHICS INTEROPERABILITY

def graphics_map_resources(count, resources, stream = 0):
	err = cuGraphicsMapResources(count, byref(resources), stream)
	check_driver_error_either(err, "cuGraphicsMapResources error")

# CUresult cuGraphicsResourceGetMappedMipmappedArray ( CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource )
#     Get a mipmapped array through which to access a mapped graphics resource. 

def graphics_resource_get_mapped_pointer(resource):
	devptr = CUdeviceptr()
	size = c_size_t()
	err = cuGraphicsResourceGetMappedPointer(byref(devptr), byref(size), resource)
	check_driver_error_either(err, "cuGraphicsResourceGetMappedPointer error")

	return devptr, size.value

# CUresult cuGraphicsResourceSetMapFlags ( CUgraphicsResource resource, unsigned int  flags )
#     Set usage flags for mapping a graphics resource. 
# CUresult cuGraphicsSubResourceGetMappedArray ( CUarray* pArray, CUgraphicsResource resource, unsigned int  arrayIndex, unsigned int  mipLevel )
#     Get an array through which to access a subresource of a mapped graphics resource. 

def graphics_unmap_resources(count, resources, stream = 0):
	err = cuGraphicsUnmapResources(count, byref(resources), stream)
	check_driver_error_either(err, "cuGraphicsUnmapResources error")

def graphics_unregister_resource(resource):
	err = cuGraphicsUnregisterResource(resource)
	check_driver_error_either(err, "cuGraphicsUnregisterResource error")

#PROFILER CONTROL

#OPENGL INTEROPERABILITY
def gl_get_devices():
	# CUresult cuGLGetDevices ( unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int  cudaDeviceCount, CUGLDeviceList deviceList )
	pass

def graphics_gl_register_buffer(buff, flags = 0):
	resource = CUgraphicsResource()
	err = cuGraphicsGLRegisterBuffer(byref(resource), buff, flags)
	check_driver_error_either(err, "cuGraphicsGLRegisterBuffer error")

	return resource

def graphics_gl_register_image():
	# CUresult cuGraphicsGLRegisterImage ( CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int  Flags )
	pass

# CUresult cuWGLGetDevice ( CUdevice* pDevice, HGPUNV hGpu )
#     Gets the CUDA device associated with hGpu. 
