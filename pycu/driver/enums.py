from ctypes import *


enum = c_int


CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS = enum
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS(0x0)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS(0x1)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS(0x3)


CUaccessProperty = enum
CU_ACCESS_PROPERTY_NORMAL     = CUaccessProperty(0)
CU_ACCESS_PROPERTY_STREAMING  = CUaccessProperty(1)
CU_ACCESS_PROPERTY_PERSISTING = CUaccessProperty(2)


CUaddress_mode = enum
CU_TR_ADDRESS_MODE_WRAP   = CUaddress_mode(0)
CU_TR_ADDRESS_MODE_CLAMP  = CUaddress_mode(1)
CU_TR_ADDRESS_MODE_MIRROR = CUaddress_mode(2)
CU_TR_ADDRESS_MODE_BORDER = CUaddress_mode(3)


CUarraySparseSubresourceType = enum
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = CUarraySparseSubresourceType(0)
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL      = CUarraySparseSubresourceType(1)


CUarray_cubemap_face = enum
CU_CUBEMAP_FACE_POSITIVE_X = CUarray_cubemap_face(0x00)
CU_CUBEMAP_FACE_NEGATIVE_X = CUarray_cubemap_face(0x01)
CU_CUBEMAP_FACE_POSITIVE_Y = CUarray_cubemap_face(0x02)
CU_CUBEMAP_FACE_NEGATIVE_Y = CUarray_cubemap_face(0x03)
CU_CUBEMAP_FACE_POSITIVE_Z = CUarray_cubemap_face(0x04)
CU_CUBEMAP_FACE_NEGATIVE_Z = CUarray_cubemap_face(0x05)


CUarray_format = enum
CU_AD_FORMAT_UNSIGNED_INT8  = CUarray_format(0x01)
CU_AD_FORMAT_UNSIGNED_INT16 = CUarray_format(0x02)
CU_AD_FORMAT_UNSIGNED_INT32 = CUarray_format(0x03)
CU_AD_FORMAT_SIGNED_INT8    = CUarray_format(0x08)
CU_AD_FORMAT_SIGNED_INT16   = CUarray_format(0x09)
CU_AD_FORMAT_SIGNED_INT32   = CUarray_format(0x0a)
CU_AD_FORMAT_HALF           = CUarray_format(0x10)
CU_AD_FORMAT_FLOAT          = CUarray_format(0x20)
CU_AD_FORMAT_NV12           = CUarray_format(0xb0)


CUcomputemode = enum
CU_COMPUTEMODE_DEFAULT           = CUcomputemode(0)
CU_COMPUTEMODE_PROHIBITED        = CUcomputemode(2)
CU_COMPUTEMODE_EXCLUSIVE_PROCESS = CUcomputemode(3)


CUctx_flags = enum
CU_CTX_SCHED_AUTO          = CUctx_flags(0x00)
CU_CTX_SCHED_SPIN          = CUctx_flags(0x01)
CU_CTX_SCHED_YIELD         = CUctx_flags(0x02)
CU_CTX_SCHED_BLOCKING_SYNC = CUctx_flags(0x04)
CU_CTX_BLOCKING_SYNC       = CUctx_flags(0x04) #deprecated
CU_CTX_SCHED_MASK          = CUctx_flags(0x07)
CU_CTX_MAP_HOST            = CUctx_flags(0x08) #deprecated
CU_CTX_LMEM_RESIZE_TO_MAX  = CUctx_flags(0x10)
CU_CTX_FLAGS_MASK          = CUctx_flags(0x1f)


CUdevice_P2PAttribute = enum
CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK              = CUdevice_P2PAttribute(0x01)
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED              = CUdevice_P2PAttribute(0x02)
CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED       = CUdevice_P2PAttribute(0x03)
CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED = CUdevice_P2PAttribute(0x04) #deprecated
CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED   = CUdevice_P2PAttribute(0x04)


CUdevice_attribute = enum
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                        = CUdevice_attribute(1)
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                              = CUdevice_attribute(2)
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                              = CUdevice_attribute(3)
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                              = CUdevice_attribute(4)
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                               = CUdevice_attribute(5)
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                               = CUdevice_attribute(6)
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                               = CUdevice_attribute(7)
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK                  = CUdevice_attribute(8)
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK                      = CUdevice_attribute(8)
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                        = CUdevice_attribute(9)
CU_DEVICE_ATTRIBUTE_WARP_SIZE                                    = CUdevice_attribute(10)
CU_DEVICE_ATTRIBUTE_MAX_PITCH                                    = CUdevice_attribute(11)
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK                      = CUdevice_attribute(12)
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK                          = CUdevice_attribute(12)
CU_DEVICE_ATTRIBUTE_CLOCK_RATE                                   = CUdevice_attribute(13)
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                            = CUdevice_attribute(14)
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                                  = CUdevice_attribute(15)
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                         = CUdevice_attribute(16)
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                          = CUdevice_attribute(17)
CU_DEVICE_ATTRIBUTE_INTEGRATED                                   = CUdevice_attribute(18)
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                          = CUdevice_attribute(19)
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                                 = CUdevice_attribute(20)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH                      = CUdevice_attribute(21)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH                      = CUdevice_attribute(22)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT                     = CUdevice_attribute(23)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH                      = CUdevice_attribute(24)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT                     = CUdevice_attribute(25)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH                      = CUdevice_attribute(26)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH              = CUdevice_attribute(27)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT             = CUdevice_attribute(28)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS             = CUdevice_attribute(29)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH                = CUdevice_attribute(27)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT               = CUdevice_attribute(28)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES            = CUdevice_attribute(29)
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                            = CUdevice_attribute(30)
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                           = CUdevice_attribute(31)
CU_DEVICE_ATTRIBUTE_ECC_ENABLED                                  = CUdevice_attribute(32)
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                                   = CUdevice_attribute(33)
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                                = CUdevice_attribute(34)
CU_DEVICE_ATTRIBUTE_TCC_DRIVER                                   = CUdevice_attribute(35)
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                            = CUdevice_attribute(36)
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH                      = CUdevice_attribute(37)
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                                = CUdevice_attribute(38)
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR               = CUdevice_attribute(39)
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                           = CUdevice_attribute(40)
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                           = CUdevice_attribute(41)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH              = CUdevice_attribute(42)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS             = CUdevice_attribute(43)
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER                             = CUdevice_attribute(44)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH               = CUdevice_attribute(45)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT              = CUdevice_attribute(46)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE            = CUdevice_attribute(47)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE           = CUdevice_attribute(48)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE            = CUdevice_attribute(49)
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                                = CUdevice_attribute(50)
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT                      = CUdevice_attribute(51)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH                 = CUdevice_attribute(52)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH         = CUdevice_attribute(53)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS        = CUdevice_attribute(54)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH                      = CUdevice_attribute(55)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH                      = CUdevice_attribute(56)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT                     = CUdevice_attribute(57)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH                      = CUdevice_attribute(58)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT                     = CUdevice_attribute(59)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH                      = CUdevice_attribute(60)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH              = CUdevice_attribute(61)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS             = CUdevice_attribute(62)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH              = CUdevice_attribute(63)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT             = CUdevice_attribute(64)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS             = CUdevice_attribute(65)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH                 = CUdevice_attribute(66)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH         = CUdevice_attribute(67)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS        = CUdevice_attribute(68)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH               = CUdevice_attribute(69) #deprecated
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH               = CUdevice_attribute(70)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT              = CUdevice_attribute(71)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH               = CUdevice_attribute(72)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH            = CUdevice_attribute(73)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT           = CUdevice_attribute(74)
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR                     = CUdevice_attribute(75)
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR                     = CUdevice_attribute(76)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH            = CUdevice_attribute(77)
CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED                  = CUdevice_attribute(78)
CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED                    = CUdevice_attribute(79)
CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED                     = CUdevice_attribute(80)
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR         = CUdevice_attribute(81)
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR             = CUdevice_attribute(82)
CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                               = CUdevice_attribute(83)
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                              = CUdevice_attribute(84)
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID                     = CUdevice_attribute(85)
CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED                 = CUdevice_attribute(86)
CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO        = CUdevice_attribute(87)
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS                       = CUdevice_attribute(88)
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS                    = CUdevice_attribute(89)
CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED                 = CUdevice_attribute(90)
CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM      = CUdevice_attribute(91)
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS                       = CUdevice_attribute(92)
CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS                = CUdevice_attribute(93)
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR                = CUdevice_attribute(94)
CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH                           = CUdevice_attribute(95)
CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH              = CUdevice_attribute(96)
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN            = CUdevice_attribute(97)
CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES                      = CUdevice_attribute(98)
CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED                      = CUdevice_attribute(99)
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = CUdevice_attribute(100)
CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST          = CUdevice_attribute(101)
CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED         = CUdevice_attribute(102)
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED          = CUdevice_attribute(102)
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED  = CUdevice_attribute(103)
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED           = CUdevice_attribute(104)
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED       = CUdevice_attribute(105)
CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR                = CUdevice_attribute(106)
CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED                = CUdevice_attribute(107)
CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE                 = CUdevice_attribute(108)
CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE                = CUdevice_attribute(109)
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED      = CUdevice_attribute(110)
CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK             = CUdevice_attribute(111)
CU_DEVICE_ATTRIBUTE_MAX                                          = CUdevice_attribute(112)
CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED            = CUdevice_attribute(113)
CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED         = CUdevice_attribute(114)
CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED                       = CUdevice_attribute(115)
CU_DEVICE_ATTRIBUTE_MAX                                          = CUdevice_attribute(116)


CUeglColorFormat = enum
CU_EGL_COLOR_FORMAT_YUV420_PLANAR            = CUeglColorFormat(0x00)
CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR        = CUeglColorFormat(0x01)
CU_EGL_COLOR_FORMAT_YUV422_PLANAR            = CUeglColorFormat(0x02)
CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR        = CUeglColorFormat(0x03)
CU_EGL_COLOR_FORMAT_RGB                      = CUeglColorFormat(0x04)
CU_EGL_COLOR_FORMAT_BGR                      = CUeglColorFormat(0x05)
CU_EGL_COLOR_FORMAT_ARGB                     = CUeglColorFormat(0x06)
CU_EGL_COLOR_FORMAT_RGBA                     = CUeglColorFormat(0x07)
CU_EGL_COLOR_FORMAT_L                        = CUeglColorFormat(0x08)
CU_EGL_COLOR_FORMAT_R                        = CUeglColorFormat(0x09)
CU_EGL_COLOR_FORMAT_YUV444_PLANAR            = CUeglColorFormat(0x0A)
CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR        = CUeglColorFormat(0x0B)
CU_EGL_COLOR_FORMAT_YUYV_422                 = CUeglColorFormat(0x0C)
CU_EGL_COLOR_FORMAT_UYVY_422                 = CUeglColorFormat(0x0D)
CU_EGL_COLOR_FORMAT_ABGR                     = CUeglColorFormat(0x0E)
CU_EGL_COLOR_FORMAT_BGRA                     = CUeglColorFormat(0x0F)
CU_EGL_COLOR_FORMAT_A                        = CUeglColorFormat(0x10)
CU_EGL_COLOR_FORMAT_RG                       = CUeglColorFormat(0x11)
CU_EGL_COLOR_FORMAT_AYUV                     = CUeglColorFormat(0x12)
CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR        = CUeglColorFormat(0x13)
CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR        = CUeglColorFormat(0x14)
CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR        = CUeglColorFormat(0x15)
CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR = CUeglColorFormat(0x16)
CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR = CUeglColorFormat(0x17)
CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR = CUeglColorFormat(0x18)
CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR = CUeglColorFormat(0x19)
CU_EGL_COLOR_FORMAT_VYUY_ER                  = CUeglColorFormat(0x1A)
CU_EGL_COLOR_FORMAT_UYVY_ER                  = CUeglColorFormat(0x1B)
CU_EGL_COLOR_FORMAT_YUYV_ER                  = CUeglColorFormat(0x1C)
CU_EGL_COLOR_FORMAT_YVYU_ER                  = CUeglColorFormat(0x1D)
CU_EGL_COLOR_FORMAT_YUV_ER                   = CUeglColorFormat(0x1E)
CU_EGL_COLOR_FORMAT_YUVA_ER                  = CUeglColorFormat(0x1F)
CU_EGL_COLOR_FORMAT_AYUV_ER                  = CUeglColorFormat(0x20)
CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER         = CUeglColorFormat(0x21)
CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER         = CUeglColorFormat(0x22)
CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER         = CUeglColorFormat(0x23)
CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER     = CUeglColorFormat(0x24)
CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER     = CUeglColorFormat(0x25)
CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER     = CUeglColorFormat(0x26)
CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER         = CUeglColorFormat(0x27)
CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER         = CUeglColorFormat(0x28)
CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER         = CUeglColorFormat(0x29)
CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER     = CUeglColorFormat(0x2A)
CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER     = CUeglColorFormat(0x2B)
CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER     = CUeglColorFormat(0x2C)
CU_EGL_COLOR_FORMAT_BAYER_RGGB               = CUeglColorFormat(0x2D)
CU_EGL_COLOR_FORMAT_BAYER_BGGR               = CUeglColorFormat(0x2E)
CU_EGL_COLOR_FORMAT_BAYER_GRBG               = CUeglColorFormat(0x2F)
CU_EGL_COLOR_FORMAT_BAYER_GBRG               = CUeglColorFormat(0x30)
CU_EGL_COLOR_FORMAT_BAYER10_RGGB             = CUeglColorFormat(0x31)
CU_EGL_COLOR_FORMAT_BAYER10_BGGR             = CUeglColorFormat(0x32)
CU_EGL_COLOR_FORMAT_BAYER10_GRBG             = CUeglColorFormat(0x33)
CU_EGL_COLOR_FORMAT_BAYER10_GBRG             = CUeglColorFormat(0x34)
CU_EGL_COLOR_FORMAT_BAYER12_RGGB             = CUeglColorFormat(0x35)
CU_EGL_COLOR_FORMAT_BAYER12_BGGR             = CUeglColorFormat(0x36)
CU_EGL_COLOR_FORMAT_BAYER12_GRBG             = CUeglColorFormat(0x37)
CU_EGL_COLOR_FORMAT_BAYER12_GBRG             = CUeglColorFormat(0x38)
CU_EGL_COLOR_FORMAT_BAYER14_RGGB             = CUeglColorFormat(0x39)
CU_EGL_COLOR_FORMAT_BAYER14_BGGR             = CUeglColorFormat(0x3A)
CU_EGL_COLOR_FORMAT_BAYER14_GRBG             = CUeglColorFormat(0x3B)
CU_EGL_COLOR_FORMAT_BAYER14_GBRG             = CUeglColorFormat(0x3C)
CU_EGL_COLOR_FORMAT_BAYER20_RGGB             = CUeglColorFormat(0x3D)
CU_EGL_COLOR_FORMAT_BAYER20_BGGR             = CUeglColorFormat(0x3E)
CU_EGL_COLOR_FORMAT_BAYER20_GRBG             = CUeglColorFormat(0x3F)
CU_EGL_COLOR_FORMAT_BAYER20_GBRG             = CUeglColorFormat(0x40)
CU_EGL_COLOR_FORMAT_YVU444_PLANAR            = CUeglColorFormat(0x41)
CU_EGL_COLOR_FORMAT_YVU422_PLANAR            = CUeglColorFormat(0x42)
CU_EGL_COLOR_FORMAT_YVU420_PLANAR            = CUeglColorFormat(0x43)
CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB           = CUeglColorFormat(0x44)
CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR           = CUeglColorFormat(0x45)
CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG           = CUeglColorFormat(0x46)
CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG           = CUeglColorFormat(0x47)
CU_EGL_COLOR_FORMAT_BAYER_BCCR               = CUeglColorFormat(0x48)
CU_EGL_COLOR_FORMAT_BAYER_RCCB               = CUeglColorFormat(0x49)
CU_EGL_COLOR_FORMAT_BAYER_CRBC               = CUeglColorFormat(0x4A)
CU_EGL_COLOR_FORMAT_BAYER_CBRC               = CUeglColorFormat(0x4B)
CU_EGL_COLOR_FORMAT_BAYER10_CCCC             = CUeglColorFormat(0x4C)
CU_EGL_COLOR_FORMAT_BAYER12_BCCR             = CUeglColorFormat(0x4D)
CU_EGL_COLOR_FORMAT_BAYER12_RCCB             = CUeglColorFormat(0x4E)
CU_EGL_COLOR_FORMAT_BAYER12_CRBC             = CUeglColorFormat(0x4F)
CU_EGL_COLOR_FORMAT_BAYER12_CBRC             = CUeglColorFormat(0x50)
CU_EGL_COLOR_FORMAT_BAYER12_CCCC             = CUeglColorFormat(0x51)
CU_EGL_COLOR_FORMAT_Y                        = CUeglColorFormat(0x52)
CU_EGL_COLOR_FORMAT_MAX                      = CUeglColorFormat(0x53)


CUeglFrameType = enum
CU_EGL_FRAME_TYPE_ARRAY = CUeglFrameType(0)
CU_EGL_FRAME_TYPE_PITCH = CUeglFrameType(1)


CUeglResourceLocationFlags = enum
CU_EGL_RESOURCE_LOCATION_SYSMEM = CUeglResourceLocationFlags(0x00)
CU_EGL_RESOURCE_LOCATION_VIDMEM = CUeglResourceLocationFlags(0x01)


CUevent_flags = enum
CU_EVENT_DEFAULT        = CUevent_flags(0x0)
CU_EVENT_BLOCKING_SYNC  = CUevent_flags(0x1)
CU_EVENT_DISABLE_TIMING = CUevent_flags(0x2)
CU_EVENT_INTERPROCESS   = CUevent_flags(0x4)


CUevent_record_flags = enum
CU_EVENT_RECORD_DEFAULT  = CUevent_record_flags(0x0)
CU_EVENT_RECORD_EXTERNAL = CUevent_record_flags(0x1)


CUevent_wait_flags = enum
CU_EVENT_WAIT_DEFAULT = CUevent_wait_flags(0x0)
CU_EVENT_WAIT_EXTERNAL = CUevent_wait_flags(0x1)


CUexternalMemoryHandleType = enum
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD          = CUexternalMemoryHandleType(1)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32       = CUexternalMemoryHandleType(2)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT   = CUexternalMemoryHandleType(3)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP         = CUexternalMemoryHandleType(4)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE     = CUexternalMemoryHandleType(5)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE     = CUexternalMemoryHandleType(6)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = CUexternalMemoryHandleType(7)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF           = CUexternalMemoryHandleType(8)

CUexternalSemaphoreHandleType = enum
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD                = CUexternalSemaphoreHandleType(1)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32             = CUexternalSemaphoreHandleType(2)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT         = CUexternalSemaphoreHandleType(3)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE              = CUexternalSemaphoreHandleType(4)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE              = CUexternalSemaphoreHandleType(5)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC                = CUexternalSemaphoreHandleType(6)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX        = CUexternalSemaphoreHandleType(7)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT    = CUexternalSemaphoreHandleType(8)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD    = CUexternalSemaphoreHandleType(9)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = CUexternalSemaphoreHandleType(10)


CUfilter_mode = enum
CU_TR_FILTER_MODE_POINT  = CUfilter_mode(0)
CU_TR_FILTER_MODE_LINEAR = CUfilter_mode(1)


CUfunc_cache = enum
CU_FUNC_CACHE_PREFER_NONE   = CUfunc_cache(0x00)
CU_FUNC_CACHE_PREFER_SHARED = CUfunc_cache(0x01)
CU_FUNC_CACHE_PREFER_L1     = CUfunc_cache(0x02)
CU_FUNC_CACHE_PREFER_EQUAL  = CUfunc_cache(0x03)


CUfunction_attribute = enum
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK            = CUfunction_attribute(0)
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES                = CUfunction_attribute(1)
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES                 = CUfunction_attribute(2)
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES                 = CUfunction_attribute(3)
CU_FUNC_ATTRIBUTE_NUM_REGS                         = CUfunction_attribute(4)
CU_FUNC_ATTRIBUTE_PTX_VERSION                      = CUfunction_attribute(5)
CU_FUNC_ATTRIBUTE_BINARY_VERSION                   = CUfunction_attribute(6)
CU_FUNC_ATTRIBUTE_CACHE_MODE_CA                    = CUfunction_attribute(7)
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES    = CUfunction_attribute(8)
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = CUfunction_attribute(9)
CU_FUNC_ATTRIBUTE_MAX                              = CUfunction_attribute(10)


CUgraphNodeType = enum
CU_GRAPH_NODE_TYPE_KERNEL           = CUgraphNodeType(0)
CU_GRAPH_NODE_TYPE_MEMCPY           = CUgraphNodeType(1)
CU_GRAPH_NODE_TYPE_MEMSET           = CUgraphNodeType(2)
CU_GRAPH_NODE_TYPE_HOST             = CUgraphNodeType(3)
CU_GRAPH_NODE_TYPE_GRAPH            = CUgraphNodeType(4)
CU_GRAPH_NODE_TYPE_EMPTY            = CUgraphNodeType(5)
CU_GRAPH_NODE_TYPE_WAIT_EVENT       = CUgraphNodeType(6)
CU_GRAPH_NODE_TYPE_EVENT_RECORD     = CUgraphNodeType(7)
CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = CUgraphNodeType(8)
CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT   = CUgraphNodeType(9)


CUgraphicsMapResourceFlags = enum
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE          = CUgraphicsMapResourceFlags(0x00)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY     = CUgraphicsMapResourceFlags(0x01)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = CUgraphicsMapResourceFlags(0x02)

CUgraphicsRegisterFlags = enum
CU_GRAPHICS_REGISTER_FLAGS_NONE           = CUgraphicsRegisterFlags(0x00)
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY      = CUgraphicsRegisterFlags(0x01)
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD  = CUgraphicsRegisterFlags(0x02)
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST   = CUgraphicsRegisterFlags(0x04)
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = CUgraphicsRegisterFlags(0x08)


CUipcMem_flags = enum
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = CUipcMem_flags(0x1)


CUjitInputType = enum
CU_JIT_INPUT_CUBIN     = CUjitInputType(0)
CU_JIT_INPUT_PTX       = CUjitInputType(1)
CU_JIT_INPUT_FATBINARY = CUjitInputType(2)
CU_JIT_INPUT_OBJECT    = CUjitInputType(3)
CU_JIT_INPUT_LIBRARY   = CUjitInputType(4)
CU_JIT_NUM_INPUT_TYPES = CUjitInputType(5)


CUjit_cacheMode = enum
CU_JIT_CACHE_OPTION_NONE = CUjit_cacheMode(0)
CU_JIT_CACHE_OPTION_CG   = CUjit_cacheMode(1)
CU_JIT_CACHE_OPTION_CA   = CUjit_cacheMode(2)


CUjit_fallback = enum
CU_PREFER_PTX    = CUjit_fallback(0)
CU_PREFER_BINARY = CUjit_fallback(1)


CUjit_option = enum
CU_JIT_MAX_REGISTERS               = CUjit_option(0)
CU_JIT_THREADS_PER_BLOCK           = CUjit_option(1)
CU_JIT_WALL_TIME                   = CUjit_option(2)
CU_JIT_INFO_LOG_BUFFER             = CUjit_option(3)
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES  = CUjit_option(4)
CU_JIT_ERROR_LOG_BUFFER            = CUjit_option(5)
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = CUjit_option(6)
CU_JIT_OPTIMIZATION_LEVEL          = CUjit_option(7)
CU_JIT_TARGET_FROM_CUCONTEXT       = CUjit_option(8)
CU_JIT_TARGET                      = CUjit_option(9)
CU_JIT_FALLBACK_STRATEGY           = CUjit_option(10)
CU_JIT_GENERATE_DEBUG_INFO         = CUjit_option(11)
CU_JIT_LOG_VERBOSE                 = CUjit_option(12)
CU_JIT_GENERATE_LINE_INFO          = CUjit_option(13)
CU_JIT_CACHE_MODE                  = CUjit_option(14)
CU_JIT_NEW_SM3X_OPT                = CUjit_option(15)
CU_JIT_FAST_COMPILE                = CUjit_option(16)
CU_JIT_GLOBAL_SYMBOL_NAMES         = CUjit_option(17)
CU_JIT_GLOBAL_SYMBOL_ADDRESSES     = CUjit_option(18)
CU_JIT_GLOBAL_SYMBOL_COUNT         = CUjit_option(19)
CU_JIT_NUM_OPTIONS                 = CUjit_option(20)


CUjit_target = enum
CU_TARGET_COMPUTE_20 = CUjit_target(20)
CU_TARGET_COMPUTE_21 = CUjit_target(21)
CU_TARGET_COMPUTE_30 = CUjit_target(30)
CU_TARGET_COMPUTE_32 = CUjit_target(32)
CU_TARGET_COMPUTE_35 = CUjit_target(35)
CU_TARGET_COMPUTE_37 = CUjit_target(37)
CU_TARGET_COMPUTE_50 = CUjit_target(50)
CU_TARGET_COMPUTE_52 = CUjit_target(52)
CU_TARGET_COMPUTE_53 = CUjit_target(53)
CU_TARGET_COMPUTE_60 = CUjit_target(60)
CU_TARGET_COMPUTE_61 = CUjit_target(61)
CU_TARGET_COMPUTE_62 = CUjit_target(62)
CU_TARGET_COMPUTE_70 = CUjit_target(70)
CU_TARGET_COMPUTE_72 = CUjit_target(72)
CU_TARGET_COMPUTE_75 = CUjit_target(75)
CU_TARGET_COMPUTE_80 = CUjit_target(80)
CU_TARGET_COMPUTE_86 = CUjit_target(86)


CUkernelNodeAttrID = enum
CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = CUkernelNodeAttrID(1)
CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE          = CUkernelNodeAttrID(2)


CUlimit = enum
CU_LIMIT_STACK_SIZE                       = CUlimit(0x00)
CU_LIMIT_PRINTF_FIFO_SIZE                 = CUlimit(0x01)
CU_LIMIT_MALLOC_HEAP_SIZE                 = CUlimit(0x02)
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           = CUlimit(0x03)
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = CUlimit(0x04)
CU_LIMIT_MAX_L2_FETCH_GRANULARITY         = CUlimit(0x05)
CU_LIMIT_PERSISTING_L2_CACHE_SIZE         = CUlimit(0x06)
CU_LIMIT_MAX                              = CUlimit(0x07)


CUmemAccess_flags = enum
CU_MEM_ACCESS_FLAGS_PROT_NONE      = CUmemAccess_flags(0x0)
CU_MEM_ACCESS_FLAGS_PROT_READ      = CUmemAccess_flags(0x1)
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = CUmemAccess_flags(0x3)
CU_MEM_ACCESS_FLAGS_PROT_MAX       = CUmemAccess_flags(0xFFFFFFFF)


CUmemAllocationCompType = enum
CU_MEM_ALLOCATION_COMP_NONE    = CUmemAllocationCompType(0x0)
CU_MEM_ALLOCATION_COMP_GENERIC = CUmemAllocationCompType(0x1)


CUmemAllocationGranularity_flags = enum
CU_MEM_ALLOC_GRANULARITY_MINIMUM     = CUmemAllocationGranularity_flags(0x0)
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = CUmemAllocationGranularity_flags(0x1)


CUmemAllocationHandleType = enum
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = CUmemAllocationHandleType(0x1)
CU_MEM_HANDLE_TYPE_WIN32                 = CUmemAllocationHandleType(0x2)
CU_MEM_HANDLE_TYPE_WIN32_KMT             = CUmemAllocationHandleType(0x4)
CU_MEM_HANDLE_TYPE_MAX                   = CUmemAllocationHandleType(0xFFFFFFFF)


CUmemAllocationType = enum
CU_MEM_ALLOCATION_TYPE_INVALID = CUmemAllocationType(0x0)
CU_MEM_ALLOCATION_TYPE_PINNED  = CUmemAllocationType(0x1)
CU_MEM_ALLOCATION_TYPE_MAX     = CUmemAllocationType(0xFFFFFFFF)


CUmemAttach_flags = enum
CU_MEM_ATTACH_GLOBAL = CUmemAttach_flags(0x1)
CU_MEM_ATTACH_HOST   = CUmemAttach_flags(0x2)
CU_MEM_ATTACH_SINGLE = CUmemAttach_flags(0x4)


CUmemHandleType = enum
CU_MEM_HANDLE_TYPE_GENERIC = CUmemHandleType(0)


CUmemLocationType = enum
CU_MEM_LOCATION_TYPE_INVALID = CUmemLocationType(0x0)
CU_MEM_LOCATION_TYPE_DEVICE  = CUmemLocationType(0x1)
CU_MEM_LOCATION_TYPE_MAX     = CUmemLocationType(0xFFFFFFFF)


CUmemOperationType = enum
CU_MEM_OPERATION_TYPE_MAP = CUmemOperationType(1)
CU_MEM_OPERATION_TYPE_UNMAP = CUmemOperationType(2)


CUmem_advise = enum
CU_MEM_ADVISE_SET_READ_MOSTLY          = CUmem_advise(1)
CU_MEM_ADVISE_UNSET_READ_MOSTLY        = CUmem_advise(2)
CU_MEM_ADVISE_SET_PREFERRED_LOCATION   = CUmem_advise(3)
CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = CUmem_advise(4)
CU_MEM_ADVISE_SET_ACCESSED_BY          = CUmem_advise(5)
CU_MEM_ADVISE_UNSET_ACCESSED_BY        = CUmem_advise(6)


CUmemorytype = enum
CU_MEMORYTYPE_HOST    = CUmemorytype(0x01)
CU_MEMORYTYPE_DEVICE  = CUmemorytype(0x02)
CU_MEMORYTYPE_ARRAY   = CUmemorytype(0x03)
CU_MEMORYTYPE_UNIFIED = CUmemorytype(0x04)


CUoccupancy_flags = enum
CU_OCCUPANCY_DEFAULT                  = CUoccupancy_flags(0x0)
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = CUoccupancy_flags(0x1)


CUpointer_attribute = enum
CU_POINTER_ATTRIBUTE_CONTEXT                    = CUpointer_attribute(1)
CU_POINTER_ATTRIBUTE_MEMORY_TYPE                = CUpointer_attribute(2)
CU_POINTER_ATTRIBUTE_DEVICE_POINTER             = CUpointer_attribute(3)
CU_POINTER_ATTRIBUTE_HOST_POINTER               = CUpointer_attribute(4)
CU_POINTER_ATTRIBUTE_P2P_TOKENS                 = CUpointer_attribute(5)
CU_POINTER_ATTRIBUTE_SYNC_MEMOPS                = CUpointer_attribute(6)
CU_POINTER_ATTRIBUTE_BUFFER_ID                  = CUpointer_attribute(7)
CU_POINTER_ATTRIBUTE_IS_MANAGED                 = CUpointer_attribute(8)
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL             = CUpointer_attribute(9)
CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = CUpointer_attribute(10)
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR           = CUpointer_attribute(11)
CU_POINTER_ATTRIBUTE_RANGE_SIZE                 = CUpointer_attribute(12)
CU_POINTER_ATTRIBUTE_MAPPED                     = CUpointer_attribute(13)
CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES       = CUpointer_attribute(14)
CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = CUpointer_attribute(15)
CU_POINTER_ATTRIBUTE_ACCESS_FLAGS               = CUpointer_attribute(16)


CUresourceViewFormat = enum
CU_RES_VIEW_FORMAT_NONE          = CUresourceViewFormat(0x00)
CU_RES_VIEW_FORMAT_UINT_1X8      = CUresourceViewFormat(0x01)
CU_RES_VIEW_FORMAT_UINT_2X8      = CUresourceViewFormat(0x02)
CU_RES_VIEW_FORMAT_UINT_4X8      = CUresourceViewFormat(0x03)
CU_RES_VIEW_FORMAT_SINT_1X8      = CUresourceViewFormat(0x04)
CU_RES_VIEW_FORMAT_SINT_2X8      = CUresourceViewFormat(0x05)
CU_RES_VIEW_FORMAT_SINT_4X8      = CUresourceViewFormat(0x06)
CU_RES_VIEW_FORMAT_UINT_1X16     = CUresourceViewFormat(0x07)
CU_RES_VIEW_FORMAT_UINT_2X16     = CUresourceViewFormat(0x08)
CU_RES_VIEW_FORMAT_UINT_4X16     = CUresourceViewFormat(0x09)
CU_RES_VIEW_FORMAT_SINT_1X16     = CUresourceViewFormat(0x0a)
CU_RES_VIEW_FORMAT_SINT_2X16     = CUresourceViewFormat(0x0b)
CU_RES_VIEW_FORMAT_SINT_4X16     = CUresourceViewFormat(0x0c)
CU_RES_VIEW_FORMAT_UINT_1X32     = CUresourceViewFormat(0x0d)
CU_RES_VIEW_FORMAT_UINT_2X32     = CUresourceViewFormat(0x0e)
CU_RES_VIEW_FORMAT_UINT_4X32     = CUresourceViewFormat(0x0f)
CU_RES_VIEW_FORMAT_SINT_1X32     = CUresourceViewFormat(0x10)
CU_RES_VIEW_FORMAT_SINT_2X32     = CUresourceViewFormat(0x11)
CU_RES_VIEW_FORMAT_SINT_4X32     = CUresourceViewFormat(0x12)
CU_RES_VIEW_FORMAT_FLOAT_1X16    = CUresourceViewFormat(0x13)
CU_RES_VIEW_FORMAT_FLOAT_2X16    = CUresourceViewFormat(0x14)
CU_RES_VIEW_FORMAT_FLOAT_4X16    = CUresourceViewFormat(0x15)
CU_RES_VIEW_FORMAT_FLOAT_1X32    = CUresourceViewFormat(0x16)
CU_RES_VIEW_FORMAT_FLOAT_2X32    = CUresourceViewFormat(0x17)
CU_RES_VIEW_FORMAT_FLOAT_4X32    = CUresourceViewFormat(0x18)
CU_RES_VIEW_FORMAT_UNSIGNED_BC1  = CUresourceViewFormat(0x19)
CU_RES_VIEW_FORMAT_UNSIGNED_BC2  = CUresourceViewFormat(0x1a)
CU_RES_VIEW_FORMAT_UNSIGNED_BC3  = CUresourceViewFormat(0x1b)
CU_RES_VIEW_FORMAT_UNSIGNED_BC4  = CUresourceViewFormat(0x1c)
CU_RES_VIEW_FORMAT_SIGNED_BC4    = CUresourceViewFormat(0x1d)
CU_RES_VIEW_FORMAT_UNSIGNED_BC5  = CUresourceViewFormat(0x1e)
CU_RES_VIEW_FORMAT_SIGNED_BC5    = CUresourceViewFormat(0x1f)
CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = CUresourceViewFormat(0x20)
CU_RES_VIEW_FORMAT_SIGNED_BC6H   = CUresourceViewFormat(0x21)
CU_RES_VIEW_FORMAT_UNSIGNED_BC7  = CUresourceViewFormat(0x22)


CUresourcetype = enum
CU_RESOURCE_TYPE_ARRAY           = CUresourcetype(0x00)
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = CUresourcetype(0x01)
CU_RESOURCE_TYPE_LINEAR          = CUresourcetype(0x02)
CU_RESOURCE_TYPE_PITCH2D         = CUresourcetype(0x03)


CUresult = enum
CUDA_SUCCESS                              = CUresult(0)
CUDA_ERROR_INVALID_VALUE                  = CUresult(1)
CUDA_ERROR_OUT_OF_MEMORY                  = CUresult(2)
CUDA_ERROR_NOT_INITIALIZED                = CUresult(3)
CUDA_ERROR_DEINITIALIZED                  = CUresult(4)
CUDA_ERROR_PROFILER_DISABLED              = CUresult(5)
CUDA_ERROR_PROFILER_NOT_INITIALIZED       = CUresult(6)
CUDA_ERROR_PROFILER_ALREADY_STARTED       = CUresult(7)
CUDA_ERROR_PROFILER_ALREADY_STOPPED       = CUresult(8)
CUDA_ERROR_STUB_LIBRARY                   = CUresult(34)
CUDA_ERROR_NO_DEVICE                      = CUresult(100)
CUDA_ERROR_INVALID_DEVICE                 = CUresult(101)
CUDA_ERROR_DEVICE_NOT_LICENSED            = CUresult(102)
CUDA_ERROR_INVALID_IMAGE                  = CUresult(200)
CUDA_ERROR_INVALID_CONTEXT                = CUresult(201)
CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = CUresult(202)
CUDA_ERROR_MAP_FAILED                     = CUresult(205)
CUDA_ERROR_UNMAP_FAILED                   = CUresult(206)
CUDA_ERROR_ARRAY_IS_MAPPED                = CUresult(207)
CUDA_ERROR_ALREADY_MAPPED                 = CUresult(208)
CUDA_ERROR_NO_BINARY_FOR_GPU              = CUresult(209)
CUDA_ERROR_ALREADY_ACQUIRED               = CUresult(210)
CUDA_ERROR_NOT_MAPPED                     = CUresult(211)
CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = CUresult(212)
CUDA_ERROR_NOT_MAPPED_AS_POINTER          = CUresult(213)
CUDA_ERROR_ECC_UNCORRECTABLE              = CUresult(214)
CUDA_ERROR_UNSUPPORTED_LIMIT              = CUresult(215)
CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = CUresult(216)
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = CUresult(217)
CUDA_ERROR_INVALID_PTX                    = CUresult(218)
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = CUresult(219)
CUDA_ERROR_NVLINK_UNCORRECTABLE           = CUresult(220)
CUDA_ERROR_JIT_COMPILER_NOT_FOUND         = CUresult(221)
CUDA_ERROR_UNSUPPORTED_PTX_VERSION        = CUresult(222)
CUDA_ERROR_JIT_COMPILATION_DISABLED       = CUresult(223)
CUDA_ERROR_INVALID_SOURCE                 = CUresult(300)
CUDA_ERROR_FILE_NOT_FOUND                 = CUresult(301)
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = CUresult(302)
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = CUresult(303)
CUDA_ERROR_OPERATING_SYSTEM               = CUresult(304)
CUDA_ERROR_INVALID_HANDLE                 = CUresult(400)
CUDA_ERROR_ILLEGAL_STATE                  = CUresult(401)
CUDA_ERROR_NOT_FOUND                      = CUresult(500)
CUDA_ERROR_NOT_READY                      = CUresult(600)
CUDA_ERROR_ILLEGAL_ADDRESS                = CUresult(700)
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = CUresult(701)
CUDA_ERROR_LAUNCH_TIMEOUT                 = CUresult(702)
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = CUresult(703)
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = CUresult(704)
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = CUresult(705)
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = CUresult(708)
CUDA_ERROR_CONTEXT_IS_DESTROYED           = CUresult(709)
CUDA_ERROR_ASSERT                         = CUresult(710)
CUDA_ERROR_TOO_MANY_PEERS                 = CUresult(711)
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = CUresult(712)
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = CUresult(713)
CUDA_ERROR_HARDWARE_STACK_ERROR           = CUresult(714)
CUDA_ERROR_ILLEGAL_INSTRUCTION            = CUresult(715)
CUDA_ERROR_MISALIGNED_ADDRESS             = CUresult(716)
CUDA_ERROR_INVALID_ADDRESS_SPACE          = CUresult(717)
CUDA_ERROR_INVALID_PC                     = CUresult(718)
CUDA_ERROR_LAUNCH_FAILED                  = CUresult(719)
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = CUresult(720)
CUDA_ERROR_NOT_PERMITTED                  = CUresult(800)
CUDA_ERROR_NOT_SUPPORTED                  = CUresult(801)
CUDA_ERROR_SYSTEM_NOT_READY               = CUresult(802)
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH         = CUresult(803)
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = CUresult(804)
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED     = CUresult(900)
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED     = CUresult(901)
CUDA_ERROR_STREAM_CAPTURE_MERGE           = CUresult(902)
CUDA_ERROR_STREAM_CAPTURE_UNMATCHED       = CUresult(903)
CUDA_ERROR_STREAM_CAPTURE_UNJOINED        = CUresult(904)
CUDA_ERROR_STREAM_CAPTURE_ISOLATION       = CUresult(905)
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT        = CUresult(906)
CUDA_ERROR_CAPTURED_EVENT                 = CUresult(907)
CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD    = CUresult(908)
CUDA_ERROR_TIMEOUT                        = CUresult(909)
CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE      = CUresult(910)
CUDA_ERROR_UNKNOWN                        = CUresult(999)


CUshared_carveout = enum
CU_SHAREDMEM_CARVEOUT_DEFAULT    = CUshared_carveout(-1)
CU_SHAREDMEM_CARVEOUT_MAX_SHARED = CUshared_carveout(100)
CU_SHAREDMEM_CARVEOUT_MAX_L1     = CUshared_carveout(0)


CUsharedconfig = enum
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE    = CUsharedconfig(0x00)
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  = CUsharedconfig(0x01)
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = CUsharedconfig(0x02)


CUstreamAttrID = enum
CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW   = CUstreamAttrID(1)
CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = CUstreamAttrID(3)


CUstreamBatchMemOpType = enum
CU_STREAM_MEM_OP_WAIT_VALUE_32       = CUstreamBatchMemOpType(1)
CU_STREAM_MEM_OP_WRITE_VALUE_32      = CUstreamBatchMemOpType(2)
CU_STREAM_MEM_OP_WAIT_VALUE_64       = CUstreamBatchMemOpType(4)
CU_STREAM_MEM_OP_WRITE_VALUE_64      = CUstreamBatchMemOpType(5)
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = CUstreamBatchMemOpType(3)


CUstreamCaptureMode = enum
CU_STREAM_CAPTURE_MODE_GLOBAL       = CUstreamCaptureMode(0)
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = CUstreamCaptureMode(1)
CU_STREAM_CAPTURE_MODE_RELAXED      = CUstreamCaptureMode(2)


CUstreamCaptureStatus = enum
CU_STREAM_CAPTURE_STATUS_NONE        = CUstreamCaptureStatus(0)
CU_STREAM_CAPTURE_STATUS_ACTIVE      = CUstreamCaptureStatus(1)
CU_STREAM_CAPTURE_STATUS_INVALIDATED = CUstreamCaptureStatus(2)


CUstreamWaitValue_flags = enum
CU_STREAM_WAIT_VALUE_GEQ   = CUstreamWaitValue_flags(0x0)
CU_STREAM_WAIT_VALUE_EQ    = CUstreamWaitValue_flags(0x1)
CU_STREAM_WAIT_VALUE_AND   = CUstreamWaitValue_flags(0x2)
CU_STREAM_WAIT_VALUE_NOR   = CUstreamWaitValue_flags(0x3)
CU_STREAM_WAIT_VALUE_FLUSH = CUstreamWaitValue_flags(1<<30)


CUstreamWriteValue_flags = enum
CU_STREAM_WRITE_VALUE_DEFAULT           = CUstreamWriteValue_flags(0x0)
CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = CUstreamWriteValue_flags(0x1)


CUstream_flags = enum
CU_STREAM_DEFAULT      = CUstream_flags(0x0)
CU_STREAM_NON_BLOCKING = CUstream_flags(0x1)





CUmem_range_attribute = enum
CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY            = CUmem_range_attribute(1)
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION     = CUmem_range_attribute(2)
CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY            = CUmem_range_attribute(3)
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = CUmem_range_attribute(4)


CUsynchronizationPolicy = enum
CU_SYNC_POLICY_AUTO          = CUsynchronizationPolicy(1)
CU_SYNC_POLICY_SPIN          = CUsynchronizationPolicy(2)
CU_SYNC_POLICY_YIELD         = CUsynchronizationPolicy(3)
CU_SYNC_POLICY_BLOCKING_SYNC = CUsynchronizationPolicy(4)


CUGLDeviceList = enum
CU_GL_DEVICE_LIST_ALL = CUGLDeviceList(0x01)
CU_GL_DEVICE_LIST_CURRENT_FRAME = CUGLDeviceList(0x02)
CU_GL_DEVICE_LIST_NEXT_FRAME = CUGLDeviceList(0x03)
