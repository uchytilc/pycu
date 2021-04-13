from .api import API_PROTOTYPES
from .error import NvrtcSupportError
from pycu.libutils import find_lib, get_cuda_libvar, determine_file_name

import sys
import ctypes
import threading

_nvrtc_lock = threading.Lock()

class CudaSupportError(ImportError):
	pass

def find_nvrtc():
	#installed in runtime library location (wherever CUDA toolkit is installed)

	if sys.platform == 'linux':
		paths = get_cuda_libvar()
		loader = ctypes.CDLL
		paths.append('/usr/local/cuda')
		names = ['libnvrtc.so']
	elif sys.platform == 'win32':
		paths = get_cuda_libvar('bin')
		loader = ctypes.WinDLL
		# paths.append('\\windows\\system32')
		names = [*determine_file_name(paths, 'nvrtc64.*\.dll')] # nvrtc-builtins64_110.dll
	# elif sys.platform == 'darwin': #OS x
		# loader = ctypes.CDLL
		# paths.append(['/usr/local/cuda/lib'])
		# names = ['libcuda.dylib']
	else:
		raise ValueError('platform not supported')

	try:
		lib = find_lib(loader, paths, names)
	except Exception as err:
		raise CudaSupportError('NVRTC cannot be found')

	return lib

class Driver(object):
	__singleton = None

	def __new__(cls):
		with _nvrtc_lock:
			if cls.__singleton is None:
				cls.__singleton = inst = object.__new__(cls)
				try:
					driver = find_nvrtc()
				except OSError as e:
					cls.__singleton = None
					errmsg = ("Cannot find nvrtc")
					raise NvrtcSupportError(errmsg % e)

				#Find & populate functions
				for name, proto in API_PROTOTYPES.items():
					func = getattr(driver, name)
					func.restype = proto[0]
					func.argtypes = proto[1:]
					setattr(inst, name, func)

		return cls.__singleton
