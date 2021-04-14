from .api import API_PROTOTYPES
from .error import NvrtcSupportError
from pycu.libutils import *
from pycu.utils import open_file

import sys
import ctypes
import threading

_nvrtc_lock = threading.Lock()

def find_nvrtc():
	if sys.platform == 'win32':
		paths = get_cuda_libpath('bin')
		# paths.append('\\windows\\system32')
		loader = ctypes.WinDLL
		names = [*determine_file_name(paths, 'nvrtc64.*\.dll')] # nvrtc-builtins64_110.dll
	# elif sys.platform == 'darwin': #OS x
		# loader = ctypes.CDLL
		# paths.append(['/usr/local/cuda/lib'])
		# names = ['libcuda.dylib']
	else:
		paths = get_cuda_libpath('cuda/lib64')
		loader = ctypes.CDLL
		names = ['libnvrtc.so']

	try:
		lib = find_lib(loader, paths, names)
	except Exception as err:
		raise CudaSupportError('NVRTC could not be found')

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

#cuda device runtime
_libcudadevrt = None

def get_libcudadevrt():
	global _libcudadevrt

	if _libcudadevrt is None:
		if sys.platform == 'win32':
			paths = get_cuda_libpath(os.path.join('lib', 'x64'))
			loader = ctypes.WinDLL
			names = ['cudadevrt.lib']
		# elif sys.platform == 'darwin': #OS x
			# loader = ctypes.CDLL
			# paths.append(['/usr/local/cuda/lib'])
			# names = ['libcuda.dylib']
		else:
			paths = get_cuda_libpath('cuda/lib64')
			loader = ctypes.CDLL
			names = ['libcudadevrt.a']

		candidates = get_file_candidates(paths, names)
		for path in candidates:
			try:
				_libcudadevrt = open_file(path, 'rb')
				break
			except Exception as err:
				raise CudaSupportError('libcudadevrt.a could not be found')
	return _libcudadevrt
