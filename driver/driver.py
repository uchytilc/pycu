from .api import VERSION_PROTOTYPE, API_PROTOTYPES
from .error import DriverSupportError
from pycu.libutils import find_lib, CudaSupportError

import sys
import ctypes
import threading

_driver_lock = threading.Lock()

# import platform
# print(platform.architecture())

def find_driver():
	#installed at GPU driver location

	paths = []

	if sys.platform == 'win32':
		loader = ctypes.WinDLL
		paths.append('\\windows\\system32')
		names = ['nvcuda.dll']
	# elif sys.platform == 'darwin': #OS x
		# loader = ctypes.CDLL
		# path = ['/usr/local/cuda/lib']
		# name = ['libcuda.dylib']
	else:
		loader = ctypes.CDLL
		paths.extend(['/usr/lib/x86_64-linux-gnu','/usr/lib','/usr/lib64'])
		names = ['libcuda.so', 'libcuda.so.1']

	try:
		lib = find_lib(loader, paths, names)
	except Exception as err:
		raise CudaSupportError('CUDA Driver could not be found')

	return lib

def get_driver_version(driver):
	#load cuDriverGetVersion to determine the appropriate CUDA api to load

	name, proto = VERSION_PROTOTYPE
	cuDriverGetVersion = getattr(driver, name, None)
	cuDriverGetVersion.restype = proto[0]
	cuDriverGetVersion.argtypes = proto[1:]

	version = ctypes.c_int()
	err = cuDriverGetVersion(ctypes.byref(version))

	return version.value

class Driver:
	__singleton = None

	# def __getattr__(self, attr):
		# raise AttributeError(f"The CUDA driver function '{attr}' is not supported in CUDA Driver API version {self.version()}")

	def __new__(cls):
		with _driver_lock:
			#if there is no instance of the class create a new one, otherwise return the original instance
			if cls.__singleton is None:
				cls.__singleton = inst = object.__new__(cls)
				try:
					driver = find_driver()
				except OSError as e:
					cls.__singleton = None
					errmsg = ("Cannot find Driver API")
					raise DriverSupportError(errmsg % e)

				api = API_PROTOTYPES[get_driver_version(driver)]()

				#find & populate functions
				for name, proto in api.items():
					#most (newer) CUDA functions include a _v2 in their name and are aliased to the same name without _v2
					func = getattr(driver, name + "_v2", None)
					#if function isn't found with _v2 (or v2 is set to false) see if non-_v2 version exists
					if not func:
						func = getattr(driver, name, None)

					if func:
						func.restype = proto[0]
						func.argtypes = proto[1:]
						setattr(inst, name, func)

		return cls.__singleton
