from .api import VERSION_PROTOTYPE, API_PROTOTYPES
from .error import DriverSupportError
from pycu.libutils import find_lib, CudaSupportError

import sys
import os
import ctypes
import threading
import string

_driver_lock = threading.Lock()

def find_driver():
	#installed at GPU driver location

	paths = []

	#TODO
		#look for python function to get all letter drives
		#insert letter drives into each path check
			#have paths sorted in alphabetical order

	if sys.platform == 'win32':
		loader = ctypes.WinDLL
		for drive in [f"{drive}:" for drive in string.ascii_uppercase if os.path.exists(f"{drive}:")]:
			paths.append(f"{drive}\\windows\\system32")
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

				version = get_driver_version(driver)

				try:
					api, alias = API_PROTOTYPES[version]()
				except:
					#raise warning

					api = {}

					#fall back to older version API
					for supported_version in sorted(API_PROTOTYPES.keys(), reverse = True):
						if supported_version < version:
							api, alias = API_PROTOTYPES[supported_version]()
							break

					if not api:
						raise CudaSupportError("Could not find supported API based on driver verison")

				#find & populate functions
				for name, proto in api.items():
					func = getattr(driver, name, None)

					if func:
						func.restype = proto[0]
						func.argtypes = proto[1:]
						setattr(inst, name, func)

				for name, name_v in alias.items():
					func = getattr(driver, name_v, None)
					setattr(inst, name, func)

		return cls.__singleton
