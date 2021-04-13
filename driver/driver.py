from .api import VERSION_PROTOTYPE, API_PROTOTYPES
from .error import DriverSupportError
from pycu.libutils import find_lib

import sys
import ctypes
import threading

_driver_lock = threading.Lock()

class CudaSupportError(ImportError):
	pass

def find_driver():
	#installed at GPU driver location

	paths = []

	if sys.platform == 'linux':
		loader = ctypes.CDLL
		paths.extend(['/usr/lib/x86_64-linux-gnu','/usr/lib', '/usr/lib64'])
		names = ['libcuda.so', 'libcuda.so.1']
	elif sys.platform == 'win32':
		loader = ctypes.WinDLL
		paths.append('\\windows\\system32')
		names = ['nvcuda.dll']
	# elif sys.platform == 'darwin': #OS x
	# 	loader = ctypes.CDLL
	# 	path = ['/usr/local/cuda/lib']
	# 	name = ['libcuda.dylib']
	else:
		raise ValueError('platform not supported')

	try:
		lib = find_lib(loader, paths, names)
	except Exception as err:
		raise CudaSupportError('CUDA driver cannot be found')

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

def find_runtime():
	pass
	# /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudadevrt.a

	# def find_nvvm():
	# 	#installed in runtime library location (wherever CUDA toolkit is installed)

	# 	if sys.platform == 'linux':
	# 		paths = get_cuda_libvar()
	# 		loader = ctypes.CDLL
	# 		paths.extend(['/usr/local/cuda', '/usr/local/cuda/nvvm', '/usr/local/cuda/nvvm/lib64'])
	# 		names = ['libnvvm.so']
	# 	elif sys.platform == 'win32':
	# 		paths = get_cuda_libvar(os.path.join('nvvm', 'bin'))
	# 		loader = ctypes.WinDLL
	# 		names = [*determine_file_name(paths, 'nvvm64.*\.dll')]
	# 		# names = [*determine_file_name(paths, 'nvvm64.*\\.dll')]

	# 	# elif sys.platform == 'darwin': #OS x
	# 		# loader = ctypes.CDLL
	# 		# paths.append(['/usr/local/cuda/lib'])
	# 		# names = ['libcuda.dylib']
	# 	else:
	# 		raise ValueError('platform not supported')

	# 	try:
	# 		lib = find_lib(loader, paths, names)
	# 	except Exception as err:
	# 		raise CudaSupportError('NVVM cannot be found')

	# 	return lib


	# def find_libdevice():
	# 	cuda_dir = get_cuda_dir()

	# 	paths = []
	# 	if cuda_dir:
	# 		paths.append(os.path.join(cuda_dir, 'nvvm', 'libdevice'))

	# 	if sys.platform == 'linux':
	# 		paths.extend(['/usr/local/cuda/nvvm/libdevice'])
	# 	elif sys.platform == 'win32':
	# 		pass
	# 	# elif sys.platform == 'darwin': #OS x
	# 		# names = ['libcuda.dylib']
	# 	else:
	# 		raise ValueError('platform not supported')

	# 	pat = r'libdevice(\.(?P<arch>compute_\d+))?(\.\d+)*\.bc$'
	# 	paths = find_file(re.compile(pat), paths)

	# 	arches = defaultdict(list)
	# 	for path in paths:
	# 		m = re.search(pat, path)
	# 		arch = m.group('arch')
	# 		if arch:
	# 			arch = int(arch.split('_'[-1]))
	# 		arches[arch].append(path)
	# 	arches = {k: max(v) for k, v in arches.items()}
	# 	return arches

	# def find_file(pat, dirs):
	# 	if isinstance(dirs, str):
	# 		dirs = [dirs]

	# 	paths = []
	# 	for d in dirs:
	# 		files = os.listdir(d)
	# 		candidates = [os.path.join(d, file) for file in files if pat.match(file)]
	# 		paths.extend([c for c in candidates if os.path.isfile(c)])
	# 	return paths




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
