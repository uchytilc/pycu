from .api import API_PROTOTYPES
from .error import NvvmSupportError
from pycu.libutils import open_file, find_lib, get_cuda_libvar, get_cuda_dir, determine_file_name

from collections import defaultdict
import sys
import ctypes
import threading
import re
import os

_nvvm_lock = threading.Lock()

class CudaSupportError(ImportError):
	pass

def find_nvvm():
	#installed in runtime library location (wherever CUDA toolkit is installed)

	if sys.platform == 'linux':
		paths = get_cuda_libvar()
		loader = ctypes.CDLL
		paths.extend(['/usr/local/cuda', '/usr/local/cuda/nvvm', '/usr/local/cuda/nvvm/lib64'])
		names = ['libnvvm.so']
	elif sys.platform == 'win32':
		paths = get_cuda_libvar(os.path.join('nvvm', 'bin'))
		loader = ctypes.WinDLL
		names = [*determine_file_name(paths, 'nvvm64.*\.dll')]
		# names = [*determine_file_name(paths, 'nvvm64.*\\.dll')]

	# elif sys.platform == 'darwin': #OS x
		# loader = ctypes.CDLL
		# paths.append(['/usr/local/cuda/lib'])
		# names = ['libcuda.dylib']
	else:
		raise ValueError('platform not supported')

	try:
		lib = find_lib(loader, paths, names)
	except Exception as err:
		raise CudaSupportError('NVVM cannot be found')

	return lib

class Driver(object):
	__singleton = None

	def __new__(cls):
		with _nvvm_lock:
			if cls.__singleton is None:
				cls.__singleton = inst = object.__new__(cls)
				try:
					inst.driver = find_nvvm()
				except OSError as e:
					cls.__singleton = None
					errmsg = ("Cannot find nvvm")
					raise NvvmSupportError(errmsg % e)

				# Find & populate functions
				for name, proto in API_PROTOTYPES.items():
					try:
						func = getattr(inst.driver, name)
						func.restype = proto[0]
						func.argtypes = proto[1:]
						setattr(inst, name, func)
					except:
						pass

		return cls.__singleton

#known_archs example:
	#arch = [None,20,30,35,50]

def get_libdevice(arch = None):
	files = find_libdevice()
	#sorts with None at the end
	options = sorted(list(files.keys()), key = lambda x: (x is None, x))
	arch = get_closest_arch(arch, options)

	return {arch:open_file(files[arch], 'rb')}

def get_closest_arch(arch, options):
	if arch is None:
		return options[-1]

	for potential in reversed(options[:-1]):
		if potential <= arch:
			return potential
	return options[-1]

def find_libdevice():
	cuda_dir = get_cuda_dir()

	paths = []
	if cuda_dir:
		paths.append(os.path.join(cuda_dir, 'nvvm', 'libdevice'))

	if sys.platform == 'linux':
		paths.extend(['/usr/local/cuda/nvvm/libdevice'])
	elif sys.platform == 'win32':
		pass
	# elif sys.platform == 'darwin': #OS x
		# names = ['libcuda.dylib']
	else:
		raise ValueError('platform not supported')

	pat = r'libdevice(\.(?P<arch>compute_\d+))?(\.\d+)*\.bc$'
	paths = find_file(re.compile(pat), paths)

	arches = defaultdict(list)
	for path in paths:
		m = re.search(pat, path)
		arch = m.group('arch')
		if arch:
			arch = int(arch.split('_'[-1]))
		arches[arch].append(path)
	arches = {k: max(v) for k, v in arches.items()}
	return arches

def find_file(pat, dirs):
	if isinstance(dirs, str):
		dirs = [dirs]

	paths = []
	for d in dirs:
		files = os.listdir(d)
		candidates = [os.path.join(d, file) for file in files if pat.match(file)]
		paths.extend([c for c in candidates if os.path.isfile(c)])
	return paths
