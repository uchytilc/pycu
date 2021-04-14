from .api import API_PROTOTYPES
from .error import NvvmSupportError
from pycu.libutils import *
from pycu.utils import open_file

from collections import defaultdict
import sys
import ctypes
import threading
import re
import os

_nvvm_lock = threading.Lock()

def find_nvvm():
	#installed in runtime library location (wherever CUDA toolkit is installed)

	if sys.platform == 'linux':
		paths = get_cuda_libpath()
		paths.extend(['/usr/local/cuda/nvvm', '/usr/local/cuda/nvvm/lib64'])
		loader = ctypes.CDLL
		names = ['libnvvm.so']
	elif sys.platform == 'win32':
		paths = get_cuda_libpath(os.path.join('nvvm', 'bin'))
		loader = ctypes.WinDLL
		names = [*determine_file_name(paths, 'nvvm64.*\.dll')]
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

#key: arch associated with libdevice (None indicates libdevice is not arch specific)
#value: libdevice source
_libdevice = {}

#key:given arch
#value: closest available arch found
_searched_arch = {}

def get_libdevice(arch = None):
	#known_archs example:
		#arch = [None,20,30,35,50]
	global _libdevice, _searched_arch

	lib = _libdevice.get(arch, None)
	if lib is None:
		#note: use False instead of None in searched_arch.get when indicating failure to prevent getting None key from libdevice (libdevice with no "compute_" is stored under None key)
		lib = _libdevice.get(_searched_arch.get(arch, False), None)
	if lib is None:
		libdevice = find_libdevice()
		#sort with None at the end
		options = sorted(list(libdevice.keys()), key = lambda x: (x is None, x))
		found_arch = get_closest_arch(arch, options)
		lib = open_file(libdevice[found_arch], 'rb')
		#cache found libdevice and arch
		_searched_arch[arch] = found_arch
		_libdevice[arch] = lib

	return lib

def get_closest_arch(arch, options):
	if arch is None:
		return options[-1]
	for potential in reversed(options[:-1]):
		if potential <= arch:
			return potential
	return options[-1]

def find_libdevice():
	paths = get_cuda_libpath(os.path.join('nvvm', 'libdevice'))
	pat = r'libdevice(\.(?P<arch>compute_\d+))?(\.\d+)*\.bc$'
	found = find_file_from_pattern(re.compile(pat), paths)

	if not found:
		locations = '\n'.join(paths)
		raise CudaSupportError(f'libdevice could not be found at any of the following locations {locations}')

	arches = defaultdict(list)
	for path in found:
		m = re.search(pat, path)
		arch = m.group('arch')
		if arch:
			arch = int(arch.split('_'[-1]))
		arches[arch].append(path)
	arches = {k: max(v) for k, v in arches.items()}
	return arches
