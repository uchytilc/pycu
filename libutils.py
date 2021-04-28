import os
import sys
from itertools import product
import re

class CudaSupportError(ImportError):
	pass

def determine_file_name(paths, pattern):
	#finds all files within paths that match the given pattern

	candidates = []
	regex = re.compile(pattern)
	for path in paths:
		if os.path.exists(path):
			for file in os.listdir(path):
				if re.match(regex, file):
					candidates.append(file)
	return candidates

def get_cuda_libpath(*subdirs):
	#tries to find all CUDA related directories. All found paths will also have subdir appeneded to them

	paths = []

	#Windows/Linux
	path = get_cuda_dir()
	if path:
		for subdir in subdirs:
			paths.append(os.path.join(path, subdir))

	#Linux
	if sys.platform == 'linux':
		#default install location
		paths.append('/usr/local/cuda')
		for subdir in subdirs:
			paths.append(os.path.join('/usr/local/cuda', subdir))
		environ = os.environ.get("LD_LIBRARY_PATH", '')
		if environ:
			for path in environ.split(':'):
				if 'cuda' in path:
					paths.append(path)
					for subdir in subdirs:
						paths.append(os.path.join(path, subdir))

	if not paths:
		raise CudaSupportError("CUDA toolkit could not be found")

	return paths

def get_cuda_pathvar():
	paths = []
	for path in os.environ['PATH'].split(':'):
		if 'cuda' in path:
			paths.append(path)
	return paths

def get_cuda_dir():
	cuda_dir = os.environ.get('CUDA_HOME', '')
	if not cuda_dir:
		cuda_dir = os.environ.get('CUDA_PATH', '')
		#TO DO
			#look for version specific cuda paths
		# if not cuda_dir:
			# cuda_dir= os.environ.get('CUDA_PATH_V11_0', '')
	return cuda_dir

def get_file_candidates(paths, names):
	candidates = names + [os.path.join(path, name) for path, name in product(paths, names)]

	paths = []
	path_not_exist = []
	for path in candidates:
		if os.path.isfile(path):
			paths.append(path)
		else:
			#currently does nothing
			path_not_exist.append(path)
	return paths

def find_lib(loader, paths, names):
	candidates = get_file_candidates(paths, names)

	driver_load_error = []
	for path in candidates:
		try:
			dll = loader(path)
		# Problem opening the DLL
		except OSError as e:
			driver_load_error.append(e)
			# path_not_exist.append(not os.path.isfile(path))
		else:
			return dll

	raise CudaSupportError("CUDA driver and/or runtime library cannot be found")

def find_file_from_pattern(pat, dirs):
	if isinstance(dirs, str):
		dirs = [dirs]

	paths = []
	file_not_found = []
	for d in dirs:
		try:
			files = os.listdir(d)
			candidates = [os.path.join(d, file) for file in files if pat.match(file)]
			paths.extend([c for c in candidates if os.path.isfile(c)])
		except FileNotFoundError:
			#currently does nothing
			file_not_found.append(d)

	return paths
