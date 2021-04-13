import os
from itertools import product
import re
import hashlib

def open_file(path, mode = 'r'):
	with open(path, mode) as source:
		return source.read()

def encode(data, typ = "utf8"):
	return data.encode(typ)

def generate_hash(string):
	if not isinstance(string, bytes):
		string = string.encode()

	m = hashlib.md5() #sha256
	m.update(string)
	return m.hexdigest() #digest()

def determine_file_name(paths, pattern):
	candidates = []
	regex = re.compile(pattern)
	for path in paths:
		if os.path.exists(path):
			for file in os.listdir(path):
				if re.match(regex, file):
					candidates.append(file)
	return candidates

def get_cuda_libvar(subdir = ''):
	paths = []

	#Windows/Linux
	path = os.environ.get("CUDA_PATH", '')
	paths.append(os.path.join(path, subdir))

	#Linux
	environ = os.environ.get("LD_LIBRARY_PATH", None)
	if environ:
		for path in environ.split(':'):
			if 'cuda' in path:
				paths.append(os.path.join(path, subdir))
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
	return cuda_dir


class CudaSupportError(ImportError):
	pass

def find_lib(loader, paths, names):
	# if envpath is not None:
		# try:
		# 	envpath = os.path.abspath(envpath)
		# except ValueError:
		# 	raise ValueError("NUMBA_CUDA_DRIVER %s is not a valid path" %
		# 					 envpath)
		# if not os.path.isfile(envpath):
		# 	raise ValueError("NUMBA_CUDA_DRIVER %s is not a valid file "
		# 					 "path.  Note it must be a filepath of the .so/"
		# 					 ".dll/.dylib or the driver" % envpath)
		# candidates = [envpath]

	# First search for the name in the default library path.
	# If that is not found, try the specific path.

	#if a name is a regular expression
		#check if path exists
		#grab all files that match regular expression


	candidates = names + [os.path.join(path, name) for path, name in product(paths, names)]

	# Load the driver; Collect driver error information
	path_not_exist = []
	driver_load_error = []

	for path in candidates:
		try:
			dll = loader(path)
		# Problem opening the DLL
		except OSError as e:
			path_not_exist.append(not os.path.isfile(path))
			driver_load_error.append(e)
		else:
			return dll

	# Problem loading driver
	if all(path_not_exist):
		# print(path_not_exist)
		raise CudaSupportError("CUDA driver and/or runtime library cannot be found")
	# else:
		# errmsg = '\n'.join(str(e) for e in driver_load_error)
		# _raise_driver_error(errmsg)
