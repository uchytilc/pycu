from . import context_manager
from .jitify import Jitify
from pycu.driver.core import Kernel

# from numba.cuda.cudadrv.nvvm import get_supported_ccs, find_closest_arch, get_arch_option

# def numba_nvvmir():
# 	pass

# def get_nvvmir():
# 	a = numba_nvvmir()

# 	# if not device:
# 	# 	mangled_name = cres.fndesc.mangled_name[:3] + '6cudapy' + cres.fndesc.mangled_name[3:]
# 	# else:
# 	# 	mangled_name = cres.fndesc.mangled_name
# 	# llvmir = llvmir.replace(mangled_name, n_dfunc.entry)

# 	# return llvmir






#Allow for NUMBA_SUPPORTED_CC to be an argument
	#if not provided fall back to NVVM().get_version() otherwise don't load NVVM and let user use their own



# DEFAULT_ARCH = 52

# NVVM_VERSION = NVVM().get_version()
# if NVVM_VERSION < (1, 4):
# 	# CUDA 8.0
# 	NUMBA_SUPPORTED_CC = 20, 21, 30, 35, 50, 52, 53, 60, 61, 62
# elif NVVM_VERSION < (1, 5):
# 	# CUDA 9.0 and later
# 	NUMBA_SUPPORTED_CC = 30, 35, 50, 52, 53, 60, 61, 62, 70
# else:
# 	# CUDA 10.0 and later
# 	NUMBA_SUPPORTED_CC = 30, 35, 50, 52, 53, 60, 61, 62, 70, 72, 75

def find_closest_supported_arch(mycc = None):
	return 52
	# global NUMBA_SUPPORTED_CC

	# #if arch isn't provided grab the lowest cc as defualt
	# if mycc is None:
	# 	return NUMBA_SUPPORTED_CC[0]

	# for n, cc in enumerate(NUMBA_SUPPORTED_CC):
	# 	if cc == mycc:
	# 		return cc
	# 	elif cc > mycc:
	# 		# exceeded, cc lower than supported
	# 		if n == 0:
	# 			raise RuntimeError("GPU compute capability %d is "
	# 							   "not supported (requires >=%d)" %(mycc, cc))
	# 		# return previous cc
	# 		else:
	# 			return NUMBA_SUPPORTED_CC[n - 1]

	# # cc higher than supported select highest
	# return NUMBA_SUPPORTED_CC[-1]
















def is_name_expression(entry):
	symbols = {':','&','<','>','(',')'}
	if len(set(entry) & symbols) != 0:
		return True
	return False

def get_name_expressions(entry_points):
	if isinstance(entry_points, tuple):
		entry_points = list(entry_points)
	if not isinstance(entry_points, list):
		entry_points = [entry_points]

	_entry_points = []
	name_expressions = []
	while entry_points:
		entry = entry_points.pop(0)
		if is_name_expression(entry):
			name_expressions.append(entry)
		else:
			_entry_points.append(entry)
	return _entry_points, name_expressions


#make source a list
def compile_source(source, entry_points, I = [], arch = None, include_runtime = True):
	if isinstance(source, str):
		source = [source]

	arch = 52

	entry_points, name_expressions = get_name_expressions(entry_points)
	arch = find_closest_supported_arch(arch)

	jitify = Jitify()

	nvrtc_options = {"I":I, "arch":arch,"name-expression":name_expressions}

	if include_runtime:
		nvrtc_options.update({"relocatable-device-code":True})
		path = "/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudadevrt.a"
		# path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\lib\\x64\\cudadevrt.lib"
		# jitify.add_file(path)

		with open(path, 'rb') as runtime:
			jitify.add_library(runtime.read())

	for s in source:
		lowered_names = jitify.add_source(s, nvrtc_options = nvrtc_options)
		entry_points += lowered_names

	module = jitify.compile()

	#save module to prevent it from being garbage collected
	context_manager.add_module(module)

	kernels = []
	for entry in entry_points:
		kernels.append(Kernel(module.get_function(entry)))

	if len(kernels) == 1:
		kernels = kernels[0]
	return kernels
