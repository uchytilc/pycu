from . import context_manager
from .jitify import Jitify
from pycu.driver.core import Kernel
from pycu.nvrtc.driver import get_libcudadevrt

#TO DO
	#re-add in Numba function calls to compile python to PTX for Python compilation support

def find_closest_supported_arch(arch):
	#TO DO
		#implement

	return 52

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

def compile_source(source, entry_points, I = [], arch = None, include_runtime = True):
	if isinstance(source, str):
		source = [source]

	entry_points, name_expressions = get_name_expressions(entry_points)
	arch = find_closest_supported_arch(arch)

	jitify = Jitify()

	nvrtc_options = {"I":I, "arch":arch,"name-expression":name_expressions}

	if include_runtime:
		jitify.add_library(get_libcudadevrt())

		# nvrtc_options.update({"relocatable-device-code":True})
		# path = "/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudadevrt.a"
		# path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\lib\\x64\\cudadevrt.lib"

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
