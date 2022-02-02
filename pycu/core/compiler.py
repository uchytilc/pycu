from . import context_manager
from .jitify import jitify

from ..driver.core import Kernel
from ..nvrtc.driver import get_libcudadevrt
from ..utils import open_file

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
	elif isinstance(entry_points, str):
		entry_points = [entry_points]

	name_expressions = []
	for _ in range(len(entry_points)):
		entry = entry_points.pop(0)
		if is_name_expression(entry):
			name_expressions.append(entry)
		else:
			entry_points.append(entry)
	return entry_points, name_expressions

def compile_source(sources, entry_points, nvrtc_options = {}, libcudadert = False):
	#create copy so that the contents can be altered
	nvrtc_options_updated = nvrtc_options.copy()

	entry_points, name_expressions = get_name_expressions(entry_points)
	nvrtc_options_updated.update({"arch":find_closest_supported_arch(nvrtc_options.get('arch', None))})
	nvrtc_options_updated.update({"default-device":nvrtc_options.get("default-device", True)})

	if name_expressions:
		nvrtc_options_updated.update({"name-expression":name_expressions})

	if isinstance(sources, str):
		sources = [sources]

	# jitify.clear()

	# if len(source) > 1:
		# nvrtc_options_updated.update({"dc":True})
	if libcudadert:
		jitify.add_library(get_libcudadevrt())

	for source in sources:
		lowered_names = jitify.add_source(source, nvrtc_options = nvrtc_options_updated)
		entry_points.extend(lowered_names)

	module = jitify.compile()

	#note: save module to prevent it from being garbage collected
	context_manager.add_module(module)

	kernels = []
	for entry in entry_points:
		kernels.append(Kernel(module.get_function(entry)))

	if len(kernels) == 1:
		kernels = kernels[0]
	return kernels

def compile_file(path, *args, **kwargs):
	return compile_source(open_file(path), *args, **kwargs)
