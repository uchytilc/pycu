from .context import *
from .graphics import *
from .kernel import *
from .memory import *
from .module import *
from .event import *
from .stream import *

from pycu.driver import init, CUhostFn, launch_host_func
from ctypes import c_int, py_object, CFUNCTYPE, pythonapi

_py_decref = pythonapi.Py_DecRef
_py_decref.argtypes = [py_object]

def callback(py_callback):
	def wrapper(data):
		try:
			callback, arg = data
			callback(*arg)
		except Exception as e:
			warnings.warn(f"Exception in stream callback: {e}")
		finally:
			_py_decref(data)

	callback = CUhostFn(wrapper)
	callback.py_callback = py_callback
	return callback

_py_incref = pythonapi.Py_IncRef
_py_incref.argtypes = [py_object]

def host_function_callback(stream, callback, data):
	if not isinstance(callback, CUhostFn):
		raise TypeError('callback functions must be of type CUhostFn')
	if getattr(callback, "py_callback", None) is None:
		raise AttributeError('callback functions must be decorated with the `callback` decorator')

	data = (callback.py_callback, data)
	_py_incref(data)
	launch_host_func(stream.handle, callback, data)
