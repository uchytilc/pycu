from pycu.driver import device_primary_ctx_retain, device_primary_ctx_release, ctx_set_current, ctx_synchronize, ctx_create, ctx_destroy, ctx_push_current, ctx_pop_current

import threading
import weakref
import atexit
from collections import defaultdict

class _ContextStack(threading.local):
	def __init__(self):
		super().__init__()
		self._stack = []

	def __bool__(self):
		return bool(self._stack)

	def push(self, ctx):
		self._stack.append(ctx)

	def pop(self):
		if self._stack:
			return self._stack.pop()
		return None

	def current(self):
		if not self._stack:
			return None
		return self._stack[-1]

class _PrimaryContextDict(defaultdict):
	#allow for lambda function as default_factory for defaultdict
	def __missing__(self, key):
		if self.default_factory is None:
			raise KeyError(key)
		else:
			ret = self[key] = self.default_factory(key)
			return ret

_context_lock = threading.Lock()

class ContextManager:
	#note: This class should not be mixed with direct calls to ctx_push_current, ctx_pop_current, ctx_set_current or any other driver API calls that alter the context stack

	__singleton = None

	#note: each thread has its own unique (non-primary) context stack within CUDA
	__contexts = _ContextStack()
	__primary = _PrimaryContextDict(lambda dev: PrimaryContext(dev))

	def __new__(cls, v2 = True):
		with _context_lock:
			if cls.__singleton is None:
				cls.__singleton = inst = object.__new__(cls)
		return cls.__singleton

	def __repr__(self):
		return f"ContextManager()"

	def create_context(self, dev, flags = 0):
		return Context(dev, flags)

	def retain_primary(self, dev, flags = 0):
		pctx = self.__primary[dev]
		if flags:
			pctx.set_flags(flags)
		return pctx

	def get_current(self):
		#ctx_get_current/cuCtxGetCurrent
		return self.__contexts.current()

	def set_current(self, ctx = None):
		old_ctx = self.__contexts.pop()
		if not ctx:
			ctx_set_current(ctx)
		else:
			ctx_set_current(ctx.handle)
			self.__contexts.push(ctx)
		return old_ctx

	def push(self, ctx):
		ctx_push_current(ctx.handle)
		self.__contexts.push(ctx)

	def pop(self):
		ctx_pop_current()
		ctx = self.__contexts.pop()
		return ctx

	def test(self):
		print(self.__contexts._stack)

	# def get_device(self):
		# # return devices.get()
		# # ctx_get_device()
		# pass

	def stream_priority_range(self):
		return ctx_get_stream_priority_range()

	def get_cache_config(self):
		return ctx_get_cache_config()

	def set_cache_config(self, config):
		ctx_set_cache_config(config)

	def reset_persisting_l2_cache(self):
		ctx_reset_persisting_l2_cache()

	def get_limit(self, limit):
		return ctx_get_limit(limit)

	def set_limit(self, limit, value):
		ctx_set_limit(limit, value)

	def get_shared_mem_config(self):
		return ctx_get_shared_mem_config()

	def set_shared_mem_config(self, config):
		return ctx_set_shared_mem_config(config)

	def get_flags(self):
		return ctx_get_flags()

	def add_module(self, module):
		#This simply adds the module to a list to prevent it from being garbage collected
		current = self.get_current()
		if current:
			current.add_module(module)

	def synchronize(self):
		current = self.get_current()
		if current:
			current.synchronize()

def _make_current_ctx(func):
	#note: makes a context current for length of function call. The
	#current context should not be changed within the scope of the
	#function or the Context Manager and the CUDA context stack will
	#become out of sync.
	def wrapper(ctx, *args, **kwargs):
		# ctx_push_current(ctx.handle)
		# func(*args, **kwargs)
		# ctx_pop_current()

		not_current = ctx_get_current() != ctx.handle
		if not_current:
			ctx_push_current(ctx.handle)
		func(*args, **kwargs)
		if not_current:
			ctx_pop_current()
	return wrapper

class _ContextBase:
	# lock = threading.Lock()
	def __init__(self, handle):
		self.handle = handle

		#note: optional storage dict to prevent garbage collecetion of modules. The
		#storage is tied to the context to help reflect ownership of module (although there
		#is nothing preventing a module not owned by the context from being added)
		self.modules = {}

	def __int__(self):
		return self.handle.value

	def __index__(self):
		return int(self)

	def add_module(self, module):
		# self.lock.acquire()
		# try:
		# 	self.modules[module] = module
		# finally:
		# 	self.lock.release()
		self.modules[module] = module

	def synchronize(self):
		ctx_synchronize()

	def api_version(self):
		return ctx_get_api_version(self.handle)

	@_make_current_ctx
	def get_flags(self):
		return ctx_get_flags()

	@_make_current_ctx
	def stream_priority_range(self):
		return ctx_get_stream_priority_range()

	@_make_current_ctx
	def get_cache_config(self):
		return ctx_get_cache_config()

	@_make_current_ctx
	def set_cache_config(self, config):
		ctx_set_cache_config(config)

	@_make_current_ctx
	def reset_persisting_l2_cache(self):
		ctx_reset_persisting_l2_cache()

	@_make_current_ctx
	def get_limit(self, limit):
		return ctx_get_limit(limit)

	@_make_current_ctx
	def set_limit(self, limit, value):
		ctx_set_limit(limit, value)

	@_make_current_ctx
	def get_shared_mem_config(self):
		return ctx_get_shared_mem_config()

	@_make_current_ctx
	def set_shared_mem_config(self, config):
		return ctx_set_shared_mem_config(config)

def _device_primary_ctx_release(dev, modules):
	# print("device_primary_ctx_release")
	# pycu.ctx_synchronize()
	modules.clear()
	device_primary_ctx_release(dev)

class PrimaryContext(_ContextBase):
	def __init__(self, dev = 0, flags = 0, *, handle = None, auto_free = True):
		if handle is None:
			handle = device_primary_ctx_retain(dev)

		super().__init__(handle)

		if auto_free:
			weakref.finalize(self, _device_primary_ctx_release, dev, self.modules)

		self.dev = dev

		if flags:
			self.set_flags(flags)

	def __repr__(self):
		return f"PrimaryContext({self.dev}) <{int(self)}>"

	@property
	def is_primary(self):
		return True

	def set_flags(self, flags):
		device_primary_ctx_set_flags_(self.dev, flags)

	def reset(self):
		# self.module.clear()
		pass
		#device_primary_ctx_reset(self.dev):

	def get_flags(self):
		device_primary_ctx_get_state(self.dev)


def _ctx_destroy(handle, modules):
	# print("ctx_destroy")
	modules.clear()
	ctx_destroy(handle)

class Context(_ContextBase):
	def __init__(self, dev = 0, flags = 0, *, handle = None, auto_free = True):
		if handle is None:
			handle = ctx_create(dev, flags)

		super().__init__(handle)

		if auto_free:
			weakref.finalize(self, ctx_destroy, handle, self.modules)

		self.dev = dev

	def __repr__(self):
		return f"Context() <{int(self)}>"

	@property
	def is_primary(self):
		return False

context_manager = ContextManager()

# def _del_context():
# 	ctx_synchronize()

# 	print("############del_context##############")

# atexit.register(_del_context)

def get_current_context():
	return context_manager.get_current()

def retain_primary_context(dev, flags = None):
	return context_manager.retain_primary(dev, flags)

def create_context(dev, flags = None):
	return context_manager.create_context(dev, flags)

def push_context(ctx):
	context_manager.push(ctx)

def pop_context(ctx):
	return context_manager.pop(ctx)

def set_context(ctx):
	return context_manager.set_current(ctx)

def add_module(module):
	context_manager.add_module(module)

def synchronize():
	context_manager.synchronize()
