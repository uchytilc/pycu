from pycu.driver import stream_create, stream_destroy, stream_synchronize, stream_query

import weakref

class Stream:
	def __init__(self, *, handle = None, auto_free = True):
		if handle is None:
			handle = stream_create()
		if auto_free:
			weakref.finalize(self, stream_destroy, handle)

		self.handle = handle

	def __repr__(self):
		return f"Stream() <{int(self)}>"

	def __int__(self):
		return self.handle.value

	def __index__(self):
		return int(self)

	def synchronize(self):
		stream_synchronize(self.handle)

	def get_attribute(self):
		# stream_get_attribute(stream, attr)
		pass

	def set_attributes(self):
		# stream_set_attributes(stream, attr, value)
		pass

	def query(self):
		return stream_query(self.handle)

	# def flags(self):
		# return stream_get_flags(stream)

	# def add_callback(self, callback, arg, flags = 0):
		# data = (self, callback, arg)
		# _py_incref(data)
		# stream_add_callback(self.handle, stream_callback, data, flags)

	# def get_ctx(self):
		# #import context manager
		# ctx = get_context(stream_get_ctx(self.handle))
		# return ctx

def stream():
	return Stream()
