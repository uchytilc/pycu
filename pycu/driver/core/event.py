from pycu.driver import (event_create, event_destroy, event_ellapsed_time as _event_ellapsed_time,
						 event_query, event_record, event_synchronize)
import weakref

class Event:
	def __init__(self, flags = 0, *, handle = None, auto_free = True):
		if handle is None:
			handle = event_create(flags)
		if auto_free:
			weakref.finalize(self, event_destroy, handle)

		self.handle = handle

	def __repr__(self):
		return f"Event() <{int(self)}>"

	def __int__(self):
		return self.handle.value

	def __index__(self):
		return int(self)

	def record(self, stream = 0):
		event_record(self.handle, stream.handle if stream else 0)

	def ellapsed_time(self, end):
		return event_ellapsed_time(self, end)

	def query(self):
		return event_query(self.handle)

	def synchronize(self):
		event_synchronize(self.handle)

def event(flags = 0):
	return Event(flags)

def event_ellapsed_time(start, end):
	return _event_ellapsed_time(start.handle, end.handle)
