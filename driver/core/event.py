from pycu.driver import (event_create, event_destroy, event_ellapsed_time as _event_ellapsed_time,
						 event_query, event_record, event_synchronize)
import weakref

class EventPtr:
	def __init__(self, handle):
		self.handle = handle

	def record(self, stream = 0):
		event_record(self.handle, stream.handle if stream else 0)

	def ellapsed_time(self, end):
		return event_ellapsed_time(self, end)

	def query(self):
		return event_query(self.handle)

	def synchronize(self):
		event_synchronize(self.handle)

class Event(EventPtr):
	def __init__(self, flags = 0):
		handle = event_create(flags)
		weakref.finalize(self, event_destroy, handle)

		super().__init__(handle)

def event(flags = 0):
	return Event(flags)

def event_ellapsed_time(start, end):
	return _event_ellapsed_time(start.handle, end.handle)
