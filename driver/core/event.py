from pycu.driver import (event_create, event_destroy, event_ellapsed_time,
						 event_query, event_record, event_synchronize)
import weakref


class Event:
	pass



def event_create(flags = 0):
	event = CUvevent()
	err = cuEventCreate(byref(event), flags)
	check_driver_error_either(err, "cuEventCreate error")

	return event

def event_destroy(event):
	err = cuEventDestroy(event)
	check_driver_error_either(err, "cuEventDestroy error")

def event_ellapsed_time(start, end):
	time = c_float()
	err = cuEventElapsedTime(byref(time), start, end)
	check_driver_error_either(err, "cuEventElapsedTime error")

	return time.value

def event_query(event):
	err = cuEventQuery(event)
	if err and err != CUDA_ERROR_NOT_READY:
		check_driver_error_either(err, "cuStreamQuery error")

def event_record(event, stream = 0):
	err = cuEventRecord(event, stream)
	check_driver_error_either(err, "cuEventRecord error")

def event_synchronize(event):
	err = cuEventSynchronize(event)
	check_driver_error_either(err, "cuEventSynchronize error")

