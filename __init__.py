# from pycu.core import *
# from pycu.core import __global__, __device__, __nglobal__, __ndevice__ , __cglobal__, __cdevice__

#TO DO
	#Abstract Numba Device Functions
	#declare Numba functions used for kernel variables within pycy
		#cuda.grid -> pycu.grid
		#cuda.threadIdx.x -> pycu.threadIdx.x
		#...
	#maybe edit C device/kerne functions slightly



#imports the contents of core into the pycu namespace
auto_import_core = True
if auto_import_core:
	from .core import *

	# #note: pycu needs to first import all stubs from cudafuncs into the pycu namespace (done in above import) before cudaimpl can be imported
	# from .core.numba_extension import cudaimpl
