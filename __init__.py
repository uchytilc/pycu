#imports the contents of core into the pycu namespace
auto_import_core = True
auto_import_numba_extentions = True
if auto_import_core:
	from .core import *

	if auto_import_numba_extentions:
		from .core.numba_extension import *
		#need to import the mathfunc stubs into the pycu namespace so they are usable
		from .core.numba_extension.mathfuncs import *
		from .core.numba_extension.vector.vectorfuncs import *

#__global__, __device__, __nglobal__, __ndevice__ , __cglobal__, __cdevice__
