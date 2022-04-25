from .config import auto_driver_init, auto_import_core

if auto_import_core:
	from .core import *

if auto_driver_init:
	from .autoinit import *
