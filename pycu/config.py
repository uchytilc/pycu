#automatically initializes cuda by calling cuInit()
auto_driver_init = False

#automatically imports all core functionality into the pycu namespace
auto_import_core = True

#automatically initializes a primary context on device 0 and sets it current
#auto_driver_init and auto_import_core must be true for this boolean to take effect
#if autoinit is called only auto_import_core must be true for this boolean to take effect
auto_context_init = True

#imports numba and and adds all numba extensions within PyCu
#auto_import_core must be true for this to take effect
include_numba_extentions = True





# class Config:
# 	pass

# config = Config()

# def __getattr__(name):
# 	try:
# 		return getattr(config, name)
# 	except:
# 		raise ImportError(f'"{name}" is not a valid configuration option')
