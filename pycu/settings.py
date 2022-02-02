#automatically initializes cuda by calling cuInit()
auto_driver_init = True

#automatically imports all core functionality
auto_import_core = True

#automatically initializes a primary context on device 0 and sets it current
#auto_drivier_init and auto_import_core must be true for this to take effect
auto_context_init = True

#automatically imports numba and and adds all numba extensions within PyCu
#auto_import_core must be true for this to take effect
auto_import_numba_extentions = True


# class PyCuSettings():
# 	auto_import_core = True
# 	auto_import_numba_extentions = True
# 	auto_driver_init = True
# 	auto_context_init = True

