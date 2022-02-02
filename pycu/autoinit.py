from .config import auto_import_core, auto_context_init

from .driver import init

init()

if auto_import_core and auto_context_init:
	from . import context_manager
	context_manager.set_current(context_manager.retain_primary(0))
