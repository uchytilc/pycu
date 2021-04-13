class Config:
	pass

config = Config()

def __getattr__(name):
	try:
		return getattr(config, name)
	except:
		raise ImportError(f'"{name}" is not a valid configuration option')
