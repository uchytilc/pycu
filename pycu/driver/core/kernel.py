from pycu.driver import launch_kernel
from .preparer import preparer

import numpy as np
import ctypes

def check_dim(dim, name):
	if not isinstance(dim, (tuple, list)):
		dim = [dim]
	else:
		dim = list(dim)

	if len(dim) > 3:
		raise ValueError('%s must be a sequence of 1, 2 or 3 integers, '
						 'got %r' % (name, dim))
	while len(dim) < 3:
		dim.append(1)
	return dim

def normalize_griddim_blockdim(griddim, blockdim):
	if griddim is None or blockdim is None:
		return griddim, blockdim

	griddim = check_dim(griddim, 'griddim')
	blockdim = check_dim(blockdim, 'blockdim')

	return griddim, blockdim

class Kernel:
	def __init__(self, handle, prepare = True, griddim = None, blockdim = None, stream = 0, sharedmem = 0):
		self.handle = handle
		#If a signature is provided, the inputs are parsed and converted
		#to the proper type for the kernel call. If the signature is not
		#provided it is left up to the user to pass in the correct input
		#arguments in the correct types
		self.prepare = prepare
		self.configure(griddim, blockdim, stream, sharedmem)

	def configure(self, griddim, blockdim, stream = 0, sharedmem = 0):
		self.griddim, self.blockdim = normalize_griddim_blockdim(griddim, blockdim)
		self.stream = stream #stream.handle is isinstance(stream, Stream) else stream
		self.sharedmem = sharedmem

	def launch(self, *args):
		launch_kernel(self.handle, self.griddim, self.blockdim, args, self.sharedmem, self.stream.handle if self.stream else 0)

	def __repr__(self):
		return f"Kernel({self.griddim}, {self.blockdim}) <{int(self)}>" #stream, sharedmem

	def __int__(self):
		return self.handle.value

	def __index__(self):
		return int(self)

	# # #<<
	# def __lshift__(self, config):
	# 	return self[config]

	# #>>
	# def __rshift__(self, args):
		# #call kernel
		# if isinstance(args, (tuple, list)):
		# 	return self(*args)
		# return self(args)

	def __getitem__(self, config):
		#set kernel configuration
		if len(config) not in [2, 3, 4]:
			raise ValueError('must specify at least the griddim and blockdim')
		return Kernel(self.handle, self.prepare, *config)
		# self.configure(*config)
		# return self

	def __call__(self, *args):
		if not self.griddim or not self.blockdim:
			raise ValueError("The kernel's griddim and blockdim must be set before the kernel can be called.")
		if self.prepare:
			args = preparer(args)
		self.launch(*args)
		# if self.preparer._keep_alive:
			#pass
			#this clears the temporary values stored for the length of the kernel to pass back to the user

		# if self.prepare:
			# #TODO
				# need to move temorary pycu array back into input numpy array
			# preparer.clear()
