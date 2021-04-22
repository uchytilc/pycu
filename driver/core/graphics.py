from pycu.driver import (graphics_map_resources, graphics_unmap_resources,
						 graphics_gl_register_buffer, graphics_unregister_resource,
						 graphics_resource_get_mapped_pointer)
import weakref

#Note: unregister should be called before the OpenGL context in which the graphics resource is bound to is desroyed. If it is not the OpenGL context may be destroyed by the time the graphics resource is garbage collected

def unregister(handle):
	#check if the handle is None to prevent calling unregister on an already unregistered handle
	if handle:
		graphics_unregister_resource(handle)

# class GraphicResources:
	# #list of GraphicResource objects
	# def __init__(self, graphic_resources):
	# 	self.graphic_resources = graphic_resources
	# 	self.count = len(graphic_resources)

	# 	# self.handle = 

	# def push(self, graphic_resource)
	# 	self.graphic_resources.append(graphic_resource)

	# def pop(self, idx = -1)
	# 	return self.graphic_resources.pop(idx)

	# # def map(self, stream = 0):
	# # 	graphics_map_resources(self.count, self.handle, stream)

	# # def unmap(self, stream = 0):
	# # 	graphics_unmap_resources(self.count, self.handle, stream)

class GraphicResource:
	def __init__(self, handle = None, size = 0):
		self.handle = handle
		self.size = size

	def __repr__(self):
		return f"GraphicResource() <{self.handle.value}>"

	@property
	def mapped_ptr(self):
		if self.handle is None:
			raise ValueError("GraphicResource has no registered buffer")
		devptr, size = graphics_resource_get_mapped_pointer(self.handle)
		self.size = size

		return devptr
		# return self.get_mapped_pointer()

	def get_mapped_pointer(self):
		if self.handle is None:
			raise ValueError("GraphicResource has no registered buffer")
		devptr, size = graphics_resource_get_mapped_pointer(self.handle)
		self.size = size

		return devptr, size

	def get_mapped_pointer_size(self):
		# #get mapped pointer to update size
		# self.get_mapped_pointer()
		return self.size

	def map(self, stream = 0):
		# if self.handle is None:
		# 	raise ValueError("GraphicResource has no registered buffer")
		graphics_map_resources(1, self.handle, stream)

	def unmap(self, stream = 0):
		# if self.handle is None:
		# 	raise ValueError("GraphicResource has no registered buffer")
		graphics_unmap_resources(1, self.handle, stream)

	def unregister(self):
		# if self.handle is None:
		# 	raise ValueError("GraphicResource has no registered buffer")
		graphics_unregister_resource(self.handle)
		self.handle = None

	# CUresult cuGraphicsResourceSetMapFlags ( CUgraphicsResource resource, unsigned int  flags )
	#     Set usage flags for mapping a graphics resource. 
	# CUresult cuGraphicsResourceGetMappedMipmappedArray ( CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource )
	#     Get a mipmapped array through which to access a mapped graphics resource. 
	# CUresult cuGraphicsSubResourceGetMappedArray ( CUarray* pArray, CUgraphicsResource resource, unsigned int  arrayIndex, unsigned int  mipLevel )
	#     Get an array through which to access a subresource of a mapped graphics resource. 

class OpenGLGraphicsResource(GraphicResource):
	def __init__(self, handle = None):
		super().__init__()

		self.handle = handle

	def __repr__(self):
		return f"OpenGLGraphicsResource() <{self.handle.value}>"

	def __long__(self):
		return self.handle.value

	def register_buffer(self, buff, flags = 0):
		# if self.handle is not None:
		# 	raise ValueError("GraphicResource already has a registered buffer")
		self.handle = graphics_gl_register_buffer(buff, flags)

	def get_devices(self):
		# CUresult cuGLGetDevices ( unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int  cudaDeviceCount, CUGLDeviceList deviceList )
		pass

	# CUresult cuGraphicsGLRegisterImage ( CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int  Flags )
	#     Register an OpenGL texture or renderbuffer object. 
	# CUresult cuWGLGetDevice ( CUdevice* pDevice, HGPUNV hGpu )
	#     Gets the CUDA device associated with hGpu. 

# class OpenGLGraphicsResource(_OpenGLGraphicsResource):
# 	def __init__(self, buff = None, flags = 0):
# 		super().__init__()

# 		self.handle = None
# 		# if buff is not None:
# 		# 	self.handle = handle = self.register_buffer(buff, flags)
# 		# 	weakref.finalize(self, unregister, handle)
# 		# 	#cuGraphicsUnregisterResource(CUgraphicsResource resource)

class VDPAUGraphicsResource(GraphicResource):
	pass

class EGLGraphicResource(GraphicResource):
	pass



def opengl_resource():
	return OpenGLGraphicsResource()