import ctypes

def is_class(cls):
	try:
		return issubclass(cls, cls)
	except TypeError:
		return False

def flatten_c_stuct(struct):
	# return [typ(getattr(struct, member)) for member, typ in struct._fields_]

	flattened_stuct = []
	for member, typ in struct._fields_:
		val = getattr(struct, member)
		if isinstance(val, ctypes.Structure):
			flattened_stuct.extend(flatten_c_stuct(val))
		else:
			flattened_stuct.append(val)
	return flattened_stuct

def flatten_ctype(ctype):
	if isinstance(ctype, ctypes.Structure):
		return flatten_c_stuct(ctype)
	else:
		return ctype

class Preparer:
	#expand ctype structs when passing arguments to kernels
	flatten = False
	type_mappings = {}

	# def __init__(self, keep_alive):
		# self._keep_alive = []

	def __call__(self, args, sig = []):
		args = list(args)

		#if no signature is provided use the input args to construct the signature (so no type checking is done)
		if not sig:
			sig = [None] + [type(arg) for arg in args]
		# if len(args) != len(sig[1:]):
		# 	raise ValueError("The number of args provided does not match the signature length.")

		for n in range(len(args)):
			sig[n + 1] = self.get_mapping(sig[n + 1]) 

		for n in range(len(args)):
			arg = args.pop(0)
			if not isinstance(arg, (ctypes._SimpleCData, ctypes.Array, ctypes._Pointer, ctypes.Union, ctypes.Structure)):
				pycu_type = sig[n + 1]
				if is_class(pycu_type):
					pycu_type = pycu_type.construct(arg)
				c_arg = pycu_type.as_ctypes()(arg)
			else:
				c_arg = arg

			#convert struct into a list of its members
			if self.flatten:
				c_arg = flatten_ctype(c_arg)
			if not isinstance(c_arg, (list, tuple)):
				c_arg = [c_arg]
			args.extend(c_arg)

		return args

	@staticmethod
	def get_mapping(typ):
		mapping = Preparer.type_mappings.get(typ, None)
		if mapping is None:
			raise TypeError(f"{typ} is not supported")
		return mapping

	@staticmethod
	def add_mapping(typ, mapping):
		# if not isinstance(typ, type):
		# 	raise TypeError("Input type must be an instance of `type`")
		Preparer.type_mappings[typ] = mapping

	@staticmethod
	def remove_mapping(typ):
		return Preparer.type_mappings.pop(typ)

	def clear(self):
		#self._keep_alive.clear()
		pass

preparer = Preparer()



# #TO DO
# 	#allow a preparer to create an object to hold until the kernel has finished
# 		#This would allow numpy arrays to be given as input
# 			#simply create a buffer, hold it for the length of the kernel, write back into the numpy array, and destroy the device array when the kernel ends




# #type determines which ctype/constructor to get
# #arg is passed into this constructor to generate the ctypes object that is passed to the kernel

# #for classes that are not fully determined until initialized there needs to be a fallback
# 	#i.e. for ndarrays the dtype and ndims are not specific to the class, these are only known by the instance

# #I can have each pycu class have a __pycu_preparer_type__ function
# 	#the prepare first checks if the object has a __pycu_preparer_type__ method
# 		#if yes
# 			#use this to check the preparer dict
# 		#if no
# 			#get objects type via type()
# 			#use this to get a custom type instance

# # def __pycu_preparer_type__
# # 	return self


