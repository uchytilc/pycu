import ctypes

def is_class(cls):
	try:
		return issubclass(cls, cls)
	except TypeError:
		return False

def flatten_stuct(struct):
	return [typ(getattr(struct, member)) for member, typ in struct._fields_]

def flatten_ctype(ctype):
	if isinstance(ctype, ctypes.Structure):
		return flatten_stuct(ctype)
	else:
		return ctype

class Preparer:
	#expand ctype structs when passing arguments to kernels
	flatten = True
	# prepare = True
	type_mappings = {}

	# def __init__(self, keep_alive):
		# self._keep_alive = []

	def __call__(self, args, sig = []):
		# if not self.prepare:
			# return args

		args = list(args)
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





# #TO DO
# 	#allow a preparer to create an object to hold until the kernel has finished
# 		#This would allow numpy arrays to be given as input
# 			#simply create a buffer, hold it for the length of the kernel, write back into the numpy array, and destroy the device array when the kernel ends

# # class CtypesConverter:
# # 	type_mapping = {}

# # 	@staticmethod
# # 	def get_converter(typ):
# # 		# #objects can use a seperate class to represent their type (e.g Numba's Type objects). This checks if
# # 		# #the input type is an instanced class, which means the input `typ` is an instanced type object which
# # 		# #represents a different class.
# # 		# if not isinstance(typ, type):
# # 		# 	typ = type(typ)
# # 		return CtypesConverter.type_mapping[typ]

# # 	@staticmethod
# # 	def add_converter(typ, converter):
# # 		# if not isinstance(typ, type):
# # 		# 	raise TypeError("Input type must be an instance of `type`")
# # 		CtypesConverter.type_mapping[typ] = converter

# # 	@staticmethod
# # 	def remove_converter(typ):
# # 		return CtypesConverter.type_mapping.pop(typ)

# def scalar_constructor(ctype, arg):
# 	return ctype(arg)

# def struct_constructor(ctype, arg):
# 	struct = ctype()
# 	for member, typ in struct._fields_:
# 		if issubclass(typ, ctypes.Structure):
# 			setattr(struct, member, struct_constructor(typ, getattr(arg, member)))
# 		else:
# 			setattr(struct, member, getattr(arg, member))
# 	return struct


# def ctypes_ctype(ctype):
# 	return ctype, scalar_constructor

# def bool_ctype():
# 	return ctypes.c_bool, scalar_constructor

# def int_ctype():
# 	return ctypes.c_int, scalar_constructor

# def float_ctype():
# 	return ctypes.c_float, scalar_constructor

# def numpy_ctype(nptype):
# 	return np.ctypeslib.as_ctypes_type(nptype), scalar_constructor


# def expand_stuct(struct):
# 	return [typ(getattr(struct, member)) for member, typ in struct._fields_]

# def expand_ctype(ctype):
# 	if isinstance(ctype, ctypes.Structure):
# 		return expand_stuct(ctype)
# 	else:
# 		return ctype

# #the signature type might not match the input type
# 	#this is ok
# 		#ex. arg: numpy array, sig: numba ndarray
# 	#signature can either be a type or an instance which represents a type
# 		#if type
# 			#dict[type]
# 		#if instance
# 			#dict[type(instance)]

# class ArgPreparer:
# 	prepare = True
# 	#expand ctype structs when passing arguments to kernels
# 	expand = True

# 	arg_mapping = {}

# 	def __call__(self, args):
# 		if not self.prepare:
# 			return args

# 		#first argument is the return type
# 		sig = [None] + [type(arg) for arg in args]
# 		args = list(args)

# 		# if len(args) != len(sig[1:]):
# 		# 	raise ValueError("The number of args provided does not match the signature length.")

# 		for n in range(len(args)):
# 			arg = args.pop(0)
# 			typ = sig[n + 1]

# 			arg_mapping = ArgPreparer.arg_mapping.get(typ, None)(arg)
# 			if arg_mapping is None:
# 				unsupported_arg(arg)
# 			ctype, constructor = arg_mapping
# 			constructed = constructor(ctype, arg)

# 			#convert constructed object into a list of its members
# 			if self.expand:
# 				constructed = expand_ctype(constructed)

# 			if isinstance(constructed, (list, tuple)):
# 				args.extend(constructed)
# 			else:
# 				args.append(constructed)

# 		return args

# 	@staticmethod
# 	def add_preparer(typ, preparer):
# 		# if not isinstance(typ, type):
# 		# 	raise TypeError("Input type must be an instance of `type`")
# 		ArgPreparer.arg_mapping[typ] = preparer

# 	@staticmethod
# 	def remove_preparer(typ):
# 		return ArgPreparer.arg_mapping.pop(typ)

# arg_preparer = ArgPreparer()

# def unsupported_arg(arg):
# 	raise TypeError(f"{type(arg)} is currently not supported")

# def ctypes_arg(arg):
# 	return ctypes_ctype(arg)

# def numpy_arg(arg):
# 	return numpy_ctype(arg)

# def bool_arg(arg):
# 	return bool_ctype()

# def int_arg(arg):
# 	return int_ctype()

# def float_arg(arg):
# 	return float_ctype()

# arg_preparer.add_preparer(bool, bool_arg)
# arg_preparer.add_preparer(int, int_arg)
# arg_preparer.add_preparer(float, float_arg)
# #TO DO
# arg_preparer.add_preparer(list, unsupported_arg)
# arg_preparer.add_preparer(tuple, unsupported_arg)
# arg_preparer.add_preparer(complex, unsupported_arg)
# arg_preparer.add_preparer(bytes, unsupported_arg)
# arg_preparer.add_preparer(str, unsupported_arg)

# types = [ctypes.c_byte, ctypes.c_ubyte, ctypes.c_char, ctypes.c_char_p,
# 		 ctypes.c_double, ctypes.c_longdouble, ctypes.c_float,
# 		 ctypes.c_int, ctypes.c_int8, ctypes.c_int16, ctypes.c_int32, ctypes.c_int64,
# 		 ctypes.c_uint, ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64,
# 		 ctypes.c_long, ctypes.c_longlong, ctypes.c_short, ctypes.c_size_t, ctypes.c_ssize_t,
# 		 ctypes.c_ulong, ctypes.c_ulonglong, ctypes.c_ushort, ctypes.c_void_p, ctypes.c_bool]
# # class ctypes.c_wchar
# # class ctypes.c_wchar_p
# # class ctypes.HRESULT
# # class ctypes.py_object

# for typ in types:
# 	arg_preparer.add_preparer(typ, ctypes_arg)

# #note: some of these types are alias to other types in the list
# types = [np.byte, np.short, np.intc, np.int_, np.longlong,
# 		 np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong,
# 		 np.int8, np.int16, np.int32, np.int64,
# 		 np.uint8, np.uint16, np.uint32, np.uint64,
# 		 np.float_, np.float16, np.float32, np.float64,
# 		 np.half, np.single, np.double, np.longdouble,
# 		 np.bool_, np.uintp, np.intp]
# 		 # np.complex_, np.complex64, np.complex128
# 		 # np.str_, timedelta64, datetime64
# 		 # np.void

# for typ in types:
# 	arg_preparer.add_preparer(typ, numpy_arg)




# 			#check ctypes_converter with type given
# 				#if not present 
# 					#take arg and check type converter
# 						#if type converter has type, convert type and retry ctypes_converter
# 						#else fail


# 			#if isinstance(arg, Type):
# 				#typ = arg._numba_type_

# 			#Check if signature types are actually types or if they are instances of type classes
# 			# if isinstance(typ, type):
# 				# pass


# #get_type(arg)

# #default(arg)
# 	#return type(arg)

# #numba(arg):
# 	#return arg._numba_type_


# #numba types are instances of classes
# 	#give all numba classes _numba_type_ property
# 		#returns instance of corrisponding type class
# #pycu classes do not have type classes







# # def prepare_args(args, sig, expand):
# # 	'''
# # 	converts input args to ctypes for kernel input
# # 	if ctypes args are given they are skipped
# # 	'''

# # 	#TO DO
# # 		#Note: pointer(dtype) probably shouldn't work how it does for CuNDArrays
# # 			#pointer should grab the handle of the struct, not the device memory
# # 			#if pointer(dtype) is used, to get device memory I should explicitly pass in array.handle

# # 	c_args = []
# # 	for arg, typ in zip(args, sig[1:]):
# # 		#convert input to ctypes type
# # 		if not isinstance(arg, (_ctypes._SimpleCData, _ctypes.Array,
# # 								_ctypes._Pointer, _ctypes.Union,
# # 								_ctypes.Structure)):
# # 			#TO DO
# # 				#change this
# # 			##########################
# # 			if isinstance(typ, types.Pointer):
# # 				arg = arg.handle
# # 			##########################
# # 			else:
# # 				if isinstance(arg, (tuple, list)):
# # 					ctype = typ.as_ctypes()(*arg)
# # 				else:
# # 					ctype = typ.as_ctypes()(arg)
# # 				#expand the struct into individual values for Numba
# # 				if isinstance(ctype, _ctypes.Structure) and expand:
# # 					c_args.extend(expand_struct(ctype))
# # 				else:
# # 					c_args.append(ctype)
# # 		#already a ctypes value, do nothing
# # 		else:
# # 			c_args.append(arg)

# # 	return c_args










