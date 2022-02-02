from numba import types
from numba.core.extending import typeof_impl, type_callable
from numba.core.typing import signature

from numba.core.extending import models, register_model, lower_builtin
from numba.core import cgutils
from numba.cpython import mathimpl

from numba.cuda.cudadecl import registry as decl_registry
from numba.cuda.cudaimpl import registry as impl_registry
from numba.core.typing.templates import ConcreteTemplate, AttributeTemplate

from .vectortype import (char2,  short2,  int2,  long2,
						uchar2, ushort2, uint2, ulong2,
						float2, double2,
						 char3,  short3,  int3,  long3,
						uchar3, ushort3, uint3, ulong3,
						float3, double3)
from .vectordecl import VectorType, Vector2Type, Vector3Type
from .vectordecl import (char2_t,  short2_t,  int2_t,  long2_t,
						uchar2_t, ushort2_t, uint2_t, ulong2_t,
						float2_t, double2_t,
						 char3_t,  short3_t,  int3_t,  long3_t,
						uchar3_t, ushort3_t, uint3_t, ulong3_t,
						float3_t, double3_t)

from llvmlite import ir

import operator

from .. import mathfuncs

import numpy as np

#for loop example
	#https://github.com/numba/numba/blob/7df40d2e36d15c4965b849b109c06593d6cc7b64/numba/core/cgutils.py#L487


def allocate_vector(context, builder, vec_t):
	OutProxy = cgutils.create_struct_proxy(vec_t, 'data')
	#convert numba type to ir type
	datamodel = context.data_model_manager[OutProxy._fe_type]

	out_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
	out = OutProxy(context, builder, ref = out_ptr)
	return out

def vectorize_vector(context, builder, sig, args, n):
	#returns a vectorized list of elements to operate over as well as the type of each entry
	vector = cgutils.create_struct_proxy(sig.args[n], 'data')(context, builder, ref = args[n])
	return [getattr(vector, attr) for attr in sig.args[n]._fields], sig.args[n]._member_t

def vectorize_member(context, builder, sig, args, n):
	member = args[n]
	return [member for attr in sig.return_type._fields], sig.args[n]

def vectorize_scalar_cast(cast_type):
	def vectorize_scalar(context, builder, sig, args, n):
		scalar = context.cast(builder, args[n], sig.args[n], cast_type)
		return [scalar for attr in sig.return_type._fields], cast_type
	return vectorize_scalar

##########################
#generic caller functions#
##########################

def vectorized_op_caller_wrapper(vectorized_op_helper):
	def vectorized_op_caller(context, builder, sig, *args):
		out = allocate_vector(context, builder, sig.return_type)
		return vectorized_op_helper(context, builder, sig, out, *args)
	return vectorized_op_caller

def libfunc_vectorized_op_helper_wrapper(libop):
	def libfunc_vectorized_op_helper(context, builder, sig, out, *args):
		for n, attr in enumerate(out._fe_type._fields):
			op_sig = signature(*([sig.return_type._member_t] + [arg_t for (arg, arg_t) in args]))
			op = context.get_function(libop, op_sig)
			setattr(out, attr, op(builder, [arg[n] for (arg, arg_t) in args]))
		return out._getpointer()
	return libfunc_vectorized_op_helper

def libfunc_vectorized_op_caller_wrapper(op):
	#generate op specific vectorization helper method called by the caller
	libfunc_vectorized_op_helper = libfunc_vectorized_op_helper_wrapper(op)
	#assign vectorizer helper to binary caller to operate on the inputs
	return vectorized_op_caller_wrapper(libfunc_vectorized_op_helper)

###########
#UNARY OPS#
###########

#unary_op_factory_wrapper
def unary_op_vectorized_wrapper(op, caller, vec):
	vec_t, vectorize_vector = vec

	@impl_registry.lower(op, vec_t)
	def unary_op_vectorized_vector(context, builder, sig, args):
		f = vectorize_vector(context, builder, sig, args, 0)
		return caller(context, builder, sig, f)

def unary_op_wrapper(op, caller, vec_t):
	@impl_registry.lower(op, vec_t)
	def unary_op_vectorized_vector(context, builder, sig, args):
		f = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		return caller(context, builder, sig, f)

#########################
#unary caller functions#
#########################

#######################################
#reduction caller functions#
#######################################

def libfunc_reduction_op_caller_wrapper(libop):
	def libfunc_reduction_op_helper(context, builder, sig, vec):
		members, member_t = vec

		op = context.get_function(libop, signature(member_t, member_t, member_t))

		val = members[0]
		for n, member in enumerate(members[1:]):
			val = op(builder, [val, member])
		return val
	return libfunc_reduction_op_helper

def length_reduction_op_caller(context, builder, sig, vec):
	unary_sig = signature(sig.return_type, sig.return_type)
	binary_reduction_sig = signature(sig.return_type, sig.args[0], sig.args[0])

	sqrt = context.get_function(mathfuncs.sqrt, unary_sig)
	dot = context.get_function(mathfuncs.dot, binary_reduction_sig)

	return sqrt(builder, [dot(builder, [vec._getpointer(), vec._getpointer()])])

# def reduction_builder_op_caller(op, builder, vec):
	# val = vec[0]
	# for n in range(len(vec) - 1):
	# 	val = op(val, vec[n + 1])
	# return val

# def reduction_libfunc_op_caller(op, builder, vec):
	# val = vec[0]
	# for n in range(len(vec) - 1):
	# 	val = op(builder, [val, vec[n + 1]])
	# return val

############
#BINARY OPS#
############

#binary_op_vectorized_factory
def _binary_op_vectorized_other_other_wrapper(op, caller, f, g):
	f_t, vectorize_f = f
	g_t, vectorize_g = g

	@impl_registry.lower(op, f_t, g_t)
	def binary_op_vectorized_vector_other(context, builder, sig, args):
		f = vectorize_f(context, builder, sig, args, 0)
		g = vectorize_g(context, builder, sig, args, 1)

		# if len(f) != len(g):
		# 	raise ValueError("")

		return caller(context, builder, sig, f, g)

def binary_op_vectorized_vector_vector_wrapper(op, caller, vec):
	_binary_op_vectorized_other_other_wrapper(op, caller, vec, vec)

def binary_op_vectorized_vector_other_wrapper(op, caller, vec, other):
	_binary_op_vectorized_other_other_wrapper(op, caller, vec, other)

def binary_op_vectorized_other_vector_wrapper(op, caller, vec, other):
	_binary_op_vectorized_other_other_wrapper(op, caller, other, vec)

#binary_op_vectorized_factory_wrapper
def binary_op_vectorized_wrapper(op, caller, vec, other):
	binary_op_vectorized_vector_vector_wrapper(op, caller, vec)
	binary_op_vectorized_vector_other_wrapper(op, caller, vec, other)
	binary_op_vectorized_other_vector_wrapper(op, caller, vec, other)

def binary_op_vector_vector_wrapper(op, caller, vec_t, other_t = None):
	if other_t is None:
		other_t = vec_t

	@impl_registry.lower(op, vec_t, other_t)
	def binary_op_vectorized_vector_vector(context, builder, sig, args):
		f = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		g = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
		return caller(context, builder, sig, f, g)

def binary_op_vector_other_wrapper(op, caller, vec_t, other_t):
	@impl_registry.lower(op, vec_t, other_t)
	def binary_op_vectorized_vector_other(context, builder, sig, args):
		f = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		g = args[1]
		return caller(context, builder, sig, f, g)

def binary_op_other_vector_wrapper(op, caller, vec_t, other_t):
	@impl_registry.lower(op, vec_t, other_t)
	def binary_op_vectorized_other_vector(context, builder, sig, args):
		f = args[0]
		g = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
		return caller(context, builder, sig, f, g)

##########################
#binary caller functions#
##########################

def shift_binary_op_caller(context, builder, sig, *args):
	def shift_impl(out, vec, n, member_current):
		# out.x = context.cast(builder, n, types.uint32, types.float32)
		# out.y = context.cast(builder, n, types.uint32, types.float32)
		# out.z = context.cast(builder, n, types.uint32, types.float32)

		member_names = out._fe_type._fields
		member_count = len(member_names)

		idx = context.get_constant(types.int32, np.int32(member_current))

		if member_current < member_count:
			with builder.if_else(builder.icmp_unsigned('==', n, idx)) as (true, false):
				with true:
					for m in range(member_count):
						setattr(out, member_names[(m + member_current)%member_count], vec[m])
				with false:
					shift_impl(out, vec, n, member_current + 1)
		return out

	def shift(vec, n):
		out = allocate_vector(context, builder, sig.return_type)

		modulo = context.get_function(operator.mod, signature(types.int32, types.int32, types.int32))

		n = context.cast(builder, n, sig.args[1], types.int32)
		length = context.get_constant(types.int32, len(out._fe_type._fields))

		n = modulo(builder, [n, length])

		out = shift_impl(out, vec, n, 0)

		return out._getpointer()
	return shift(*args)

def dot_binary_op_caller(context, builder, sig, *args):
	def dot(f, g):
		mul = context.get_function(operator.mul, signature(sig.return_type, sig.args[0]._member_t, sig.args[1]._member_t))
		#after the multiply the result will be the same type as the return type
		add = context.get_function(operator.add, signature(sig.return_type, sig.return_type, sig.return_type))

		val = mul(builder, [getattr(f, 'x'), getattr(g, 'x')])
		for attr1, attr2 in zip(sig.args[0]._fields[1:], sig.args[1]._fields[1:]):
			val = add(builder, [val, mul(builder, [getattr(f, attr1), getattr(g, attr2)])])
		return val
	return dot(*args)

#############
#TERNARY OPS#
#############

def _ternary_op_vectorized_other_other_other_wrapper(op, caller, f, g, h):
	f_t, vectorize_f = f
	g_t, vectorize_g = g
	h_t, vectorize_h = h

	@impl_registry.lower(op, f_t, g_t, h_t)
	def binary_op_vectorized_vector_other(context, builder, sig, args):
		f = vectorize_f(context, builder, sig, args, 0)
		g = vectorize_g(context, builder, sig, args, 1)
		h = vectorize_h(context, builder, sig, args, 2)

		# if len(f) != len(g) or len(g) != len(h):
		# 	raise ValueError("")

		return caller(context, builder, sig, f, g, h)

#ternary_op_factory
def ternary_op_vectorized_vec_vec_vec_wrapper(op, caller, vec):
	_ternary_op_vectorized_other_other_other_wrapper(op, caller, vec, vec, vec)

def ternary_op_vectorized_vec_vec_other_wrapper(op, caller, vec, other):
	_ternary_op_vectorized_other_other_other_wrapper(op, caller, vec, vec, other)

def ternary_op_vectorized_vec_other_vec_wrapper(op, caller, vec, other):
	_ternary_op_vectorized_other_other_other_wrapper(op, caller, vec, other, vec)

def ternary_op_vectorized_other_vec_vec_wrapper(op, caller, vec, other):
	_ternary_op_vectorized_other_other_other_wrapper(op, caller, other, vec, vec)

def ternary_op_vectorized_vec_other_other_wrapper(op, caller, vec, other_a, other_b = None):
	if other_b is None:
		other_b = other_a

	_ternary_op_vectorized_other_other_other_wrapper(op, caller, vec, other_a, other_b)

def ternary_op_vectorized_other_vec_other_wrapper(op, caller, vec, other_a, other_b = None):
	if other_b is None:
		other_b = other_a
	_ternary_op_vectorized_other_other_other_wrapper(op, caller, other_a, vec, other_b)

def ternary_op_vectorized_other_other_vec_wrapper(op, caller, vec, other_a, other_b = None):
	if other_b is None:
		other_b = other_a
	_ternary_op_vectorized_other_other_other_wrapper(op, caller, other_a, other_b, vec)

#ternary_op_vectorized_factory_wrapper
def ternary_op_vectorized_wrapper(op, caller, vec, other):
	ternary_op_vectorized_vec_vec_vec_wrapper(op, caller, vec)
	ternary_op_vectorized_vec_vec_other_wrapper(op, caller, vec, other)
	ternary_op_vectorized_vec_other_vec_wrapper(op, caller, vec, other)
	ternary_op_vectorized_other_vec_vec_wrapper(op, caller, vec, other)
	ternary_op_vectorized_vec_other_other_wrapper(op, caller, vec, other)
	ternary_op_vectorized_other_vec_other_wrapper(op, caller, vec, other)
	ternary_op_vectorized_other_other_vec_wrapper(op, caller, vec, other)

###########################
#ternary caller functions#
###########################

##################
#VECTOR FUNCTIONS#
##################

from . import vectorfuncs
from ..mathdecl import register_op, unary_op_fixed_type_factory, binary_op_fixed_type_factory

#inject/insert functions into mathfuncs if they don't already exist (because functions aren't typed only a single instance of a function with the given name needs to exist within core)
vec_funcs = [(vectorfuncs.dot, binary_op_fixed_type_factory, None),
			 (vectorfuncs.length, unary_op_fixed_type_factory, None),
			 (vectorfuncs.shift, binary_op_fixed_type_factory, None)]
			 #(vectorfuncs.sum, unary_op_fixed_type_factory, None),

for vec_func in vec_funcs:
	func, factory, *factory_args = vec_func
	name = func.__name__
	libfunc = getattr(mathfuncs, name, None)
	#only register a vector function if a stub doesn't already exist within core
	if libfunc is None:
		setattr(mathfuncs, name, func)
		#note: None is set as default return type for some of the vec_funcs so that a typer_ method must be defined to use the op on input types
		register_op(name, factory, *factory_args)

def vec_decl(vec, vec_t, Vec, vec_typer):
	##########################################
	#Define Vector Type
	##########################################

	#note: the typer function must have the same number of arguments as inputs given to the struct so each initialize style needs its own typer

	@typeof_impl.register(vec)
	def typeof_vec(val, c):
		return vec_t

	#initialize: vec()
	@type_callable(vec)
	def type_vec(context):
		def typer():
			return vec_typer()
		return typer

	#vector length specific initialization

	#initialize: vec(x)
	@type_callable(vec)
	def type_vec(context):
		def typer(x):
			return vec_typer(x)
		return typer

	if len(vec_t._fields) >= 2:
		#initialize: vec(x,y)
		@type_callable(vec)
		def type_vec2(context):
			def typer(x,y):
				return vec_typer(x,y)
			return typer

	if len(vec_t._fields) >= 3:
		#initialize: vec(x,y,z)
		@type_callable(vec)
		def type_vec3(context):
			def typer(x,y,z):
				return vec_typer(x,y,z)
			return typer

		# #initialize: 
		# @type_callable(vec)
		# def type_vec3(context):
		# 	def typer(vec_or_input_t_a, vec_or_input_t_b):
		# 		return vec_typer(vec_or_input_t_a, vec_or_input_t_b)
		# 	return typer

	# if len(vec_t._members) >= 4:
		# pass

	@register_model(type(vec_t)) 
	class VecModel(models.StructModel):
		def __init__(self, dmm, fe_type):
			models.StructModel.__init__(self, dmm, fe_type, vec_t._members)

		def get_value_type(self):
			return super().get_value_type().as_pointer()

		# def get_argument_type(self):
			# print("get_argument_type")
			# return super().get_argument_type()

		def as_argument(self, builder, value):
			#defines how the arguments are extacted from the object and puts them into the function args
				#example:
					# %"extracted.x" = extractvalue {float, float, float} %".23", 0
					# %"extracted.y" = extractvalue {float, float, float} %".23", 1
					# %"extracted.z" = extractvalue {float, float, float} %".23", 2
			return super().as_argument(builder, builder.load(value))

		def from_argument(self, builder, value):
			#defines the argument output (the object operated on within the function) and how the input arguments get mapped on to this object.
			vec_ptr = cgutils.alloca_once(builder, self.get_data_type(), size = 1)
			for i, (dm, val) in enumerate(zip(self._models, value)):
				v = getattr(dm, "from_argument")(builder, val)
				i_ptr = cgutils.gep_inbounds(builder, vec_ptr, 0, i)
				builder.store(v, i_ptr)
			return vec_ptr

		def as_return(self, builder, value):
			# as_return START
			# %".23" = load {float, float, float}, {float, float, float}* %".15"
			# as_return END
			return super().as_return(builder, builder.load(value))

		def from_return(self, builder, value):
			vec_ptr = cgutils.alloca_once(builder, self.get_data_type(), size = 1)
			builder.store(value, vec_ptr)
			return vec_ptr

	# def get(self, builder, val, pos):
		# if isinstance(pos, str):
		# 	pos = self.get_field_position(pos)
		# return builder.extract_value(val, [pos], name="extracted." + self._fields[pos])

	# def set(self, builder, stval, val, pos):
		# if isinstance(pos, str):
		# 	pos = self.get_field_position(pos)
		# return builder.insert_value(stval, val, [pos], name="inserted." + self._fields[pos])

	##########################################
	#Initializer/Constructor Methods
	##########################################

	#initialize: vec()
	@lower_builtin(vec)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, context.get_constant(sig.return_type._member_t, 0))

		return out._getpointer()

	#initialize: vec(x)
	@lower_builtin(vec, type(vec_t._member_t))
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, context.cast(builder, args[0], sig.args[0], sig.return_type._member_t))

		return out._getpointer()

	#initialize: vec(vec)
	@lower_builtin(vec, Vec)
	def impl_vec(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, context.cast(builder, getattr(vec, attr), sig.args[0]._member_t, sig.return_type._member_t))

		return out._getpointer()

	#initialize: vec(x,y,...)
	@lower_builtin(vec, *[type(vec_t._member_t)]*len(vec_t._members))
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, context.cast(builder, args[n], sig.args[n], sig.return_type._member_t))

		return out._getpointer()

	if Vec == Vector3Type:
		#initialize: vec(x, vec2)
		@lower_builtin(vec, type(vec_t._member_t), Vector2Type)
		def impl_vec(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
			vals = [context.cast(builder, args[0], sig.args[0], sig.return_type._member_t),
					context.cast(builder, vec.x, sig.args[1]._member_t, sig.return_type._member_t),
					context.cast(builder, vec.y, sig.args[1]._member_t, sig.return_type._member_t)]

			out = allocate_vector(context, builder, sig.return_type)

			for n, attr in enumerate(sig.return_type._fields):
				setattr(out, attr, vals[n])

			return out._getpointer()

		#initialize: vec(vec2, x)
		@lower_builtin(vec, Vector2Type, type(vec_t._member_t))
		def impl_vec(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
			vals = [context.cast(builder, vec.x, sig.args[0]._member_t, sig.return_type._member_t),
					context.cast(builder, vec.y, sig.args[0]._member_t, sig.return_type._member_t),
					context.cast(builder, args[1], sig.args[1], sig.return_type._member_t)]

			out = allocate_vector(context, builder, sig.return_type)

			for n, attr in enumerate(sig.return_type._fields):
				setattr(out, attr, vals[n])

			return out._getpointer()

	##########################################
	#Define Vector Attributes
	##########################################

	@decl_registry.register_attr
	class Vec_attrs(AttributeTemplate):
		key = vec_t

		def resolve_x(self, mod):
			return vec_t._member_t

		def resolve_y(self, mod):
			return vec_t._member_t

		def resolve_z(self, mod):
			return vec_t._member_t

		# def resolve___get_item__(self, mod):
		# 	return vec_t._member_t

		#def resolve_length(self):
			#pass

	@impl_registry.lower_getattr(vec_t, 'x')
	def vec_get_x(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig, 'data')(context, builder, ref = args)
		return vec.x

	@impl_registry.lower_setattr(vec_t, 'x')
	def vec_set_x(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		x = context.cast(builder, args[1], sig.args[1], sig.args[0]._member_t)
		setattr(vec, 'x', x)

	@impl_registry.lower_getattr(vec_t, 'y')
	def vec_get_y(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig, 'data')(context, builder, ref = args)
		return vec.y

	@impl_registry.lower_setattr(vec_t, 'y')
	def vec_set_y(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		y = context.cast(builder, args[1], sig.args[1], sig.args[0]._member_t)
		setattr(vec, 'y', y)

	@impl_registry.lower_getattr(vec_t, 'z')
	def vec_get_z(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig, 'data')(context, builder, ref = args)
		return vec.z

	@impl_registry.lower_setattr(vec_t, 'z')
	def vec_set_z(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		z = context.cast(builder, args[1], sig.args[1], sig.args[0]._member_t)
		setattr(vec, 'z', z)

	#.xy, .xz, .yz methods
	if Vec == Vector3Type:
		@decl_registry.register_attr
		class Vector3Type_attrs(AttributeTemplate):
			key = vec_t

			def resolve_xy(self, mod):
				return vec_t._vec2_t

			def resolve_xz(self, mod):
				return vec_t._vec2_t

			def resolve_yz(self, mod):
				return vec_t._vec2_t

		@impl_registry.lower_getattr(vec_t, 'xy')
		def vec3_get_xy(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig, 'data')(context, builder, ref = args)
			xy = allocate_vector(context, builder, vec_t._vec2_t)

			for n, attr in enumerate(['x', 'y']):
				setattr(xy, attr, getattr(vec, attr))

			return xy._getpointer()

		@impl_registry.lower_setattr(vec_t, 'xy')
		def vec3_set_xy(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
			xy = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
			vec.x = xy.x
			vec.y = xy.y

		@impl_registry.lower_getattr(vec_t, 'xz')
		def vec3_get_xz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig, 'data')(context, builder, ref = args)
			xz = allocate_vector(context, builder, vec_t._vec2_t)

			for attr2, attr3 in zip(['x', 'y'], ['x', 'z']):
				setattr(xz, attr2, getattr(vec, attr3))

			return xz._getpointer()

		@impl_registry.lower_setattr(vec_t, 'xz')
		def vec3_set_xz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
			xz = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
			vec.x = xz.x
			vec.z = xz.y

		@impl_registry.lower_getattr(vec_t, 'yz')
		def vec3_get_yz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig, 'data')(context, builder, ref = args)
			yz = allocate_vector(context, builder, vec_t._vec2_t)

			for attr2, attr3 in zip(['x', 'y'], ['y', 'z']):
				setattr(yz, attr2, getattr(vec, attr3))

			return yz._getpointer()

		@impl_registry.lower_setattr(vec_t, 'yz')
		def vec3_set_yz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
			yz = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
			vec.y = yz.x
			vec.z = yz.y

##########################################
#Register Vector Methods
##########################################

def base_vec_decl(vec, vec_t):
	#####
	#+, -, *, /
	#####
		# +=, -=, *=, /= (iadd,isub,imul,idiv)

	class vec_op_template(ConcreteTemplate):
		cases = [signature(vec_t, vec_t, vec_t),
				 signature(vec_t, vec_t._member_t, vec_t),
				 signature(vec_t, vec_t, vec_t._member_t)]

	for op in [operator.add, operator.sub, operator.mul, operator.truediv]:  #[operator.iadd, operator.isub, operator.imul, operator.itruediv]
		decl_registry.register_global(op)(vec_op_template)
		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		binary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, (vec_t, vectorize_vector), (types.Number, vectorize_scalar_cast(vec_t._member_t)))

	#####
	#% (should this be vec_t specific?)
	#####

	#####
	#** (should this be vec_t specific?)
	#####

	#TO DO
	#"pow", "powi"

	#####
	#**=, %= (ipow,imod)
	#####

	#TO DO

	#####
	#min, max
	#####

	class vec_op_template(ConcreteTemplate):
				 #binary
		cases = [signature(vec_t, vec_t, vec_t),
				 signature(vec_t, vec_t._member_t, vec_t),
				 signature(vec_t, vec_t, vec_t._member_t),
				 #reduction
				 signature(vec_t._member_t, vec_t)]

	for op in [min, max]:
		decl_registry.register_global(op)(vec_op_template)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		binary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, (vec_t, vectorize_vector), (types.Number, vectorize_scalar_cast(vec_t._member_t)))

		libfunc_reduction_op_caller = libfunc_reduction_op_caller_wrapper(op)
		unary_op_vectorized_wrapper(op, libfunc_reduction_op_caller, (vec_t, vectorize_vector))

	#####
	#sum
	#####

	class vec_op_template(ConcreteTemplate):
		cases = [signature(vec_t._member_t, vec_t)]

	decl_registry.register_global(sum)(vec_op_template)
	libfunc_reduction_op_caller = libfunc_reduction_op_caller_wrapper(operator.add)
	unary_op_vectorized_wrapper(sum, libfunc_reduction_op_caller, (vec_t, vectorize_vector))

	#####
	#dot
	#####

	binary_op_vector_vector_wrapper(mathfuncs.dot, dot_binary_op_caller, vec_t)

	#####
	#getitem
	#####

	# class vec_getitem_template(ConcreteTemplate):
	# 			 #binary
	# 	cases = [signature(vec_t._member_t, vec_t, types.intp),
	# 			 signature(vec_t._member_t, vec_t, types.uintp)]

	# decl_registry.register_global(operator.getitem)(vec_getitem_template)
	# # getitem_vec_factory(vec_t)

	#####
	#shift
	#####

	for scalar_type in [types.uint8, types.uint16, types.uint32, types.uint64, types.int8, types.int16, types.int32, types.int64]:
		binary_op_vector_other_wrapper(mathfuncs.shift, shift_binary_op_caller, vec_t, scalar_type)

def vec_2_typer_wrapper(vec_t): #, Vec
	def vec_2_typer(*attrs):
		#initialize: vec()
		if len(attrs) == 0:
			return vec_t
		#initialize: vec(x) or vec(vec)
		elif len(attrs) == 1:
			vec_or_input_t = attrs
			if isinstance(vec_or_input_t, type(vec_t._member_t)) or isinstance(vec_or_input_t, Vec):
				return vec_t
		#initialize: vec2(x,y)
		elif len(attrs) == 2:
			if all([isinstance(attr, type(vec_t._member_t)) for attr in attrs]):
				return vec_t
		raise ValueError(f"Input to {vec_t.__class__.__name__} not understood")
	return vec_2_typer

def vec_3_typer_wrapper(vec_t): #, Vec
	def vec_3_typer(*attrs):
		#initialize: vec()
		if len(attrs) == 0:
			return vec_t
		#initialize: vec(x) or vec(vec)
		elif len(attrs) == 1:
			vec_or_input_t = attrs
			if isinstance(vec_or_input_t, type(vec_t._member_t)) or isinstance(vec_or_input_t, Vec):
				return vec_t
		#vec3(vec2,y) or vec3(x,vec2)
		elif len(attrs) == 2:
			vec_or_input_t_a, vec_or_input_t_b = attrs
			if isinstance(vec_or_input_t_a, type(vec_t._member_t)) and isinstance(vec_or_input_t_b, Vector2Type):
				return vec_t
			elif isinstance(vec_or_input_t_a, Vector2Type) and isinstance(vec_or_input_t_b, type(vec_t._member_t)):
				return vec_t
		#initialize: vec3(x,y,z)
		elif len(attrs) == 3:
			if all([isinstance(attr, type(vec_t._member_t)) for attr in attrs]):
				return vec_t
		raise ValueError(f"Input to {vec_t.__class__.__name__} not understood")
	return vec_3_typer

unary_floating_ops = ["sin", "cos", "tan",
					  "sinh", "cosh", "tanh",
					  "asin", "acos", "atan",
					  "asinh", "acosh", "atanh",
					  "sqrt", #"rsqrt", "cbrt", "rcbrt",
					  "exp" , "exp10", "exp2", #"expm1",
					  "log", "log10", "log2", #"log1p", "logb",
					  "floor", "ceil"]

binary_floating_ops = ["fmod", "mod", "step"]

ternary_floating_ops = ["clamp", "lerp", "param", "smooth_step"]

def floating_vec_decl(vec, vec_t):
	#####
	#neg, abs
	#####

	class vec_unary_op(ConcreteTemplate):
		cases = [signature(vec_t, vec_t)]

	for op in [operator.neg, abs]: #mathfuncs.abs
		decl_registry.register_global(op)(vec_unary_op)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		unary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, (vec_t, vectorize_vector))

	#####
	#round
	#####

	decl_registry.register_global(round)(vec_unary_op)

	#note: use mathfuncs.round as this returns the correct type. built-in round returns an int
	libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(mathfuncs.round)
	unary_op_vectorized_wrapper(round, libfunc_vectorized_op_caller, (vec_t, vectorize_vector))

	#####
	#unary functions
	#####

	#float/double vector
	for op in unary_floating_ops:
		op = getattr(mathfuncs, op)
		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		unary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, (vec_t, vectorize_vector))

	#####
	#length
	#####

	# class vec_unary_op(ConcreteTemplate):
	# 	cases = [signature(vec_t._member_t, vec_t)]

	# decl_registry.register_global(mathfuncs.length)(vec_unary_op)
	unary_op_wrapper(mathfuncs.length, length_reduction_op_caller, vec_t)

	#####
	#binary functions
	#####

	for op in binary_floating_ops:
		op = getattr(mathfuncs, op)
		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		binary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, (vec_t, vectorize_vector), (types.Number, vectorize_scalar_cast(vec_t._member_t)))

	#non-vectorized ops
		# atan2, cross, etc.

	#####
	#ternary functions
	#####

	for op in ternary_floating_ops:
		op = getattr(mathfuncs, op)
		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		ternary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, (vec_t, vectorize_vector), (types.Number, vectorize_scalar_cast(vec_t._member_t)))

def float_vec_decl(vec, vec_t):
	pass
	# for op in floating_ops:
		# libopf = getattr(mathfuncs, op + 'f')

		# libfunc_vectorized_op_helper = libfunc_vectorized_op_helper_wrapper(libop)
		# vectorized_op_caller = vectorized_op_caller_wrapper(libfunc_vectorized_op_helper)
		# unary_op_wrapper(libop,
		# 				(vec_t, vectorize_vector),
		# 				 vectorized_op_caller)

	# unary_op_factory(mathfuncs.fabs, vec_t, libfunc_helper_wrapper(mathfuncs.fabsf))

	# binary_op_factory(mathfuncs.fminf, vec_t, libfunc_helper_wrapper(mathfuncs.fminf))
	# binary_op_factory(mathfuncs.fmaxf, vec_t, libfunc_helper_wrapper(mathfuncs.fmaxf))

def double_vec_decl(vec, vec_t):
	pass
	# unary_op_factory(mathfuncs.fabs, vec_t, libfunc_helper_wrapper(mathfuncs.fabs))

	# binary_op_factory(mathfuncs.fmin, vec_t, libfunc_helper_wrapper(mathfuncs.fmin))
	# binary_op_factory(mathfuncs.fmax, vec_t, libfunc_helper_wrapper(mathfuncs.fmax))



def integer_vec_decl(vec, vec_t):
	# class vec_op_template(ConcreteTemplate):
	# 	cases = [signature(vec_t, vec_t, vec_t),
	# 			 signature(vec_t, vec_t._member_t, vec_t),
	# 			 signature(vec_t, vec_t, vec_t._member_t)]

	# 	# cases = [signature(vec_t, vec_t, vec_t),
	# 	# 		 signature(vec_t, vec_t._member_t, vec_t),
	# 	# 		 signature(vec_t, vec_t, vec_t._member_t)]

	for op in [operator.and_, operator.or_]:
		# decl_registry.register_global(op)(vec_op_template)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		binary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, (vec_t, vectorize_vector), (types.Integer, vectorize_scalar_cast(vec_t._member_t)))

def signed_vec_decl(vec, vec_t):
	#####
	#neg, abs
	#####

	class vec_unary_op(ConcreteTemplate):
		cases = [signature(vec_t, vec_t)]

	for op in [operator.neg, abs]: #mathfuncs.abs
		decl_registry.register_global(op)(vec_unary_op)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		unary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, (vec_t, vectorize_vector))

# def int_vec_decl(vec, vec_t):
	# pass

# def long_vec_decl(vec, vec_t):
	# unary_op_factory(mathfuncs.llabs, vec_t, libfunc_helper_wrapper(mathfuncs.llabs))

	# binary_op_factory(mathfuncs.llmin, vec_t, libfunc_helper_wrapper(mathfuncs.llmin))
	# binary_op_factory(mathfuncs.llmax, vec_t, libfunc_helper_wrapper(mathfuncs.llmax))



# def unsigned_vec_decl(vec, vec_t):
	# pass

# def uint_vec_decl(vec, vec_t):
	# binary_op_factory(mathfuncs.umin, vec_t, libfunc_helper_wrapper(mathfuncs.umin))
	# binary_op_factory(mathfuncs.umax, vec_t, libfunc_helper_wrapper(mathfuncs.umax))

# def ulong_vec_decl(vec, vec_t):
	# binary_op_factory(mathfuncs.ullmin, vec_t, libfunc_helper_wrapper(mathfuncs.ullmin))
	# binary_op_factory(mathfuncs.ullmax, vec_t, libfunc_helper_wrapper(mathfuncs.ullmax))



#python classes
vec2s = [uchar2, ushort2, uint2, ulong2,
		char2, short2, int2, long2,
		float2, double2]
vec3s = [uchar3, ushort3, uint3, ulong3,
		char3, short3, int3, long3,
		float3, double3]

#initialized numba type classes
vec2s_t = [uchar2_t, ushort2_t, uint2_t, ulong2_t,
		   char2_t, short2_t, int2_t, long2_t,
		   float2_t, double2_t]

vec3s_t = [uchar3_t, ushort3_t, uint3_t, ulong3_t,
		   char3_t, short3_t, int3_t, long3_t,
		   float3_t, double3_t]

vec_groups = [vec2s, vec3s]
vec_t_groups = [vec2s_t, vec3s_t]
Vecs = [Vector2Type, Vector3Type]
vec_typer_wrappers = [vec_2_typer_wrapper, vec_3_typer_wrapper]

for vecs, vecs_t, Vec, vec_typer_wrapper in zip(vec_groups, vec_t_groups, Vecs, vec_typer_wrappers):
	for vec, vec_t in zip(vecs, vecs_t):
		vec_decl(vec, vec_t, Vec, vec_typer_wrapper(vec_t))

##########################################
#Vector Type Specific Methods
##########################################

vec2s = [float2, double2]
vec3s = [float3, double3]

vec2s_t = [float2_t, double2_t]
vec3s_t = [float3_t, double3_t]

vec_groups = [vec2s, vec3s]
vec_t_groups = [vec2s_t, vec3s_t]

for vecs, vecs_t in zip(vec_groups, vec_t_groups):
	for vec, vec_t in zip(vecs, vecs_t):
		base_vec_decl(vec, vec_t)
		floating_vec_decl(vec, vec_t)
# float_vec_decl(vec, vec_t)
# double_vec_decl(vec, vec_t)



vec2s = [char2, short2, int2, long2, uchar2, ushort2, uint2, ulong2]
vec3s = [char3, short3, int3, long3, uchar3, ushort3, uint3, ulong3]
vec2s_t = [char2_t, short2_t, int2_t, long2_t, uchar2_t, ushort2_t, uint2_t, ulong2_t]
vec3s_t = [char3_t, short3_t, int3_t, long3_t, char3_t, short3_t, int3_t, long3_t]

vec_groups = [vec2s, vec3s]
vec_t_groups = [vec2s_t, vec3s_t]

for vecs, vecs_t in zip(vec_groups, vec_t_groups):
	for vec, vec_t in zip(vecs, vecs_t):
		base_vec_decl(vec, vec_t)		
		integer_vec_decl(vec, vec_t)

vec_groups = [vec2s[:4], vec3s[:4]]
vec_t_groups = [vec2s_t[:4], vec3s_t[:4]]

for vecs, vecs_t in zip(vec_groups, vec_t_groups):
	for vec, vec_t in zip(vecs, vecs_t):
		signed_vec_decl(vec, vec_t)
# int_vec_decl(vec, vec_ts)
# long_vec_decl(vec, vec_ts)

# vec_groups = [vec2s[4:], vec3s[4:]]
# vec_t_groups = [vec2s_t[4:], vec3s_t[4:]]

# for vecs, vec_ts in zip(vec_groups, vec_t_groups):
# 	for vec, vec_t in zip(vecs, vec_ts):
# 		unsigned_vec_decl(vec, vec_ts)
# # uint_vec_decl(vec, vec_ts)
# # ulong_vec_decl(vec, vec_ts)
