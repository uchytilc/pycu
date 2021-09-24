from numba import types
from numba.core.extending import typeof_impl, type_callable
from numba.core.typing import signature

from numba.core.extending import models, register_model, lower_builtin
from numba.core import cgutils
from numba.cpython import mathimpl

from numba.cuda.cudadecl import registry as decl_registry
from numba.cuda.cudaimpl import registry as impl_registry
from numba.core.typing.templates import ConcreteTemplate, AttributeTemplate

import operator
from .vectortype import ( char2,  short2,  int2,  long2,
						 uchar2, ushort2, uint2, ulong2,
						 float2, double2,
						  char3,  short3,  int3,  long3,
						 uchar3, ushort3, uint3, ulong3,
						 float3, double3)
from .vectordecl import Vector2Type, Vector3Type
from .vectordecl import (char2_t,  short2_t,  int2_t,  long2_t,
						uchar2_t, ushort2_t, uint2_t, ulong2_t,
						float2_t, double2_t,
						 char3_t,  short3_t,  int3_t,  long3_t,
						uchar3_t, ushort3_t, uint3_t, ulong3_t,
						float3_t, double3_t)

from .. import mathfuncs

#generic libfunc helper function for unary and binary vector ops
def libfunc_helper_wrapper(cuda_op):
	def libfunc_helper(context, builder, sig, args):
		types = [sig.return_type.__member_type__]*(len(sig.args) + 1)
		sig = signature(*types)
		return context.get_function(cuda_op, sig), libfunc_op_caller
	return libfunc_helper

#####################
#op caller functions#
#####################

#these are returned from a helper function along with the op(s) that the caller function will call
#builder, mathimpl, and libfunc each have their own calling convention so a unique caller is required for each

def builder_op_caller(op,builder,rtrn,*args):
	for n, attr in enumerate(rtrn._fe_type.__member_names__):
		setattr(rtrn, attr, op(*[arg[n] for arg in args]))
	return rtrn._getvalue()

def mathimpl_op_caller(op,builder,rtrn,*args):
	for n, attr in enumerate(rtrn._fe_type.__member_names__):
		setattr(rtrn, attr, op(builder, *[arg[n] for arg in args]))
	return rtrn._getvalue()

def libfunc_op_caller(op,builder,rtrn,*args):
	for n, attr in enumerate(rtrn._fe_type.__member_names__):
		setattr(rtrn, attr, op(builder, [arg[n] for arg in args]))
	return rtrn._getvalue()

###########
#UNARY OPS#
###########

def unary_op_factory(op, vec_t, op_helper):
	@impl_registry.lower(op, vec_t)
	def cuda_op_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.return_type)(context, builder, value = args[0])
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)

		a_members = [getattr(a, attr) for attr in sig.return_type.__member_names__]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,out,a_members)

########################
#unary helper functions#
########################

#a helper function defines the operation(s) to be called on the vector as well as returns the caller that will execute the operation(s)

# def neg_helper(context, builder, sig, args):
	# # neg = builder.neg
	# # caller = builder_op_caller
	# # if isinstance(sig.return_type.__member_type__, types.Float):
	# # 	neg = mathimpl.negate_real
	# # 	caller = mathimpl_op_caller
	# # return neg, caller

	# # # #for when fneg is supported
	# # # neg = builder.neg
	# # # if isinstance(sig.return_type.__member_type__, types.Float):
	# # # 	neg = builder.fneg
	# # # return neg, builder_op_caller

	# libsig = signature(sig.args[0].__member_type__,
	# 				   sig.args[0].__member_type__)
	# neg = context.get_function(operator.neg, libsig)
	# return neg, libfunc_op_caller

############
#BINARY OPS#
############

def binary_op_factory_vec_vec(op, vec_t, op_helper):
	@impl_registry.lower(op, vec_t, vec_t)
	def cuda_op_vec_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.return_type)(context, builder, value = args[0])
		b = cgutils.create_struct_proxy(sig.return_type)(context, builder, value = args[1])
		out = cgutils.create_struct_proxy(vec_t)(context, builder)

		a_members = [getattr(a, attr) for attr in sig.return_type.__member_names__]
		b_members = [getattr(b, attr) for attr in sig.return_type.__member_names__]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,out,a_members,b_members)

def binary_op_factory_vec_scalar(op, vec_t, op_helper):
	@impl_registry.lower(op, types.Number, vec_t)
	@impl_registry.lower(op, vec_t, types.Number)
	def cuda_op_vec_scalar(context, builder, sig, args):
		a = sig.args[1] == sig.return_type
		b = 1 - a

		a = cgutils.create_struct_proxy(vec_t)(context, builder, value=args[a])
		b = context.cast(builder, args[b], sig.args[b], sig.return_type.__member_type__)
		out = cgutils.create_struct_proxy(vec_t)(context, builder)

		a_members = [getattr(a, attr) for attr in sig.return_type.__member_names__]
		b_members = [b for attr in sig.return_type.__member_names__]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,out,a_members,b_members)

def binary_op_factory(op, vec_t, op_helper):
	binary_op_factory_vec_vec(op, vec_t, op_helper)
	binary_op_factory_vec_scalar(op, vec_t, op_helper)

#########################
#binary helper functions#
#########################

# def add_helper(context, builder, sig, args):
	# add = builder.add
	# if isinstance(sig.return_type.__member_type__, types.Float):
	# 	add = builder.fadd
	# return add, builder_op_caller

# def sub_helper(context, builder, sig, args):
	# sub = builder.sub
	# if isinstance(sig.return_type.__member_type__, types.Float):
	# 	sub = builder.fsub
	# return sub, builder_op_caller

# def mul_helper(context, builder, sig, args):
	# mul = builder.mul
	# if isinstance(sig.return_type.__member_type__, types.Float):
	# 	mul = builder.fmul
	# return mul, builder_op_caller

# def div_helper(context, builder, sig, args):
	# div = builder.udiv
	# if isinstance(sig.return_type.__member_type__, types.Float):
	# 	div = builder.fdiv
	# elif sig.return_type.__member_type__.__member_type__.signed:
	# 	div = builder.sdiv
	# return div, builder_op_caller




# def mod_helper(context, builder, sig, args):
	# libfunc_impl = context.get_function(cuda_op, sig)

	# # mod = 

	# # 	for n, attr in enumerate(vec_t.__member_names__):
	# # 		setattr(out, attr, libfunc_impl(builder, [getattr(a, attr)]))
	# # 	# out.x = libfunc_impl(builder, [a.x])
	# # 	# out.y = libfunc_impl(builder, [a.y])

# def pow_helper(context, builder, sig, args):
	# pass
	#DONT WANT TO CAST SCALAR VALUE USED FOR POWER
	# out = cgutils.create_struct_proxy(vec_t)(context, builder)

	# libfunc_impl = context.get_function(operator.pow, signature(ax, ax, ax))
	# out.x = libfunc_impl(builder, [ax, bx]) 
	# out.y = libfunc_impl(builder, [ay, by]) 

	# return out._getvalue()




###############
#REDUCTION OPS# (ops that take in a vector and return a scalar)
###############

def reduction_op_factory(op, vec_t, op_helper):
	#functions that reduce a vector input to a scalar output by applying op to each element of the input vector
	@impl_registry.lower(op, vec_t)
	def cuda_op_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0])(context, builder, value = args[0])
		a_members = [getattr(a, attr) for attr in sig.args[0].__member_names__]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,a_members)

#############################
#reduction callers functions#
#############################

def reduction_builder_op_caller(op,builder,vec):
	val = vec[0]
	for n in range(len(vec) - 1):
		val = op(val, vec[n + 1])
	return val

def reduction_libfunc_op_caller(op,builder,vec):
	val = vec[0]
	for n in range(len(vec) - 1):
		val = op(builder, [val, vec[n + 1]])
	return val

def length_caller(ops, builder, vec):
	add, mul, sqrt = ops
	val = mul(builder, [vec[0], vec[0]])
	for n in range(len(vec) - 1):
		val = add(builder, [val, mul(builder, [vec[n + 1], vec[n + 1]])])
	return sqrt(builder, [val])

# clamp
# lerp
# smooth_step
# param

#######################################
#reduction op helper functions#
#######################################

def sum_helper(context, builder, sig, args):
	# add = builder.add
	# if isinstance(sig.args[0].__member_type__, types.Float):
	# 	add = builder.fadd
	# return add, reduction_builder_op_caller

	libsig = signature(sig.args[0].__member_type__,
					   sig.args[0].__member_type__,
					   sig.args[0].__member_type__)
	add = context.get_function(operator.add, libsig)

	return add, reduction_libfunc_op_caller

def length_helper(context, builder, sig, args):
	libsig = signature(sig.args[0].__member_type__,
					   sig.args[0].__member_type__)
	sqrt = context.get_function(mathfuncs.sqrt, libsig)

	libsig = signature(sig.args[0].__member_type__,
					   sig.args[0].__member_type__,
					   sig.args[0].__member_type__)
	mul = context.get_function(operator.mul, libsig)
	add = context.get_function(operator.add, libsig)

	return [add, mul, sqrt], length_caller

def reduction_libfunc_helper_wrapper(cuda_op):
	def reduction_libfunc_helper(context, builder, sig, args):
		sig = signature(sig.args[0].__member_type__,
						sig.args[0].__member_type__,
						sig.args[0].__member_type__)
		return context.get_function(cuda_op, sig), reduction_libfunc_op_caller
	return reduction_libfunc_helper



def getitem_vec_factory(vec_t):
	#numba/numba/cpython/tupleobj.py 
		#def getitem_unituple(context, builder, sig, args):

	@impl_registry.lower(operator.getitem, vec_t, types.Integer)
	def vec_getitem(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig.args[0])(context, builder, value = args[0])
		idx = args[1]

		# vec_len = len(vec_t.__member_names__)

		# zero = const_zero(context)
		# one = const_nan(context)
		# one_64 = context.get_constant(types.float64, np.float64(1))

		# negative = builder.fcmp_unordered('<', idx, zero)
		# with builder.if_else(negative) as (true, false):

		return vec.x

##################
#VECTOR FUNCTIONS#
##################

from . import vectorfuncs
from ..mathdecl import register_op, unary_op_fixed_type_factory, binary_op_fixed_type_factory

#inject/insert functions into mathfuncs if they don't already exist (because functions aren't typed only a single instance of a function with the given name needs to exist within core)
vec_funcs = [(vectorfuncs.sum, unary_op_fixed_type_factory, None),
			 (vectorfuncs.dot, binary_op_fixed_type_factory, None),
			 (vectorfuncs.length, unary_op_fixed_type_factory, None)]

for vec_func in vec_funcs:
	func, factory, *factory_args = vec_func
	name = func.__name__
	libfunc = getattr(mathfuncs, name, None)
	#only register a vector function if a stub doesn't already exist within core
	if libfunc is None:
		setattr(mathfuncs, name, func)
		#note: None is set as default return type for some of the vec_funcs so that a typer_ method must be defined to use the op on input types
		register_op(name, factory, *factory_args)


def vec_decl(vec, vec_t, Vec):
	##########################################
	#Define Vector Type
	##########################################

	#note: the typer function must have the same number of arguments as inputs given to the struct so each initialize style needs its own typer

	@typeof_impl.register(vec)
	def typeof_vec(val, c):
		return vec_t

	def vec_typer(*attrs):
		if all([isinstance(attr, vec_t.__input_type__) for attr in attrs]):
			return vec_t
		else:
			raise ValueError(f"Input to {vec.__name__} not understood")

	#vector length specific initialization
	if Vec == Vector2Type:
		#initialize: vec2(x,y)
		@type_callable(vec)
		def type_vec2(context):
			def typer(x, y):
				return vec_typer(x,y)
			return typer
	elif Vec == Vector3Type:
		#initialize: vec3(x,y,z)
		@type_callable(vec)
		def type_vec3(context):
			def typer(x, y, z):
				return vec_typer(x,y,z)
			return typer

		#initialize: vec3(vec2,y) or vec3(x,vec2)
		@type_callable(vec)
		def type_vec3(context):
			def typer(vec_or_scalar_a, vec_or_scalar_b):
				if isinstance(vec_or_scalar_a, vec_t.__input_type__) and isinstance(vec_or_scalar_b, Vector2Type):
					return vec_t
				elif isinstance(vec_or_scalar_a, Vector2Type) and isinstance(vec_or_scalar_b, vec_t.__input_type__):
					return vec_t
				else:
					raise ValueError(f"Input to {vec.__name__} not understood")
			return typer

	#initialize: vec(x) or vec(vec)
	@type_callable(vec)
	def type_vec(context):
		def typer(vec_or_scalar):
			if isinstance(vec_or_scalar, vec_t.__input_type__) or isinstance(vec_or_scalar, Vec):
				return vec_t
			else:
				raise ValueError(f"Input to {vec.__name__} not understood")
		return typer

	#initialize: vec()
	@type_callable(vec)
	def type_vec(context):
		def typer():
			return vec_t
		return typer

	@register_model(type(vec_t)) 
	class VecModel(models.StructModel):
		def __init__(self, dmm, fe_type):
			models.StructModel.__init__(self, dmm, fe_type, vec_t.__members__)

	##########################################
	#Initializer/Constructor Methods
	##########################################

	#initialize: vec(vec)
	@lower_builtin(vec, Vec)
	def impl_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
		for n, attr in enumerate(vec_t.__member_names__):
			setattr(out, attr, context.cast(builder, getattr(a, attr), sig.args[0].__member_type__, sig.return_type.__member_type__))
		return out._getvalue()

	#initialize: vec(x,y,...)
	@lower_builtin(vec, *[vec_t.__input_type__]*len(vec_t.__members__))
	def impl_vec(context, builder, sig, args):
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
		for n, attr in enumerate(vec_t.__member_names__):
			setattr(out, attr, context.cast(builder, args[n], sig.args[n], sig.return_type.__member_type__))
		return out._getvalue()

	#initialize: vec(x)
	@lower_builtin(vec, vec_t.__input_type__)
	def impl_vec(context, builder, sig, args):
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
		cast = context.cast(builder, args[0], sig.args[0], sig.return_type.__member_type__)
		for n, attr in enumerate(vec_t.__member_names__):
			setattr(out, attr, cast)
		return out._getvalue()

	#initialize: vec()
	@lower_builtin(vec)
	def impl_vec(context, builder, sig, args):
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
		const = context.get_constant(sig.return_type.__member_type__, 0)
		for n, attr in enumerate(vec_t.__member_names__):
			setattr(out, attr, const)
		return out._getvalue()

	if Vec == Vector3Type:
		#initialize: vec(x, vec2)
		@lower_builtin(vec, Vector2Type, vec_t.__input_type__)
		def impl_vec(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0])(context, builder, value = args[0])
			out = cgutils.create_struct_proxy(sig.return_type)(context, builder)

			out.x = context.cast(builder, vec.x, sig.args[0].__member_type__, sig.return_type.__member_type__)
			out.y = context.cast(builder, vec.y, sig.args[0].__member_type__, sig.return_type.__member_type__)
			out.z = context.cast(builder, args[1], sig.args[1], sig.return_type.__member_type__)

			return out._getvalue()

		#initialize: vec(vec2, x)
		@lower_builtin(vec, vec_t.__input_type__, Vector2Type)
		def impl_vec(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[1])(context, builder, value = args[1])
			out = cgutils.create_struct_proxy(sig.return_type)(context, builder)

			out.x = context.cast(builder, args[0], sig.args[0], sig.return_type.__member_type__)
			out.y = context.cast(builder, vec.x, sig.args[1].__member_type__, sig.return_type.__member_type__)
			out.z = context.cast(builder, vec.y, sig.args[1].__member_type__, sig.return_type.__member_type__)

			return out._getvalue()

	##########################################
	#Define Vector Attributes
	##########################################

	@decl_registry.register_attr
	class Vec_attrs(AttributeTemplate):
		key = vec_t

		def resolve_x(self, mod):
			return vec_t.__member_type__

		def resolve_y(self, mod):
			return vec_t.__member_type__

		def resolve_z(self, mod):
			return vec_t.__member_type__

		def resolve___get_item__(self, mod):
			return vec_t.__member_type__

	@impl_registry.lower_getattr(vec_t, 'x')
	def vec_get_x(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
		return vec.x

	# #SETATTR CURRENTLY NOT SUPPORTED ON GPU
	# @impl_registry.lower_setattr(vec_t, 'x')
	# def vec_set_x(context, builder, sig, args):
		# typ, valty = sig.args
		# target, val = args

		# vec = cgutils.create_struct_proxy(typ)(context, builder, value=target)
		# val = context.cast(builder, val, valty, typ.__member_type__)
		# out = cgutils.create_struct_proxy(typ)(context, builder)

		# return setattr(out, 'x', val)

	@impl_registry.lower_getattr(vec_t, 'y')
	def vec_get_y(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
		return vec.y

	# @impl_registry.lower_setattr(vec_t, 'y')
	# def vec_set_y(context, builder, sig, args):
	# 	pass

	@impl_registry.lower_getattr(vec_t, 'z')
	def vec_get_z(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
		return vec.z

	# @impl_registry.lower_setattr(vec_t, 'z')
	# def vec_set_z(context, builder, sig, args):
	# 	pass

	#.xy, .xz, .yz methods
	if Vec == Vector3Type:
		@decl_registry.register_attr
		class Vector3Type_attrs(AttributeTemplate):
			key = vec_t

			def resolve_xy(self, mod):
				return vec_t.__vec2_type__

			def resolve_xz(self, mod):
				return vec_t.__vec2_type__

			def resolve_yz(self, mod):
				return vec_t.__vec2_type__

		@impl_registry.lower_getattr(vec_t, 'xy')
		def vec3_get_xy(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
			xy = cgutils.create_struct_proxy(sig.__vec2_type__)(context, builder)
			xy.x = vec.x
			xy.y = vec.y
			return xy._getvalue()

		# @impl_registry.lower_setattr(vec_t, 'xy')
		# def vec3_set_xy(context, builder, sig, args):
			# pass
			# #scalar applies to every input
			# #vec2 applies to each component

		@impl_registry.lower_getattr(vec_t, 'xz')
		def vec3_get_xz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
			xz = cgutils.create_struct_proxy(sig.__vec2_type__)(context, builder)
			xz.x = vec.x
			xz.y = vec.z
			return xz._getvalue()

		# @impl_registry.lower_setattr(vec_t, 'xz')
		# def vec3_set_xz(context, builder, sig, args):
		# 	pass

		@impl_registry.lower_getattr(vec_t, 'yz')
		def vec3_get_yz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig)(context, builder, value=args)
			yz = cgutils.create_struct_proxy(sig.__vec2_type__)(context, builder)
			yz.x = vec.y
			yz.y = vec.z
			return yz._getvalue()

		# @impl_registry.lower_setattr(vec_t, 'yz')
		# def vec3_set_yz(context, builder, sig, args):
		# 	pass

	##########################################
	#Register Vector Methods
	##########################################

	#####
	#+, -, *, /
	#####

	class vec_op_template(ConcreteTemplate):
		cases = [signature(vec_t, vec_t, vec_t),
				 signature(vec_t, vec_t.__member_type__, vec_t),
				 signature(vec_t, vec_t, vec_t.__member_type__)]

	#OLD METHOD
	# ops = [(operator.add, add_helper),
	# 	   (operator.sub, sub_helper),
	# 	   (operator.mul, mul_helper),
	# 	   (operator.truediv, div_helper)]
	# for op, helper in ops:
	# 	decl_registry.register_global(op)(vec_op_template)
	# 	binary_op_factory(op, vec_t, helper)
	###########

	ops = [operator.add, operator.sub, operator.mul, operator.truediv]
	for op in ops:
		decl_registry.register_global(op)(vec_op_template)
		binary_op_factory(op, vec_t, libfunc_helper_wrapper(op))

	#####
	#+=, -=, *=, /= (iadd,isub,imul,idiv)
	#####

	#TO DO

	#####
	#% (should this be vec_t specific?)
	#####

	#TO DO
	#"fmod"

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
	#min, max, sum
	#####

	class vec_op_template(ConcreteTemplate):
				 #binary
		cases = [signature(vec_t, vec_t, vec_t),
				 signature(vec_t, vec_t.__member_type__, vec_t),
				 signature(vec_t, vec_t, vec_t.__member_type__),
				 #reduction
				 signature(vec_t.__member_type__, vec_t)]

	decl_registry.register_global(min)(vec_op_template)
	decl_registry.register_global(max)(vec_op_template)
	decl_registry.register_global(sum)(vec_op_template)

	min_helper = libfunc_helper_wrapper(min)
	max_helper = libfunc_helper_wrapper(max)

	binary_op_factory(min, vec_t, min_helper)
	binary_op_factory(max, vec_t, max_helper)
	# binary_op_factory(mathfuncs.min, vec_t, min_helper)
	# binary_op_factory(mathfuncs.max, vec_t, max_helper)

	min_helper = reduction_libfunc_helper_wrapper(min)
	max_helper = reduction_libfunc_helper_wrapper(max)

	reduction_op_factory(min, vec_t, min_helper)
	reduction_op_factory(max, vec_t, max_helper)
	# reduction_op_factory(mathfuncs.min, vec_t, min_helper)
	# reduction_op_factory(mathfuncs.max, vec_t, max_helper)

	reduction_op_factory(sum, vec_t, sum_helper)
	reduction_op_factory(mathfuncs.sum, vec_t, sum_helper)

	binary_op_factory_vec_vec(mathfuncs.dot, vec_t, libfunc_helper_wrapper(operator.mul))

	class vec_getitem_template(ConcreteTemplate):
				 #binary
		cases = [signature(vec_t.__member_type__, vec_t, types.intp),
				 signature(vec_t.__member_type__, vec_t, types.uintp)]

	decl_registry.register_global(operator.getitem)(vec_getitem_template)
	getitem_vec_factory(vec_t)


floating_ops = ["sin", "cos", "tan",
				"sinh", "cosh", "tanh",
				"asin", "acos", "atan",
				"asinh", "acosh", "atanh",
				"sqrt", #"rsqrt", "cbrt", "rcbrt",
				"exp" , "exp10", "exp2", #"expm1",
				"log", "log10", "log2", #"log1p", "logb",
				"floor", "ceil"]

def floating_vec_decl(vec, vec_t):
	#####
	#- (neg), abs
	#####

	class vec_unary_op(ConcreteTemplate):
		cases = [signature(vec_t, vec_t)]

	neg_helper = libfunc_helper_wrapper(operator.neg)
	decl_registry.register_global(operator.neg)(vec_unary_op)
	unary_op_factory(operator.neg, vec_t, neg_helper)

	abs_helper = libfunc_helper_wrapper(abs) #libfunc_helper_wrapper(mathfuncs.abs)
	decl_registry.register_global(abs)(vec_unary_op)
	unary_op_factory(abs, vec_t, abs_helper)
	# unary_op_factory(mathfuncs.abs, vec_t, abs_helper)

	#####
	#unary functions
	#####

	#float/double vector
	for op in floating_ops:
		libop = getattr(mathfuncs, op)
		unary_op_factory(libop, vec_t, libfunc_helper_wrapper(libop))

	#####
	#round
	#####

	class vec_unary_op(ConcreteTemplate):
		cases = [signature(vec_t, vec_t)]

	round_helper = libfunc_helper_wrapper(mathfuncs.round)
	decl_registry.register_global(round)(vec_unary_op)
	unary_op_factory(round, vec_t, round_helper)
	unary_op_factory(mathfuncs.round, vec_t, round_helper)

	reduction_op_factory(mathfuncs.length, vec_t, length_helper)

	#####
	#binary functions
	#####

	# atan2, cross, lerp, etc.

def float_vec_decl(vec, vec_t):
	for op in floating_ops:
		libopf = getattr(mathfuncs, op + 'f')
		unary_op_factory(libopf, vec_t, libfunc_helper_wrapper(libopf))
	unary_op_factory(mathfuncs.fabs, vec_t, libfunc_helper_wrapper(mathfuncs.fabsf))

	binary_op_factory(mathfuncs.fminf, vec_t, libfunc_helper_wrapper(mathfuncs.fminf))
	binary_op_factory(mathfuncs.fmaxf, vec_t, libfunc_helper_wrapper(mathfuncs.fmaxf))

def double_vec_decl(vec, vec_t):
	unary_op_factory(mathfuncs.fabs, vec_t, libfunc_helper_wrapper(mathfuncs.fabs))

	binary_op_factory(mathfuncs.fmin, vec_t, libfunc_helper_wrapper(mathfuncs.fmin))
	binary_op_factory(mathfuncs.fmax, vec_t, libfunc_helper_wrapper(mathfuncs.fmax))



def signed_vec_decl(vec, vec_t):
	#####
	#- (neg), abs
	#####

	class vec_unary_op(ConcreteTemplate):
		cases = [signature(vec_t, vec_t)]

	neg_helper = libfunc_helper_wrapper(operator.neg)
	decl_registry.register_global(operator.neg)(vec_unary_op)
	unary_op_factory(operator.neg, vec_t, neg_helper)

	abs_helper = libfunc_helper_wrapper(abs)#libfunc_helper_wrapper(mathfuncs.abs)
	decl_registry.register_global(abs)(vec_unary_op)
	unary_op_factory(abs, vec_t, abs_helper)
	# unary_op_factory(mathfuncs.abs, vec_t, abs_helper)

def int_vec_decl(vec, vec_t):
	pass

def long_vec_decl(vec, vec_t):
	unary_op_factory(mathfuncs.llabs, vec_t, libfunc_helper_wrapper(mathfuncs.llabs))

	binary_op_factory(mathfuncs.llmin, vec_t, libfunc_helper_wrapper(mathfuncs.llmin))
	binary_op_factory(mathfuncs.llmax, vec_t, libfunc_helper_wrapper(mathfuncs.llmax))



def unsigned_vec_decl(vec, vec_t):
	pass

def uint_vec_decl(vec, vec_t):
	binary_op_factory(mathfuncs.umin, vec_t, libfunc_helper_wrapper(mathfuncs.umin))
	binary_op_factory(mathfuncs.umax, vec_t, libfunc_helper_wrapper(mathfuncs.umax))

def ulong_vec_decl(vec, vec_t):
	binary_op_factory(mathfuncs.ullmin, vec_t, libfunc_helper_wrapper(mathfuncs.ullmin))
	binary_op_factory(mathfuncs.ullmax, vec_t, libfunc_helper_wrapper(mathfuncs.ullmax))














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

for vecs, vecs_t, Vec in zip(vec_groups, vec_t_groups, Vecs):
	for vec, vec_t in zip(vecs, vecs_t):
		vec_decl(vec, vec_t, Vec)

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
		floating_vec_decl(vec, vec_t)
# float_vec_decl(vec, vec_t)
# double_vec_decl(vec, vec_t)


vec2s = [char2, short2, int2, long2]
vec3s = [char3, short3, int3, long3]

vec2s_t = [char2_t, short2_t, int2_t, long2_t]
vec3s_t = [char3_t, short3_t, int3_t, long3_t]

vec_groups = [vec2s, vec3s]
vec_t_groups = [vec2s_t, vec3s_t]

for vecs, vecs_t in zip(vec_groups, vec_t_groups):
	for vec, vec_t in zip(vecs, vecs_t):
		signed_vec_decl(vec, vec_t)
# int_vec_decl(vec, vec_ts)
# long_vec_decl(vec, vec_ts)


# vec2s = [uchar2, ushort2, uint2, ulong2]
# vec3s = [uchar3, ushort3, uint3, ulong3]

# vec2types = [uchar2_type, ushort2_type, uint2_type, ulong2_type]
# vec3types = [uchar3_type, ushort3_type, uint3_type, ulong3_type]

# vec_groups = [vec2s, vec3s]
# vec_t_groups = [vec2types, vec3types]

# for vecs, vec_ts in zip(vec_groups, vec_t_groups):
# 	for vec, vec_t in zip(vecs, vec_ts):
# 		unsigned_vec_decl(vec, vec_ts)
# # uint_vec_decl(vec, vec_ts)
# # ulong_vec_decl(vec, vec_ts)
