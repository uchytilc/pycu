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
from .vectordecl import VectorType, Vector2Type, Vector3Type
from .vectordecl import (char2_t,  short2_t,  int2_t,  long2_t,
						uchar2_t, ushort2_t, uint2_t, ulong2_t,
						float2_t, double2_t,
						 char3_t,  short3_t,  int3_t,  long3_t,
						uchar3_t, ushort3_t, uint3_t, ulong3_t,
						float3_t, double3_t)

from llvmlite import ir

from .. import mathfuncs

import numpy as np

#for loop example
	#https://github.com/numba/numba/blob/7df40d2e36d15c4965b849b109c06593d6cc7b64/numba/core/cgutils.py#L487


#generic libfunc helper function for unary and binary vector ops
def libfunc_helper_wrapper(cuda_op):
	def libfunc_helper(context, builder, sig, args):
		# types = [sig.return_type._member_t]
		# for typ in sig.args:
		# 	if isinstance(typ, VectorType):
		# 		types.append(sig.return_type._member_t)
		# 	else:
		# 		types.append(sig.return_type.__precision__)

		types = [sig.return_type._member_t]*(len(sig.args) + 1)
		sig = signature(*types)
		return context.get_function(cuda_op, sig), libfunc_op_caller
	return libfunc_helper

#####################
#op caller functions#
#####################

#these are returned from a helper function along with the op(s) that the caller function will call.
#builder, mathimpl, and libfunc each have their own calling convention so a unique caller is required for each.

def builder_op_caller(op, builder, out, *args):
	for n, attr in enumerate(out._fe_type._fields):
		setattr(out, attr, op(*[arg[n] for arg in args]))
	return out._getvalue()

def mathimpl_op_caller(op, builder, out, *args):
	for n, attr in enumerate(out._fe_type._fields):
		setattr(out, attr, op(builder, *[arg[n] for arg in args]))
	return out._getvalue()

def libfunc_op_caller(op, builder, out, *args):
	for n, attr in enumerate(out._fe_type._fields):
		setattr(out, attr, op(builder, [arg[n] for arg in args]))
	return out._getvalue()

###########
#UNARY OPS#
###########


def unary_op_factory(op, vec_t, op_helper):
	@impl_registry.lower(op, vec_t)
	def cuda_op_vec(context, builder, sig, args):
		f = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = args[0])
		members = [getattr(f, attr) for attr in sig.return_type._fields]

		out_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)
		out = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = out_ptr)

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, out, members)
		return out_ptr







#########################
#unary callers functions#
#########################


########################
#unary helper functions#
########################

#a helper function defines the operation(s) to be called on the vector as well as returns the caller that will execute the operation(s)

# def neg_helper(context, builder, sig, args):
	# # neg = builder.neg
	# # caller = builder_op_caller
	# # if isinstance(sig.return_type._member_t, types.Float):
	# # 	neg = mathimpl.negate_real
	# # 	caller = mathimpl_op_caller
	# # return neg, caller

	# # # #for when fneg is supported
	# # # neg = builder.neg
	# # # if isinstance(sig.return_type._member_t, types.Float):
	# # # 	neg = builder.fneg
	# # # return neg, builder_op_caller

	# libsig = signature(sig.args[0]._member_t,
	# 				   sig.args[0]._member_t)
	# neg = context.get_function(operator.neg, libsig)
	# return neg, libfunc_op_caller

############
#BINARY OPS#
############

def binary_op_factory_vec_vec(op, vec_t, op_helper):
	@impl_registry.lower(op, vec_t, vec_t)
	def cuda_op_vec_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		b = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])

		#need to convert numba type to llvmlite type

		# temp = ir.LiteralStructType([vec_t._member_t for _ in range(len(sig.return_type._fields))])
		# print(temp)


		# _be_type

		# print(vec_t, sig.return_type)


		# vec_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)

		vec_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		# print(args[0])
		# print(args[0].type)
		# print(args[0].type.pointee)

		a_members = [getattr(a, attr) for attr in sig.return_type._fields]
		b_members = [getattr(b, attr) for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members)
		return vec_ptr

#note: mtv = member to vector (converts a scalar to a vector containing the scalar for every entry)
def binary_op_factory_vec_mtv(op, vec_t, op_helper):
	#converts scalar argument to a vector of the same length as the other argument
	@impl_registry.lower(op, vec_t, types.Number)
	def cuda_op_vec_mtv(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		b = context.cast(builder, args[1], sig.args[1], sig.return_type._member_t)

		vec_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [getattr(a, attr) for attr in sig.return_type._fields]
		b_members = [b for attr in sig.return_type._fields]

		op_, caller = op_helper(context, builder, sig, args)
		caller(op_, builder, vec, a_members, b_members)
		return vec_ptr

	@impl_registry.lower(op, types.Number, vec_t)
	def cuda_op_mtv_vec(context, builder, sig, args):
		a = context.cast(builder, args[0], sig.args[0], sig.return_type._member_t)
		b = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])

		vec_ptr = cgutils.alloca_once(builder, args[1].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [a for attr in sig.return_type._fields]
		b_members = [getattr(b, attr) for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members)
		return vec_ptr

def binary_op_factory(op, vec_t, op_helper):
	binary_op_factory_vec_vec(op, vec_t, op_helper)
	binary_op_factory_vec_mtv(op, vec_t, op_helper)

def binary_op_factory_vec_scalar(op, vec_t, op_helper, scalar_type = types.Number):
	@impl_registry.lower(op, vec_t, scalar_type)
	def cuda_op_vec_scalar(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(vec_t, 'data')(context, builder, ref = args[0])
		scalar = args[1]

		vec_members = [getattr(vec, attr) for attr in sig.args[0]._fields]

		vec_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		caller = op_helper(context, builder, sig, args)
		caller(vec, vec_members, scalar)
		return vec_ptr

#########################
#binary callers functions#
#########################

#first argument is vector, second argument is integer





	# #hard coded conditionals for up to a length 4 vector. For support of any length this will need to be a for loop
	# first = builder.icmp_signed('==', n, one)
	# with builder.if_else(first) as (true, false):
	# 	with true:
	# 		for m in range(member_count):
	# 			setattr(out, member_names[(n + 1)%member_count], getattr(vec, member_names[m]))
	# 	with false:
	# 		second = builder.icmp_signed('==', n, two)
	# 		with builder.if_else(second) as (true, false):
	# 			with true:
	# 				for m in range(member_count):
	# 					setattr(out, member_names[(n + 2)%member_count], getattr(vec, member_names[m]))
	# 			with false:
	# 				third = builder.icmp_signed('==', n, three)
	# 				with builder.if_else(third) as (true, false):
	# 					with true:
	# 						for m in range(member_count):
	# 							setattr(out, member_names[(n + 3)%member_count], getattr(vec, member_names[m]))
	# 					with false:
	# 						fourth = builder.icmp_signed('==', n, four)
	# 						with builder.if_else(fourth) as (true, false):
	# 							with true:
	# 								for m in range(member_count):
	# 									setattr(out, member_names[(n + 4)%member_count], getattr(vec, member_names[m]))
	# 							with false:
	# return out

	# else_ = builder.append_basic_block("shift.else")
	# end_ = builder.append_basic_block("shift.end")
	# switch = builder.switch(scalar, else_)

	# with 

	# loop(scalar)
	# 	out.x = vec.z
	# 	out.y = vec.x
	# 	out.z = vec.y
	# 	vec = out

	# # builder.icmp_signed()
	# # with builder.if_then():

	# # 	with true:
	# # 		pass
	# # 	with false:
	# # 		pass

	# # add, mul, sqrt = ops
	# # val = mul(builder, [vec[0], vec[0]])
	# # for n in range(len(vec) - 1):
	# # 	val = add(builder, [val, mul(builder, [vec[n + 1], vec[n + 1]])])
	# # return sqrt(builder, [val])

########################
#binary helper functions#
########################

def shift_impl(context, builder, sig, out, vec, n, member_current):
	member_names = out._fe_type._fields
	member_count = len(member_names)

	idx = context.get_constant(types.int8, np.int8(member_current))

	if member_current < member_count:
		with builder.if_else(builder.icmp_unsigned('==', n, idx)) as (true, false):
			with true:
				for m in range(member_count):
					setattr(out, member_names[(m + member_current)%member_count], vec[m])
			with false:
				shift_impl(context, builder, sig, out, vec, n, member_current + 1)
	return out

def shift_helper(context, builder, sig, args):
	# scalar = args[1]
	# zero = context.get_constant(scalar.type, 0)
	# one = context.get_constant(scalar.type, 1)
	# two = context.get_constant(scalar.type, 2)
	# three = context.get_constant(scalar.type, 3)
	# four = context.get_constant(scalar.type, 4)

	# libsig = signature(args[1].type, args[1].type, args[1].type)
	# minimum = context.get_function(min, libsig)
	# maximum = context.get_function(max, libsig)


	# def shift_caller(ops, out, vec, scalar):
	# 	#scalar.type

	# 	libsig = signature(types.int32, types.int32, types.int32)
	# 	minimum = context.get_function(min, libsig)
	# 	maximum = context.get_function(max, libsig)

	# 	member_names = out._fe_type._fields
	# 	member_count = len(member_names)
	# 	scalar = context.cast(builder, scalar, sig.args[1], types.int32)

	# 	#clip scalar to length of vec
	# 	zero = context.get_constant(types.int32, 0)
	# 	length = context.get_constant(types.int32, member_count - 1)
	# 	n = minimum(builder, [length, maximum(builder, [zero, scalar])])
	# 	return shift_impl(context, builder, out, vec, scalar, n, 0)._getvalue()

	def shift_caller(out, vec, n):
		modulo = context.get_function(operator.mod, signature(types.int8, types.int8, types.int8))

		member_names = out._fe_type._fields
		member_count = len(member_names)

		n = context.cast(builder, n, sig.args[1], types.int8)
		length = context.get_constant(types.int8, member_count)

		# out.x = context.cast(builder, n, types.uint32, types.float32)
		# out.y = context.cast(builder, n, types.uint32, types.float32)
		# out.z = context.cast(builder, n, types.uint32, types.float32)

		n = modulo(builder, [n, length])

		out = shift_impl(context, builder, sig, out, vec, n, 0)

		return out._getvalue()
	return shift_caller














# def add_helper(context, builder, sig, args):
	# add = builder.add
	# if isinstance(sig.return_type._member_t, types.Float):
	# 	add = builder.fadd
	# return add, builder_op_caller

# def sub_helper(context, builder, sig, args):
	# sub = builder.sub
	# if isinstance(sig.return_type._member_t, types.Float):
	# 	sub = builder.fsub
	# return sub, builder_op_caller

# def mul_helper(context, builder, sig, args):
	# mul = builder.mul
	# if isinstance(sig.return_type._member_t, types.Float):
	# 	mul = builder.fmul
	# return mul, builder_op_caller

# def div_helper(context, builder, sig, args):
	# div = builder.udiv
	# if isinstance(sig.return_type._member_t, types.Float):
	# 	div = builder.fdiv
	# elif sig.return_type._member_t._member_t.signed:
	# 	div = builder.sdiv
	# return div, builder_op_caller




# def mod_helper(context, builder, sig, args):
	# libfunc_impl = context.get_function(cuda_op, sig)

	# # mod = 

	# # 	for n, attr in enumerate(vec_t._fields):
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


#############
#TERNARY OPS#
#############

def ternary_op_factory_vec_vec_vec(op, vec_t, op_helper):
	@impl_registry.lower(op, vec_t, vec_t, vec_t)
	def cuda_op_vec_vec_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		b = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
		c = cgutils.create_struct_proxy(sig.args[2], 'data')(context, builder, ref = args[2])

		vec_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [getattr(a, attr) for attr in sig.return_type._fields]
		b_members = [getattr(b, attr) for attr in sig.return_type._fields]
		c_members = [getattr(c, attr) for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members, c_members)
		return vec_ptr

#note: mtv = member to vector (converts a scalar to a vector containing the scalar for every entry)
def ternary_op_factory_vec_vec_mtv(op, vec_t, op_helper):
	@impl_registry.lower(op, vec_t, vec_t, types.Number)
	def cuda_op_vec_vec_mtv(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		b = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
		c = context.cast(builder, args[2], sig.args[2], vec_t._member_t)

		vec_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [getattr(a, attr) for attr in sig.return_type._fields]
		b_members = [getattr(b, attr) for attr in sig.return_type._fields]
		c_members = [c for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members, c_members)
		return vec_ptr

	@impl_registry.lower(op, vec_t, types.Number, vec_t)
	def cuda_op_vec_mtv_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		b = context.cast(builder, args[1], sig.args[1], vec_t._member_t)
		c = cgutils.create_struct_proxy(sig.args[2], 'data')(context, builder, ref = args[2])

		vec_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [getattr(a, attr) for attr in sig.return_type._fields]
		b_members = [b for attr in sig.return_type._fields]
		c_members = [getattr(c, attr) for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members, c_members)
		return vec_ptr

	@impl_registry.lower(op, types.Number, vec_t, vec_t)
	def cuda_op_mtv_vec_vec(context, builder, sig, args):
		a = context.cast(builder, args[0], sig.args[0], vec_t._member_t)
		b = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
		c = cgutils.create_struct_proxy(sig.args[2], 'data')(context, builder, ref = args[2])

		vec_ptr = cgutils.alloca_once(builder, args[1].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [a for attr in sig.return_type._fields]
		b_members = [getattr(b, attr) for attr in sig.return_type._fields]
		c_members = [getattr(c, attr) for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members, c_members)
		return vec_ptr

def ternary_op_factory_vec_mtv_mtv(op, vec_t, op_helper):
	@impl_registry.lower(op, vec_t, types.Number, types.Number)
	def cuda_op_vec_mtv_mtv(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		b = context.cast(builder, args[1], sig.args[1], vec_t._member_t)
		c = context.cast(builder, args[2], sig.args[2], vec_t._member_t)

		vec_ptr = cgutils.alloca_once(builder, args[0].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [getattr(a, attr) for attr in sig.return_type._fields]
		b_members = [b for attr in sig.return_type._fields]
		c_members = [c for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members, c_members)
		return vec_ptr

	@impl_registry.lower(op, types.Number, vec_t, types.Number)
	def cuda_op_mtv_vec_mtv(context, builder, sig, args):
		a = context.cast(builder, args[0], sig.args[0], vec_t._member_t)
		b = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
		c = context.cast(builder, args[2], sig.args[2], vec_t._member_t)

		vec_ptr = cgutils.alloca_once(builder, args[1].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [a for attr in sig.return_type._fields]
		b_members = [getattr(b, attr) for attr in sig.return_type._fields]
		c_members = [c for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members, c_members)
		return vec_ptr

	@impl_registry.lower(op, types.Number, types.Number, vec_t)
	def cuda_op_mtv_mtv_vec(context, builder, sig, args):
		a = context.cast(builder, args[0], sig.args[0], vec_t._member_t)
		b = context.cast(builder, args[1], sig.args[1], vec_t._member_t)
		c = cgutils.create_struct_proxy(sig.args[2], 'data')(context, builder, ref = args[2])

		vec_ptr = cgutils.alloca_once(builder, args[2].type.pointee, size = 1)
		vec = cgutils.create_struct_proxy(sig.return_type, 'data')(context, builder, ref = vec_ptr)

		a_members = [a for attr in sig.return_type._fields]
		b_members = [b for attr in sig.return_type._fields]
		c_members = [getattr(c, attr) for attr in sig.return_type._fields]

		op, caller = op_helper(context, builder, sig, args)
		caller(op, builder, vec, a_members, b_members, c_members)
		return vec_ptr

def ternary_op_factory(op, vec_t, op_helper): #sig
	ternary_op_factory_vec_vec_vec(op, vec_t, op_helper)
	ternary_op_factory_vec_vec_mtv(op, vec_t, op_helper)
	ternary_op_factory_vec_mtv_mtv(op, vec_t, op_helper)

	# # @impl_registry.lower(op, vec_t, vec_t, vec_t)
	# # @impl_registry.lower(op, vec_t, vec_t, member_type)
	# # @impl_registry.lower(op, vec_t, member_type, vec_t)
	# # @impl_registry.lower(op, member_type, vec_t, vec_t)
	# # @impl_registry.lower(op, vec_t, member_type, member_type)
	# # @impl_registry.lower(op, member_type, vec_t, member_type)
	# # @impl_registry.lower(op, member_type, member_type, vec_t)
	# def cuda_op_ternary(context, builder, sig, args):
	# 	pointee = None
	# 	vals = []
	# 	members = []
	# 	for typ, arg in zip(sig.args, args):
	# 		if typ == vec_t:
	# 			vals.append(cgutils.create_struct_proxy(typ)(context, builder, ref = arg))
	# 			members.append([getattr(val, attr) for attr in arg._fields])
	# 			if not pointee:
	# 				pointee = arg.type.pointee
	# 		else:
	# 			vals.append(context.cast(builder, arg, typ, member_type))
	# 			members.append([val for attr in sig.return_type._fields])

	# 	vec_ptr = cgutils.alloca_once(builder, pointee, size = 1)
	# 	vec = cgutils.create_struct_proxy(sig.return_type)(context, builder, ref = vec_ptr)

	# 	op, caller = op_helper(context, builder, sig, args)
	# 	caller(op, builder, vec, *members)
	# 	return vec_ptr

#########################
#ternary callers functions#
#########################


########################
#ternary helper functions#
########################


###############
#REDUCTION OPS# (ops that take in a vector and return a scalar)
###############

def reduction_op_factory(op, vec_t, op_helper):
	#functions that reduce a vector input to a scalar output by applying op to each element of the input vector
	@impl_registry.lower(op, vec_t)
	def cuda_op_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		a_members = [getattr(a, attr) for attr in sig.args[0]._fields]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder, a_members)

#############################
#reduction callers functions#
#############################

def reduction_builder_op_caller(op, builder, vec):
	val = vec[0]
	for n in range(len(vec) - 1):
		val = op(val, vec[n + 1])
	return val

def reduction_libfunc_op_caller(op, builder, vec):
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

#######################################
#reduction op helper functions#
#######################################

def sum_helper(context, builder, sig, args):
	# add = builder.add
	# if isinstance(sig.args[0]._member_t, types.Float):
	# 	add = builder.fadd
	# return add, reduction_builder_op_caller

	libsig = signature(sig.args[0]._member_t,
					   sig.args[0]._member_t,
					   sig.args[0]._member_t)
	add = context.get_function(operator.add, libsig)

	return add, reduction_libfunc_op_caller

def length_helper(context, builder, sig, args):
	libsig = signature(sig.args[0]._member_t,
					   sig.args[0]._member_t)
	sqrt = context.get_function(mathfuncs.sqrt, libsig)

	libsig = signature(sig.args[0]._member_t,
					   sig.args[0]._member_t,
					   sig.args[0]._member_t)
	mul = context.get_function(operator.mul, libsig)
	add = context.get_function(operator.add, libsig)

	return [add, mul, sqrt], length_caller

def reduction_libfunc_helper_wrapper(cuda_op):
	def reduction_libfunc_helper(context, builder, sig, args):
		sig = signature(sig.args[0]._member_t,
						sig.args[0]._member_t,
						sig.args[0]._member_t)
		return context.get_function(cuda_op, sig), reduction_libfunc_op_caller
	return reduction_libfunc_helper



# def getitem_vec_factory(vec_t):
# 	#numba/numba/cpython/tupleobj.py 
# 		#def getitem_unituple(context, builder, sig, args):

# 	@impl_registry.lower(operator.getitem, vec_t, types.Integer)
# 	def vec_getitem(context, builder, sig, args):
# 		vec = cgutils.create_struct_proxy(sig.args[0])(context, builder, value = args[0])
# 		idx = args[1]

# 		# vec_len = len(vec_t._fields)

# 		# zero = const_zero(context)
# 		# one = const_nan(context)
# 		# one_64 = context.get_constant(types.float64, np.float64(1))

# 		# negative = builder.fcmp_unordered('<', idx, zero)
# 		# with builder.if_else(negative) as (true, false):

# 		return vec.x

##################
#VECTOR FUNCTIONS#
##################

from . import vectorfuncs
from ..mathdecl import register_op, unary_op_fixed_type_factory, binary_op_fixed_type_factory

#inject/insert functions into mathfuncs if they don't already exist (because functions aren't typed only a single instance of a function with the given name needs to exist within core)
vec_funcs = [(vectorfuncs.sum, unary_op_fixed_type_factory, None),
			 (vectorfuncs.dot, binary_op_fixed_type_factory, None),
			 (vectorfuncs.length, unary_op_fixed_type_factory, None),
			 (vectorfuncs.shift, binary_op_fixed_type_factory, None)]

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
	# typ_a.is_internal
		#use this to decend until an internal type is found


	# double_t = ir.DoubleType()
	# __ir_member_t__
	# ir_vec_t = ir.LiteralStructType([double_t, double_t])


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
			def typer(vec_or_input_t_a, vec_or_input_t_b):
				if isinstance(vec_or_input_t_a, vec_t.__input_type__) and isinstance(vec_or_input_t_b, Vector2Type):
					return vec_t
				elif isinstance(vec_or_input_t_a, Vector2Type) and isinstance(vec_or_input_t_b, vec_t.__input_type__):
					return vec_t
				else:
					raise ValueError(f"Input to {vec.__name__} not understood")
			return typer

	#initialize: vec(x) or vec(vec)
	@type_callable(vec)
	def type_vec(context):
		def typer(vec_or_input_t):
			if isinstance(vec_or_input_t, vec_t.__input_type__) or isinstance(vec_or_input_t, Vec):
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

	#initialize: vec(vec)
	@lower_builtin(vec, Vec)
	def impl_vec(context, builder, sig, args):
		vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])

		OutProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
		datamodel = context.data_model_manager[OutProxy._fe_type]
		out_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
		out = OutProxy(context, builder, ref = out_ptr)

		for n, attr in enumerate(datamodel._fields):
			setattr(out, attr, context.cast(builder, getattr(vec, attr), sig.args[0]._member_t, sig.return_type._member_t))

		return out_ptr

	#initialize: vec(x,y,...)
	@lower_builtin(vec, *[vec_t.__input_type__]*len(vec_t._members))
	def impl_vec(context, builder, sig, args):
		OutProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
		datamodel = context.data_model_manager[OutProxy._fe_type]
		out_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
		out = OutProxy(context, builder, ref = out_ptr)

		# context._get_constants.find((types.float32,))

		for n, attr in enumerate(datamodel._fields):
			setattr(vec, attr, context.cast(builder, args[n], sig.args[n], sig.return_type._member_t))

		return out_ptr

	#initialize: vec(x)
	@lower_builtin(vec, vec_t.__input_type__)
	def impl_vec(context, builder, sig, args):
		OutProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
		datamodel = context.data_model_manager[OutProxy._fe_type]
		out_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
		out = OutProxy(context, builder, ref = out_ptr)

		for n, attr in enumerate(datamodel._fields):
			setattr(vec, attr, context.cast(builder, args[0], sig.args[0], sig.return_type._member_t))

		return out_ptr

	#initialize: vec()
	@lower_builtin(vec)
	def impl_vec(context, builder, sig, args):
		OutProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
		datamodel = context.data_model_manager[OutProxy._fe_type]
		out_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
		out = OutProxy(context, builder, ref = out_ptr)

		for n, attr in enumerate(datamodel._fields):
			setattr(vec, attr, context.get_constant(sig.return_type._member_t, 0))

		return out_ptr

	if Vec == Vector3Type:
		#initialize: vec(x, vec2)
		@lower_builtin(vec, vec_t.__input_type__, Vector2Type)
		def impl_vec(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
			vals = [context.cast(builder, args[0], sig.args[0], sig.return_type._member_t),
					context.cast(builder, vec.x, sig.args[1]._member_t, sig.return_type._member_t),
					context.cast(builder, vec.y, sig.args[1]._member_t, sig.return_type._member_t)]

			OutProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
			datamodel = context.data_model_manager[OutProxy._fe_type]
			out_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
			out = OutProxy(context, builder, ref = out_ptr)

			for n, attr in enumerate(datamodel._fields):
				setattr(out, attr, vals[n])

			return out_ptr

		#initialize: vec(vec2, x)
		@lower_builtin(vec, Vector2Type, vec_t.__input_type__)
		def impl_vec(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
			vals = [context.cast(builder, vec.x, sig.args[0]._member_t, sig.return_type._member_t),
					context.cast(builder, vec.y, sig.args[0]._member_t, sig.return_type._member_t),
					context.cast(builder, args[1], sig.args[1], sig.return_type._member_t)]

			OutProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
			datamodel = context.data_model_manager[OutProxy._fe_type]
			out_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
			out = OutProxy(context, builder, ref = out_ptr)

			for n, attr in enumerate(datamodel._fields):
				setattr(out, attr, vals[n])

			return out_ptr

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

			XYProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
			datamodel = context.data_model_manager[XYProxy._fe_type]
			xy_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
			xy = XYProxy(context, builder, ref = xy_ptr)

			for n, attr in enumerate(['x', 'y']):
				setattr(xy, attr, getattr(vec, attr))

			return xy_ptr

		@impl_registry.lower_setattr(vec_t, 'xy')
		def vec3_set_xy(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
			xy = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
			vec.x = xy.x
			vec.y = xy.y

		@impl_registry.lower_getattr(vec_t, 'xz')
		def vec3_get_xz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig, 'data')(context, builder, ref = args)

			XZProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
			datamodel = context.data_model_manager[XZProxy._fe_type]
			xz_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
			xz = XZProxy(context, builder, ref = xz_ptr)

			for attr2, attr3 in zip(['x', 'y'], ['x', 'z']):
				setattr(xz, attr2, getattr(vec, attr3))

			return xz_ptr

		@impl_registry.lower_setattr(vec_t, 'xz')
		def vec3_set_xz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
			xz = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
			vec.x = xz.x
			vec.z = xz.y

		@impl_registry.lower_getattr(vec_t, 'yz')
		def vec3_get_yz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig, 'data')(context, builder, ref = args)

			YZProxy = cgutils.create_struct_proxy(sig.return_type, 'data')
			datamodel = context.data_model_manager[YZProxy._fe_type]
			yz_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
			yz = YZProxy(context, builder, ref = yz_ptr)

			for attr2, attr3 in zip(['x', 'y'], ['y', 'z']):
				setattr(xz, attr2, getattr(vec, attr3))

			return yz_ptr

		@impl_registry.lower_setattr(vec_t, 'yz')
		def vec3_set_yz(context, builder, sig, args):
			vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
			yz = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
			vec.y = yz.x
			vec.z = yz.y

	##########################################
	#Register Vector Methods
	##########################################

	#####
	#+, -, *, /    # +=, -=, *=, /= (iadd,isub,imul,idiv)
	#####

	class vec_op_template(ConcreteTemplate):
		cases = [signature(vec_t, vec_t, vec_t),
				 signature(vec_t, vec_t._member_t, vec_t),
				 signature(vec_t, vec_t, vec_t._member_t)]

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
		   # operator.iadd, operator.isub, operator.imul, operator.itruediv]
	for op in ops:
		decl_registry.register_global(op)(vec_op_template)
		binary_op_factory(op, vec_t, libfunc_helper_wrapper(op))

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
	#min, max, sum
	#####

	class vec_op_template(ConcreteTemplate):
				 #binary
		cases = [signature(vec_t, vec_t, vec_t),
				 signature(vec_t, vec_t._member_t, vec_t),
				 signature(vec_t, vec_t, vec_t._member_t),
				 #reduction
				 signature(vec_t._member_t, vec_t)]

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
		cases = [signature(vec_t._member_t, vec_t, types.intp),
				 signature(vec_t._member_t, vec_t, types.uintp)]

	decl_registry.register_global(operator.getitem)(vec_getitem_template)
	# getitem_vec_factory(vec_t)

	#####
	#shift
	#####

	for scalar_type in [types.uint8, types.uint16, types.uint32, types.uint64, types.int8, types.int16, types.int32, types.int64]:
		binary_op_factory_vec_scalar(mathfuncs.shift, vec_t, shift_helper, scalar_type = scalar_type)


#vectorized ops
unary_floating_ops = ["sin", "cos", "tan",
					  "sinh", "cosh", "tanh",
					  "asin", "acos", "atan",
					  "asinh", "acosh", "atanh",
					  "sqrt", #"rsqrt", "cbrt", "rcbrt",
					  "exp" , "exp10", "exp2", #"expm1",
					  "log", "log10", "log2", #"log1p", "logb",
					  "floor", "ceil"]

binary_floating_ops = ["fmod"]

ternary_floating_ops = ["clamp", "lerp", "param", "smooth_step"]

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
	for op in unary_floating_ops:
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

	for op in binary_floating_ops:
		libop = getattr(mathfuncs, op)
		binary_op_factory(libop, vec_t, libfunc_helper_wrapper(libop))

	#non-vectorized ops
		# atan2, cross, etc.

	#####
	#ternary functions
	#####

	for op in ternary_floating_ops:
		libop = getattr(mathfuncs, op)
		ternary_op_factory(libop, vec_t, libfunc_helper_wrapper(libop))

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



def integer_vec_decl(vec, vec_t):
	class vec_op_template(ConcreteTemplate):
		cases = [signature(vec_t, vec_t, vec_t),
				 signature(vec_t, vec_t._member_t, vec_t),
				 signature(vec_t, vec_t, vec_t._member_t)]

		# cases = [signature(vec_t, vec_t, vec_t),
		# 		 signature(vec_t, vec_t._member_t, vec_t),
		# 		 signature(vec_t, vec_t, vec_t._member_t)]

	ops = [operator.and_, operator.or_]
	for op in ops:
		decl_registry.register_global(op)(vec_op_template)
		binary_op_factory(op, vec_t, libfunc_helper_wrapper(op))

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



vec2s = [char2, short2, int2, long2, uchar2, ushort2, uint2, ulong2]
vec3s = [char3, short3, int3, long3, uchar3, ushort3, uint3, ulong3]
vec2s_t = [char2_t, short2_t, int2_t, long2_t, uchar2_t, ushort2_t, uint2_t, ulong2_t]
vec3s_t = [char3_t, short3_t, int3_t, long3_t, char3_t, short3_t, int3_t, long3_t]

vec_groups = [vec2s, vec3s]
vec_t_groups = [vec2s_t, vec3s_t]

for vecs, vecs_t in zip(vec_groups, vec_t_groups):
	for vec, vec_t in zip(vecs, vecs_t):
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
