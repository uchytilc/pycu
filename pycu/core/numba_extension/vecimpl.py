from numba import types
from numba.core.extending import typeof_impl, type_callable
from numba.core.typing import signature

from numba.core.extending import models, register_model, lower_builtin
from numba.core import cgutils
from numba.cpython import mathimpl

from numba.cuda.cudadecl import registry as decl_registry
from numba.cuda.cudaimpl import registry as impl_registry
from numba.core.typing.templates import ConcreteTemplate, AttributeTemplate

from .vector.vectorimpl import allocate_vector

from .vecdecl import VecType, Vec2Type, Vec3Type
from .vecdecl import vec2_t, vec3_t
from .vector.vectortype import vec2, vec3

# from .vector.vectordecl import
from .vector.vectordecl import (float2_t, double2_t,
								float3_t, double3_t)
from .interval.intervaldecl import IntervalType #(IntervalfType, IntervaldType)
from .interval.intervaldecl import (intervalf2_t, intervald2_t,
									intervalf3_t, intervald3_t)

from llvmlite import ir

import operator

from . import mathfuncs

import numpy as np

#This class takes in any input and converts itself to the appropriate vector type based on these inputs
	#all supported input types are defined within the vec_typer methods
	#the impl_vec functions define how to construct a vec2 given specific inputs provided to the vec_typer method

def vec_2_typer(*attrs):
	#initialize: vec(x)
	if len(attrs) == 1:
		attr = attrs[0]
		if isinstance(attr, types.Number):
			if attr.bitwidth < 64:
				return float2_t
			else:
				return double2_t
		elif isinstance(attr, IntervalType):
			if attr._member_t.bitwidth < 64:
				return intervalf2_t
			else:
				return intervald2_t
	#initialize: vec2(x,y)
	elif len(attrs) == 2:
		if all([isinstance(attr, types.Number) for attr in attrs]):
			if all([attr.bitwidth < 64 for attr in attrs]):
				return float2_t
			else:
				return double2_t
		elif any([isinstance(attr, IntervalType) for attr in attrs]):
			#note: bitwidth of intervals takes precedence over scalar bitwidth
			bitwidth = max([attr._member_t.bitwidth if isinstance(attr, IntervalType) else 0 for attr in attrs])
			if bitwidth < 64:
				return intervalf2_t
			else:
				return intervald2_t
	raise ValueError(f"Input to {vec2_t.__class__.__name__} not understood")

def vec2_decl(): 
	@typeof_impl.register(vec2)
	def typeof_vec(val, c):
		return vec2_t

	@type_callable(vec2)
	def type_vec(context):
		def typer(x):
			return vec_2_typer(x)
		return typer

	#initialize: vec(x,y)
	@type_callable(vec2)
	def type_vec2(context):
		def typer(x,y):
			return vec_2_typer(x,y)
		return typer

	@lower_builtin(vec2, types.Number)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			member = cgutils.create_struct_proxy(sig.return_type._member_t)(context, builder)
			member.lo = context.cast(builder, args[0], sig.args[0], sig.return_type._member_t._member_t)
			member.hi = context.cast(builder, args[0], sig.args[0], sig.return_type._member_t._member_t)
			setattr(out, attr, member._getvalue())

		return out._getpointer()

	@lower_builtin(vec2, IntervalType)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, args[0]) # setattr(out, attr, context.cast(builder, args[n], sig.args[n], sig.return_type._member_t))

		return out._getpointer()

	@lower_builtin(vec2, types.Number, types.Number)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, context.cast(builder, args[n], sig.args[n], sig.return_type._member_t))

		return out._getpointer()

	@lower_builtin(vec2, types.Number, IntervalType)
	@lower_builtin(vec2, IntervalType, types.Number)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			member = cgutils.create_struct_proxy(sig.return_type._member_t)(context, builder)
			if isinstance(sig.args[n], types.Number):
				member.lo = context.cast(builder, args[n], sig.args[n], sig.return_type._member_t._member_t)
				member.hi = context.cast(builder, args[n], sig.args[n], sig.return_type._member_t._member_t)
			else:
				member = cgutils.create_struct_proxy(sig.return_type._member_t)(context, builder, value = args[n])
			setattr(out, attr, member._getvalue())

		return out._getpointer()

	@lower_builtin(vec2, IntervalType, IntervalType)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			arg = cgutils.create_struct_proxy(sig.args[n])(context, builder, value = args[n])
			member = cgutils.create_struct_proxy(sig.return_type._member_t)(context, builder)
			member.lo = context.cast(builder, arg.lo, sig.args[n]._member_t, sig.return_type._member_t._member_t)
			member.hi = context.cast(builder, arg.hi, sig.args[n]._member_t, sig.return_type._member_t._member_t)
			setattr(out, attr, member._getvalue())

		return out._getpointer()

def vec_3_typer(*attrs):
	#initialize: vec(x)
	if len(attrs) == 1:
		attr = attrs[0]
		if isinstance(attr, types.Number):
			if attr.bitwidth < 64:
				return float3_t
			else:
				return double3_t
		elif isinstance(attr, IntervalType):
			if attr._member_t.bitwidth < 64:
				return intervalf3_t
			else:
				return intervald3_t
	#initialize: vec3(x,y,z)
	elif len(attrs) == 3:
		if all([isinstance(attr, types.Number) for attr in attrs]):
			if all([attr.bitwidth < 64 for attr in attrs]):
				return float3_t
			else:
				return double3_t
		elif any([isinstance(attr, IntervalType) for attr in attrs]):
			#note: bitwidth of intervals takes precedence over scalar bitwidth
			bitwidth = max([attr._member_t.bitwidth if isinstance(attr, IntervalType) else 0 for attr in attrs])
			if bitwidth < 64:
				return intervalf3_t
			else:
				return intervald3_t

	raise ValueError(f"Input to {vec3_t.__class__.__name__} not understood")

def vec3_decl():
	@typeof_impl.register(vec3)
	def typeof_vec(val, c):
		return vec3_t

	@type_callable(vec3)
	def type_vec(context):
		def typer(x):
			return vec_3_typer(x)
		return typer

	#initialize: vec(x,y,x)
	@type_callable(vec3)
	def type_vec3(context):
		def typer(x,y,z):
			return vec_3_typer(x,y,z)
		return typer

	@lower_builtin(vec3, types.Number)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, args[0])

		return out._getpointer()

	@lower_builtin(vec3, IntervalType)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, args[0]) # setattr(out, attr, context.cast(builder, args[n], sig.args[n], sig.return_type._member_t))

		return out._getpointer()

	@lower_builtin(vec3, types.Number, types.Number, types.Number)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			setattr(out, attr, context.cast(builder, args[n], sig.args[n], sig.return_type._member_t))

		return out._getpointer()

	@lower_builtin(vec3, types.Number, types.Number, IntervalType)
	@lower_builtin(vec3, types.Number, IntervalType, types.Number)
	@lower_builtin(vec3, IntervalType, types.Number, types.Number)
	@lower_builtin(vec3, IntervalType, IntervalType, types.Number)
	@lower_builtin(vec3, IntervalType, types.Number, IntervalType)
	@lower_builtin(vec3, types.Number, IntervalType, IntervalType)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			if isinstance(sig.args[n], types.Number):
				member = cgutils.create_struct_proxy(sig.return_type._member_t)(context, builder)
				member.lo = context.cast(builder, args[n], sig.args[n], sig.return_type._member_t._member_t)
				member.hi = context.cast(builder, args[n], sig.args[n], sig.return_type._member_t._member_t)
			else:
				#cast interval members as the output might be a dtype with a larger bitwidth
				arg = cgutils.create_struct_proxy(sig.args[n])(context, builder, value = args[n])
				member = cgutils.create_struct_proxy(sig.return_type._member_t)(context, builder, value = args[n])
				member.lo = context.cast(builder, arg.lo, sig.args[n]._member_t, sig.return_type._member_t._member_t)
				member.hi = context.cast(builder, arg.hi, sig.args[n]._member_t, sig.return_type._member_t._member_t)
			setattr(out, attr, member._getvalue())

		return out._getpointer()

	@lower_builtin(vec3, IntervalType, IntervalType, IntervalType)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			arg = cgutils.create_struct_proxy(sig.args[n])(context, builder, value = args[n])
			member = cgutils.create_struct_proxy(sig.return_type._member_t)(context, builder)
			member.lo = context.cast(builder, arg.lo, sig.args[n]._member_t, sig.return_type._member_t._member_t)
			member.hi = context.cast(builder, arg.hi, sig.args[n]._member_t, sig.return_type._member_t._member_t)
			setattr(out, attr, member._getvalue())

		return out._getpointer()

vec2_decl()
vec3_decl()
