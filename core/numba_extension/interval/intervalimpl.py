from numba import types
from numba.core.extending import typeof_impl, type_callable
from numba.core.typing import signature

from numba.core.extending import models, register_model, lower_builtin
from numba.core import cgutils

from numba.cuda.cudadecl import registry as decl_registry
from numba.cuda.cudaimpl import registry as impl_registry
from numba.core.typing.templates import ConcreteTemplate, AttributeTemplate

import operator
from .intervaltype import intervalf, intervald
from .intervaldecl import intervalf_t, intervald_t

from .. import mathfuncs

import numpy as np

##########################################
#Define Interval Type
##########################################

def interval_factory(interval, interval_t):
	@typeof_impl.register(interval)
	def typeof_interval(val, c):
		return interval_t

	@type_callable(interval)
	def type_interval(context):
		def typer(lo, hi):
			if all([isinstance(attr, types.Number) for attr in [lo, hi]]):
				return interval_t
			else:
				raise ValueError(f"Input to {interval.__name__} not understood")
		return typer

	IntervalType = type(interval_t)
	@register_model(IntervalType) 
	class IntervalModel(models.StructModel):
		def __init__(self, dmm, fe_type):
			members = [("lo", interval_t.__member_type__),
					   ("hi", interval_t.__member_type__)]
			models.StructModel.__init__(self, dmm, fe_type, members)

	##########################################
	#Initializer/Constructor Methods
	##########################################

	@lower_builtin(interval, types.Number, types.Number)
	def impl_interval(context, builder, sig, args):
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
		out.lo = context.cast(builder, args[0], sig.args[0], interval_t.__member_type__)
		out.hi = context.cast(builder, args[1], sig.args[1], interval_t.__member_type__)
		return out._getvalue()

	##########################################
	#Define Interval Attributes
	##########################################

	@decl_registry.register_attr
	class Interval_attrs(AttributeTemplate):
		key = interval_t

		def resolve_lo(self, mod):
			return interval_t.__member_type__

		def resolve_hi(self, mod):
			return interval_t.__member_type__

	@impl_registry.lower_getattr(interval_t, 'lo')
	def interval_get_lo(context, builder, sig, args):
		interval = cgutils.create_struct_proxy(sig)(context, builder, value=args)
		return interval.lo

	@impl_registry.lower_getattr(interval_t, 'hi')
	def interval_get_hi(context, builder, sig, args):
		interval = cgutils.create_struct_proxy(sig)(context, builder, value=args)
		return interval.hi

	if interval == intervalf:
		const_pi  = constant(interval_t.__member_type__, np.float32(np.pi))
		const_nan = constant(interval_t.__member_type__, np.float32(np.nan))
		const_inf = constant(interval_t.__member_type__, np.float32(np.inf))
		const_zero = constant(interval_t.__member_type__, np.float32(0))
		const_one = constant(interval_t.__member_type__, np.float32(1))
		const_n_one = constant(interval_t.__member_type__, np.float32(-1))

		cast = lambda context, builder, val : context.cast(builder, val, types.float32, types.float64)

		op_rd = op_rd_32
		op_ru = op_ru_32
		val_rd = val_rd_32
		val_ru = val_ru_32

	elif interval == intervald:
		const_pi  = constant(interval_t.__member_type__, np.float64(np.pi))
		const_nan = constant(interval_t.__member_type__, np.float64(np.nan))
		const_inf = constant(interval_t.__member_type__, np.float64(np.inf))
		const_zero = constant(interval_t.__member_type__, np.float64(0))
		const_one = constant(interval_t.__member_type__, np.float64(1))
		const_n_one = constant(interval_t.__member_type__, np.float64(-1))

		cast = lambda context, builder, val : val

		op_rd = op_rd_64
		op_ru = op_ru_64
		val_rd = val_rd_64
		val_ru = val_ru_64

	def unary_op_interval(op, caller):
		@impl_registry.lower(op, interval_t)
		def cuda_op_interval_interval(context, builder, sig, args):
			x = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			y = cgutils.create_struct_proxy(interval_t)(context, builder)
			return caller(context, builder, sig, args, x, y)._getvalue()

	class unary_op_template(ConcreteTemplate):
		cases = [signature(interval_t, interval_t)]

	def unary_impl(f):
		def unary_interval(context, builder, sig, args, *intervals):
			op = context.get_function(f, signature(interval_t.__member_type__, interval_t.__member_type__))
			def unary(x,y):
				y.lo = op(builder, [x.lo])
				y.hi = op(builder, [x.hi])
				return y
			return unary(*intervals)
		return unary_interval

	def neg_interval(context, builder, sig, args, *intervals):
		fneg = context.get_function(operator.neg, signature(interval_t.__member_type__, interval_t.__member_type__))
		def neg(x,y):
			y.lo = fneg(builder, [x.hi])
			y.hi = fneg(builder, [x.lo])
			return y
		return neg(*intervals)

	def abs_interval(context, builder, sig, args, *intervals):
		fabs = context.get_function(abs, signature(*([interval_t.__member_type__]*2)))
		maxf = context.get_function(max, signature(*([interval_t.__member_type__]*3)))
		fneg = context.get_function(operator.neg, signature(*([interval_t.__member_type__]*2)))
		def absolute(x, y):
			zero = const_zero(context)

			positive = builder.fcmp_unordered('>=', x.lo, zero)
			with builder.if_else(positive) as (true, false):
				with true:
					y.lo = x.lo
					y.hi = x.hi
				with false:
					negative = builder.fcmp_unordered('<', x.hi, zero)
					with builder.if_else(negative) as (true, false):
						with true:
							# neg_interval(context, builder, sig, args, x, y)
							y.lo = fneg(builder, [x.hi])
							y.hi = fneg(builder, [x.lo])
						with false:
							y.lo = zero
							y.hi = maxf(builder, [fabs(builder, [x.lo]), fabs(builder, [x.hi])])
			return y
		return absolute(*intervals)

	ops = [(operator.neg, neg_interval),
		   (abs, abs_interval),
		   (round, unary_impl(mathfuncs.round))]

	#fabsf
	#absf

	for op, caller in ops:
		decl_registry.register_global(op)(unary_op_template)
		unary_op_interval(op, caller)
		# print(op, interval_t)

	# unary_op_interval(mathfuncs.abs, abs_interval)

	for op in ['round', 'floor', 'ceil']:
		f1 = getattr(mathfuncs, op)
		if interval == intervalf:
			f2 = getattr(mathfuncs, op + 'f')
		else:
			f2 = f1
		unary_op_interval(f1, unary_impl(f2))

	def exp_factory(f):
		def exp_interval(context, builder, sig, args, *intervals):
			exp = context.get_function(f, signature(types.float64, types.float64))
			def exponential(x,y):
				#round ops for these functions do not exist so a cast/round method is used
				exp_lo = exp(builder, [cast(context, builder, x.lo)])
				exp_hi = exp(builder, [cast(context, builder, x.hi)])

				#value is rounded after op. For double, nextafter is used to force rounding as float128 isn't supported
				y.lo = val_rd(context, builder, exp_lo)
				y.hi = val_ru(context, builder, exp_hi)
				return y
			return exponential(*intervals)
		return exp_interval

	for ext in ['', '10', '2']:  #'m1' -> expm1 = exp(x) - 1
		f = getattr(mathfuncs, f'exp{ext}')
		unary_op_interval(f, exp_factory(f))
	# expf
	# exp10f
	# exp2f

	def log_factory(f):
		def log_interval(context, builder, sig, args, *intervals):
			log = context.get_function(f, signature(types.float64, types.float64))
			nan = const_nan(context)
			zero = const_zero(context)
			def logarithm(x,y):
				negative = builder.fcmp_unordered('<', x.hi, zero)
				with builder.if_else(negative) as (true, false):
					with true:
						y.lo = nan
						y.hi = nan
					with false:
						split = builder.fcmp_unordered('<=', x.lo, zero)
						with builder.if_else(split) as (true, false):
							with true:
								y.lo = zero
								y.hi = val_ru(context, builder, log(builder, [cast(context, builder, x.hi)]))
							with false:
								log_lo = log(builder, [cast(context, builder, x.lo)])
								log_hi = log(builder, [cast(context, builder, x.hi)])

								y.lo = val_rd(context, builder, log_lo)
								y.hi = val_ru(context, builder, log_hi)
				return y
			return logarithm(*intervals)
		return log_interval

	for ext in ['', '10', '2']: #1p -> log1p = log(1 + x)
		f = getattr(mathfuncs, f'log{ext}')
		unary_op_interval(f, log_factory(f))
	# logf
	# log10f
	# log2f

	def sqrt_interval(context, builder, sig, args, *intervals):
		def sqrt(x,y):
			zero = const_zero(context)
			nan = const_nan(context)

			sqrt_rd = op_rd(context, 'sqrt', signature(*([sig.args[0].__member_type__]*2)), mathfuncs)
			sqrt_ru = op_ru(context, 'sqrt', signature(*([sig.args[0].__member_type__]*2)), mathfuncs)

			negative = builder.fcmp_unordered('<', x.hi, zero)
			with builder.if_else(negative) as (true, false):
				with true:
					y.lo = nan
					y.hi = nan
				with false:
					split = builder.fcmp_unordered('<=', x.lo, zero)
					with builder.if_else(split) as (true, false):
						with true:
							y.lo = zero
							y.hi = sqrt_ru(builder, [x.hi])
						with false:
							y.lo = sqrt_rd(builder, [x.lo])
							y.hi = sqrt_ru(builder, [x.hi])
			return y
		return sqrt(*intervals)

	unary_op_interval(mathfuncs.sqrt, sqrt_interval)
	#sqrtf
	#rsqrt
	#cbrt
	#rcbrt

	# def sin_interval(context, builder, sig, args, *intervals):
		# const = context.get_constant(types.float32, np.pi/2)
		# z = cgutils.create_struct_proxy(intervalf_type)(context, builder)

		# y = neg_interval(context, builder, sig, args, *intervals)
		# y = add_interval_scalar(context, builder, sig, args, y, const, y)
		# return cos_interval(context, builder, sig, args, y, z)

	# def cos_interval(context, builder, sig, args, *intervals):
		# #Self-Validated Numerical Methods and Applications page 31

		# double2float_rd = context.get_function(mathfuncs.double2float_rd, signature(types.float32, types.float64))
		# double2float_ru = context.get_function(mathfuncs.double2float_ru, signature(types.float32, types.float64))
		# float2int_rd = context.get_function(mathfuncs.float2int_rd, signature(types.int32, types.float32))
		# float2int_ru = context.get_function(mathfuncs.float2int_ru, signature(types.int32, types.float32))
		# fdiv_ru = context.get_function(mathfuncs.fdiv_ru, signature(types.float32, types.float32, types.float32))
		# fdiv_rd = context.get_function(mathfuncs.fdiv_rd, signature(types.float32, types.float32, types.float32))
		# fmaxf = context.get_function(mathfuncs.fmaxf, signature(types.float32, types.float32, types.float32))
		# fminf = context.get_function(mathfuncs.fminf, signature(types.float32, types.float32, types.float32))

		# fsub_ru = context.get_function(mathfuncs.fsub_ru, signature(types.float32, types.float32, types.float32))
		# sub = context.get_function(operator.sub, signature(types.int32, types.int32, types.int32))

		# def is_even(x):
		# 	one = context.get_constant(types.int32,  1)
		# 	return context.cast(builder, builder.and_(x, one), types.int32, types.boolean) 

		# def cos(x,y):
		# 	cos = context.get_function(mathfuncs.cos, signature(types.float64, types.float64))

		# 	zero = context.get_constant(types.float32, 0)

		# 	one_ = context.get_constant(types.float32, -1)
		# 	one  = context.get_constant(types.float32,  1)

		# 	pi2_lo = context.get_constant(types.float32, np.int32(0x40c90fda).view(np.float32))
		# 	# pi2_hi = context.get_constant(types.float32, np.int32(0x40c90fdb).view(np.float32))
		# 	# pi3_hi = context.get_constant(types.float32, np.int32(0x4116cbe4).view(np.float32))
		# 	width = fsub_ru(builder, [x.hi, x.lo])
		# 	with builder.if_else(builder.fcmp_unordered('>', width, pi2_lo)) as (true, false):
		# 		with true:
		# 			y.lo = one_
		# 			y.hi = one
		# 		with false:
		# 			a = cgutils.alloca_once_value(builder, zero)
		# 			b = cgutils.alloca_once_value(builder, zero)

		# 			pi_lo = context.get_constant(types.float32, np.int32(0x40490fda).view(np.float32))
		# 			pi_hi = context.get_constant(types.float32, np.int32(0x40490fdb).view(np.float32))


		# 			cos_lo = cos(builder, [context.cast(builder, x.lo, types.float32, types.float64)])
		# 			cos_hi = cos(builder, [context.cast(builder, x.hi, types.float32, types.float64)])

		# 			with builder.if_else(builder.fcmp_unordered('>', x.lo, zero)) as (true, false):
		# 				with true:
		# 					builder.store(fdiv_rd(builder, [x.lo, pi_hi]), a)
		# 				with false:
		# 					builder.store(fdiv_rd(builder, [x.lo, pi_lo]), a)

		# 			with builder.if_else(builder.fcmp_unordered('>', x.hi, zero)) as (true, false):
		# 				with true:
		# 					builder.store(fdiv_ru(builder, [x.hi, pi_lo]), b)
		# 				with false:
		# 					builder.store(fdiv_ru(builder, [x.hi, pi_hi]), b)

		# 			m = float2int_rd(builder, [builder.load(a)])
		# 			n = float2int_ru(builder, [builder.load(b)])
		# 			periods = sub(builder, [n, m])

		# 			two = context.get_constant(types.int32, 2)
		# 			with builder.if_else(builder.icmp_signed('<', periods, two)) as (true, false):
		# 				with true:
		# 					with builder.if_else(is_even(m)) as (odd, even):
		# 						with even:
		# 							y.lo = double2float_rd(builder, [cos_lo])
		# 							y.hi = double2float_ru(builder, [cos_hi])
		# 						with odd:
		# 							y.lo = double2float_rd(builder, [cos_hi])
		# 							y.hi = double2float_ru(builder, [cos_lo])
		# 				with false:
		# 					with builder.if_else(builder.icmp_signed('==', periods, two)) as (true, false):
		# 						with true:
		# 							with builder.if_else(is_even(m)) as (odd, even):
		# 								with even:
		# 									y.lo = one_
		# 									y.hi = fmaxf(builder, [double2float_ru(builder, [cos_lo]), double2float_ru(builder, [cos_hi])])
		# 								with odd:
		# 									y.lo = fminf(builder, [double2float_rd(builder, [cos_lo]), double2float_rd(builder, [cos_hi])])
		# 									y.hi = one
		# 						#should never be reached
		# 						with false:
		# 							y.lo = one_
		# 							y.hi = one
		# 	return y
		# return cos(*intervals)

	# def tan_interval(context, builder, sig, args, *intervals):
		# def tan(x,y):
		# 	pass
		# return tan(*intervals)
		# # // get lower bound within [-pi/2, pi/2]
		# # const R pi = interval_lib::pi<R>();
		# # R tmp = fmod((const R&)x, pi);
		# # const T pi_half_d = interval_lib::constants::pi_half_lower<T>();
		# # if (tmp.lower() >= pi_half_d)
		# # 	tmp -= pi;
		# # if (tmp.lower() <= -pi_half_d || tmp.upper() >= pi_half_d)
		# # 	return I::whole();
		# # return I(rnd.tan_down(tmp.lower()), rnd.tan_up(tmp.upper()), true);

	def sinh_interval(context, builder, sig, args, *intervals):
		def sinh(x,y):
			sinh = context.get_function(mathfuncs.sinh, signature(types.float64, types.float64))

			sinh_lo = sinh(builder, [cast(context, builder, x.lo)])
			sinh_hi = sinh(builder, [cast(context, builder, x.hi)])

			y.lo = val_rd(context, builder, sinh_lo)
			y.hi = val_ru(context, builder, sinh_hi)
			return y
		return sinh(*intervals)

	def cosh_interval(context, builder, sig, args, *intervals):
		maxf = context.get_function(max, signature(*([interval_t.__member_type__]*3)))

		def cosh(x,y):
			zero = const_zero(context)
			one = const_one(context)

			cosh = context.get_function(mathfuncs.cosh, signature(types.float64, types.float64))

			cosh_lo = cosh(builder, [cast(context, builder, x.lo)])
			cosh_hi = cosh(builder, [cast(context, builder, x.hi)])

			with builder.if_else(builder.fcmp_unordered('<', x.hi, zero)) as (true, false):
				with true:
					y.lo = val_rd(context, builder, cosh_hi)
					y.hi = val_ru(context, builder, cosh_lo)
				with false:
					with builder.if_else(builder.fcmp_unordered('>', x.lo, zero)) as (true, false):
						with true:
							y.lo = val_rd(context, builder, cosh_lo)
							y.hi = val_ru(context, builder, cosh_hi)
						with false:
							y.lo = one
							y.hi = maxf(builder, val_ru(context, builder, cosh_lo), val_ru(context, builder, cosh_hi))
			return y
		return cosh(*intervals)

	def tanh_interval(context, builder, sig, args, *intervals):
		def tanh(x,y):
			tanh = context.get_function(mathfuncs.tanh, signature(types.float64, types.float64))

			tanh_lo = tanh(builder, [cast(context, builder, x.lo)])
			tanh_hi = tanh(builder, [cast(context, builder, x.hi)])

			y.lo = val_rd(context, builder, tanh_lo)
			y.hi = val_ru(context, builder, tanh_hi)
			return y
		return tanh(*intervals)

	def asin_interval(context, builder, sig, args, *intervals):
		maxf = context.get_function(max, signature(*([interval_t.__member_type__]*3)))
		minf = context.get_function(min, signature(*([interval_t.__member_type__]*3)))

		def asin(x,y):
			nan = const_nan(context)
			n_one = const_n_one(context)
			one = const_one(context)

			asin = context.get_function(mathfuncs.asin, signature(types.float64, types.float64))

			with builder.if_else(builder.fcmp_unordered('<', x.hi, n_one)) as (true, false):
				with true:
					y.lo = nan
					y.hi = nan
				with false:
					with builder.if_else(builder.fcmp_unordered('>', x.lo, one)) as (true, false):
						with true:
							y.lo = nan
							y.hi = nan
						with false:
							lo = maxf(builder, [x.lo, n_one])
							hi = minf(builder, [x.hi, one])

							asin_lo = asin(builder, [cast(context, builder, lo)])
							asin_hi = asin(builder, [cast(context, builder, hi)])

							y.lo = val_rd(context, builder, asin_lo)
							y.hi = val_ru(context, builder, asin_hi)
			return y
		return asin(*intervals)

	def acos_interval(context, builder, sig, args, *intervals):
		maxf = context.get_function(max, signature(*([interval_t.__member_type__]*3)))
		minf = context.get_function(min, signature(*([interval_t.__member_type__]*3)))

		def acos(x,y):
			nan = const_nan(context)
			n_one = const_n_one(context)
			one = const_one(context)

			acos = context.get_function(mathfuncs.acos, signature(types.float64, types.float64))

			with builder.if_else(builder.fcmp_unordered('<', x.hi, n_one)) as (true, false):
				with true:
					y.lo = nan
					y.hi = nan
				with false:
					with builder.if_else(builder.fcmp_unordered('>', x.lo, one)) as (true, false):
						with true:
							y.lo = nan
							y.hi = nan
						with false:
							lo = maxf(builder, [x.lo, n_one])
							hi = minf(builder, [x.hi, one])

							acos_lo = acos(builder, [cast(context, builder, lo)])
							acos_hi = acos(builder, [cast(context, builder, hi)])

							y.lo = val_rd(context, builder, acos_lo)
							y.hi = val_ru(context, builder, acos_hi)
			return y
		return acos(*intervals)

	def atan_interval(context, builder, sig, args, *intervals):
		def atan(x,y):
			atan = context.get_function(mathfuncs.atan, signature(types.float64, types.float64))

			atan_lo = atan(builder, [cast(context, builder, x.lo)])
			atan_hi = atan(builder, [cast(context, builder, x.hi)])

			y.lo = val_rd(context, builder, atan_lo)
			y.hi = val_ru(context, builder, atan_hi)
			return y
		return atan(*intervals)

	def asinh_interval(context, builder, sig, args, *intervals):
		def asinh(x,y):
			asinh = context.get_function(mathfuncs.asinh, signature(types.float64, types.float64))

			asinh_lo = asinh(builder, [cast(context, builder, x.lo)])
			asinh_hi = asinh(builder, [cast(context, builder, x.hi)])

			y.lo = val_rd(context, builder, asinh_lo)
			y.hi = val_ru(context, builder, asinh_hi)
			return y
		return asinh(*intervals)

	def acosh_interval(context, builder, sig, args, *intervals):
		maxf = context.get_function(max, signature(*([interval_t.__member_type__]*3)))

		def acosh(x,y):
			nan = const_nan(context)
			one = const_one(context)
			one_64 = context.get_constant(types.float64, np.float64(1))

			acosh = context.get_function(mathfuncs.acosh, signature(types.float64, types.float64))

			with builder.if_else(builder.fcmp_unordered('<', x.hi, one)) as (true, false):
				with true:
					y.lo = nan
					y.hi = nan
				with false:
					lo = maxf(builder, [one_64, cast(context, builder, x.lo)])
					hi = cast(context, builder, x.hi)

					acosh_lo = acosh(builder, [lo])
					acosh_hi = acosh(builder, [hi])

					y.lo = val_rd(context, builder, acosh_lo)
					y.hi = val_ru(context, builder, acosh_hi)
			return y
		return acosh(*intervals)

	def atanh_interval(context, builder, sig, args, *intervals):
		minf = context.get_function(min, signature(*([interval_t.__member_type__]*3)))
		maxf = context.get_function(max, signature(*([interval_t.__member_type__]*3)))

		def atanh(x,y):
			nan = const_nan(context)
			one = const_one(context)
			n_one = const_n_one(context)

			atanh = context.get_function(mathfuncs.atanh, signature(types.float64, types.float64))

			with builder.if_else(builder.fcmp_unordered('<', x.hi, n_one)) as (true, false):
				with true:
					y.lo = nanf
					y.hi = nanf
				with false:
					with builder.if_else(builder.fcmp_unordered('>', x.lo, one)) as (true, false):
						with true:
							y.lo = nanf
							y.hi = nanf
						with false:
							lo = maxf(builder, [x.lo, n_one])
							hi = minf(builder, [x.hi, one])

							atanh_lo = atanh(builder, [cast(context, builder, lo)])
							atanh_hi = atanh(builder, [cast(context, builder, hi)])

							y.lo = val_rd(context, builder, atanh_lo)
							y.hi = val_ru(context, builder, atanh_hi)
			return y
		return atanh(*intervals)

	ops = [#("sin", sin_interval),
		   #("cos", cos_interval),
		   #("tan", tan_interval),
		   ("sinh", sinh_interval),
		   ("cosh", cosh_interval),
		   ("tanh", tanh_interval),
		   ("asin", asin_interval),
		   ("acos", acos_interval),
		   ("atan", atan_interval),
		   ("asinh", asinh_interval),
		   ("acosh", acosh_interval),
		   ("atanh", atanh_interval)]

	for op, caller in ops:
		unary_op_interval(getattr(mathfuncs, op), caller)
		# if interval == intervalf:
			# unary_op_interval(getattr(mathfuncs, op + 'f'), caller)

	# sinf
	# cosf
	# tanf
	# sinhf
	# coshf
	# tanhf
	# asinf
	# acosf
	# atanf
	# asinhf
	# acoshf
	# atanhf

	def binary_op_interval_interval(op, caller):
		@impl_registry.lower(op, interval_t, interval_t)
		def cuda_op_interval_interval(context, builder, sig, args):
			x = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			y = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			z = cgutils.create_struct_proxy(interval_t)(context, builder)
			return caller(context, builder, sig, args, x, y, z)._getvalue()

	def binary_op_interval_scalar(op, caller, scalar_type = types.Number):
		@impl_registry.lower(op, interval_t, scalar_type)
		def cuda_op_interval_scalar(context, builder, sig, args):
			x = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			y = context.cast(builder, args[1], sig.args[1], interval_t.__member_type__)
			z = cgutils.create_struct_proxy(interval_t)(context, builder)
			return caller(context, builder, sig, args, x, y, z)._getvalue()

	def binary_op_scalar_interval(op, caller, scalar_type = types.Number):
		@impl_registry.lower(op, scalar_type, interval_t)
		def cuda_op_scalar_interval(context, builder, sig, args):
			x = context.cast(builder, args[0], sig.args[0], interval_t.__member_type__)
			y = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			z = cgutils.create_struct_proxy(interval_t)(context, builder)
			return caller(context, builder, sig, args, x, y, z)._getvalue()

	class binary_op_template(ConcreteTemplate):
		cases = [signature(interval_t, interval_t, interval_t),
				 signature(interval_t, interval_t.__member_type__, interval_t),
				 signature(interval_t, interval_t, interval_t.__member_type__)]

	def add_interval_interval(context, builder, sig, args, *intervals):
		add_rd = op_rd(context, 'add')
		add_ru = op_ru(context, 'add')
		def add(x,y,z):
			z.lo = add_rd(builder, [x.lo, y.lo])
			z.hi = add_ru(builder, [x.hi, y.hi])
			return z
		return add(*intervals)

	def add_interval_scalar(context, builder, sig, args, *intervals):
		add_rd = op_rd(context, 'add')
		add_ru = op_ru(context, 'add')
		def add(x,y,z):
			z.lo = add_rd(builder, [x.lo, y])
			z.hi = add_rd(builder, [x.hi, y])
			return z
		return add(*intervals)

	def add_scalar_interval(context, builder, sig, args, *intervals):
		x,y,z = intervals
		return add_interval_scalar(context, builder, sig, args, y, x, z)

	def sub_interval_interval(context, builder, sig, args, *intervals):
		sub_rd = op_rd(context, 'sub')
		sub_ru = op_ru(context, 'sub')
		def sub(x,y,z):
			z.lo = sub_rd(builder, [x.lo, y.hi])
			z.hi = sub_ru(builder, [x.hi, y.lo])
			return z
		return sub(*intervals)

	def sub_interval_scalar(context, builder, sig, args, *intervals):
		sub_rd = op_rd(context, 'sub')
		sub_ru = op_ru(context, 'sub')
		def sub(x,y,z):
			z.lo = sub_rd(builder, [x.lo, y])
			z.hi = sub_ru(builder, [x.hi, y])
			return z
		return sub(*intervals)

	def sub_scalar_interval(context, builder, sig, args, *intervals):
		x,y,z = intervals
		y = neg_interval(context, builder, sig, args, y, z)
		return add_interval_scalar(context, builder, sig, args, y, x, z)

	def mul_interval_interval(context, builder, sig, args, *intervals):
		mul_rd = op_rd(context, 'mul')
		mul_ru = op_ru(context, 'mul')
		minf = context.get_function(min, signature(*[interval_t.__member_type__]*3))
		maxf = context.get_function(max, signature(*[interval_t.__member_type__]*3))
		def mul(x,y,z):
			lo = mul_rd(builder, [x.lo, y.lo])
			hi = mul_ru(builder, [x.lo, y.lo])
			for a, b in [(x.lo, y.hi), (x.hi, y.lo), (x.hi, y.hi)]:
				lo = minf(builder, [lo, mul_rd(builder, [a, b])])
				hi = maxf(builder, [hi, mul_ru(builder, [a, b])])
			z.lo = lo
			z.hi = hi
			return z
		return mul(*intervals)

	def mul_interval_scalar(context, builder, sig, args, *intervals):
		mul_rd = op_rd(context, 'mul')
		mul_ru = op_ru(context, 'mul')
		minf = context.get_function(min, signature(*[interval_t.__member_type__]*3))
		maxf = context.get_function(max, signature(*[interval_t.__member_type__]*3))
		def mul(x,y,z):
			lo = mul_rd(builder, [y, x.lo])
			hi = mul_ru(builder, [y, x.hi])
			z.lo = minf(builder, [lo, hi])
			z.hi = maxf(builder, [lo, hi])
			return z
		return mul(*intervals)

	def mul_scalar_interval(context, builder, sig, args, *intervals):
		x, y, z = intervals
		return mul_interval_scalar(context, builder, sig, args, y, x, z)

	# def div_interval_interval(context, builder, sig, args, *intervals):
		# def div(x,y,z):
		# 	return z
		# return div(*intervals)

	# def div_interval_scalar(context, builder, sig, args, *intervals):
		# def div(x,y,z):
		# 	return z
		# return div(*intervals)

	# def div_scalar_interval(context, builder, sig, args, *intervals):
		# def div(x,y,z):
		# 	return z
		# return div(*intervals)

	def pow_interval_interval(context, builder, sig, args, *intervals):
		log = context.get_function(mathfuncs.log, signature(*([interval_t]*2)))
		mul = context.get_function(operator.mul, signature(*([interval_t]*3)))
		exp = context.get_function(mathfuncs.exp, signature(*([interval_t]*2)))
		def pow(x,y,z):
			a = cgutils.create_struct_proxy(interval_t)(context, builder, value = log(builder, [x._getvalue()]))
			b = cgutils.create_struct_proxy(interval_t)(context, builder, value = mul(builder, [a._getvalue(), y._getvalue()]))
			c = cgutils.create_struct_proxy(interval_t)(context, builder, value = exp(builder, [b._getvalue()]))
			z.lo = c.lo
			z.hi = c.hi
			return z
		return pow(*intervals)

	def pow_interval_int(context, builder, sig, args, *intervals):
		def pow(x,n,z):
			# with builder.if_else(builder.icmp_signed('==', n, zero_int)) as (true, false):
			# 	with true:
			# 		z.lo = zero_float
			# 		z.hi = zero_float
			# 	with false:
			# 		# n = 1
			# 		# x = interval(self.lo, self.hi)
			# 		with builder.if_then(builder.icmp_signed('<', n, zero_int)):
			# 			# x = 1 / self
			# 			# n = -n

			# 		# while n > 1:
			# 		# 	if n % 2 == 0:
			# 		# 		x = x * x
			# 		# 		n = n / 2
			# 		# 	else:
			# 		# 		y = x * y
			# 		# 		x = x * x
			# 		# 		n = (n - 1) / 2
			# 		# return x * y
			return z
		return pow(*intervals)

	def pow_interval_float(context, builder, sig, args, *intervals):
		log = context.get_function(mathfuncs.log, signature(*([interval_t]*2)))
		mul = context.get_function(operator.mul, sig)
		exp = context.get_function(mathfuncs.exp, signature(*([interval_t]*2)))
		def pow(x,y,z):
			# return exp(log(x)*y)
			a = cgutils.create_struct_proxy(interval_t)(context, builder, value = log(builder, [x._getvalue()]))
			b = cgutils.create_struct_proxy(interval_t)(context, builder, value = mul(builder, [a._getvalue(), y]))
			c = cgutils.create_struct_proxy(interval_t)(context, builder, value = exp(builder, [b._getvalue()]))
			z.lo = c.lo
			z.hi = c.hi
			return z
		return pow(*intervals)

	# def pow_scalar_interval(context, builder, sig, args, *intervals):
		# def pow(x,y,z):
		# 	return z
		# return pow(*intervals)

	def min_interval_interval(context, builder, sig, args, *intervals):
		minf = context.get_function(min, signature(*[interval_t.__member_type__]*3))
		def minimum(x,y,z):
			z.lo = minf(builder, [x.lo, y.lo])
			z.hi = minf(builder, [x.hi, y.hi])
			return z
		return minimum(*intervals)

	def min_interval_scalar(context, builder, sig, args, *intervals):
		minf = context.get_function(min, signature(*[interval_t.__member_type__]*3))
		def minimum(x,y,z):
			z.lo = minf(builder, [x.lo, y])
			z.hi = minf(builder, [x.hi, y])
			return z
		return minimum(*intervals)

	def min_scalar_interval(context, builder, sig, args, *intervals):
		x,y,z = intervals
		return min_interval_scalar(context, builder, sig, args, y, x, z)

	def max_interval_interval(context, builder, sig, args, *intervals):
		maxf = context.get_function(max, signature(*[interval_t.__member_type__]*3))
		def maximum(x,y,z):
			z.lo = maxf(builder, [x.lo, y.lo])
			z.hi = maxf(builder, [x.hi, y.hi])
			return z
		return maximum(*intervals)

	def max_interval_scalar(context, builder, sig, args, *intervals):
		maxf = context.get_function(max, signature(*[interval_t.__member_type__]*3))
		def maximum(x,y,z):
			z.lo = maxf(builder, [x.lo, y])
			z.hi = maxf(builder, [x.hi, y])
			return z
		return maximum(*intervals)

	def max_scalar_interval(context, builder, sig, args, *intervals):
		x,y,z = intervals
		return max_interval_scalar(context, builder, sig, args, y, x, z)

	ops = [(operator.add, [add_interval_interval, add_interval_scalar, add_scalar_interval]),
		   (operator.sub, [sub_interval_interval, sub_interval_scalar, sub_scalar_interval]),
		   (operator.mul, [mul_interval_interval, mul_interval_scalar, mul_scalar_interval]),
		   #(operator.truediv, [div_interval_interval, div_interval_scalar, div_scalar_interval]),
		   (min, [min_interval_interval, min_interval_scalar, min_scalar_interval]),
		   (max, [max_interval_interval, max_interval_scalar, max_scalar_interval])]
	#maxf
	#fmaxf
	#minf
	#fminf

	for op, callers in ops:
		decl_registry.register_global(op)(binary_op_template)

		interval_interval, interval_scalar, scalar_interval = callers
		binary_op_interval_interval(op, interval_interval)
		binary_op_interval_scalar(op, interval_scalar)
		binary_op_scalar_interval(op, scalar_interval) 

	##############################################
	##############################################
	##############################################

	decl_registry.register_global(operator.pow)(binary_op_template)
	binary_op_interval_interval(operator.pow, pow_interval_interval)
	# binary_op_interval_scalar(operator.pow, pow_interval_int, types.Integer)
	binary_op_interval_scalar(operator.pow, pow_interval_float, types.Float)
	# binary_op_scalar_interval(operator.pow, pow_scalar_interval), types.Integer

	##############################################
	##############################################
	##############################################


	def binary_iop_interval_interval(op, caller):
		@impl_registry.lower(op, interval_t, interval_t)
		def cuda_op_interval_interval(context, builder, sig, args):
			x = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			y = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			return caller(context, builder, sig, args, x, y, x)._getvalue()

	def binary_iop_interval_scalar(op, caller):
		@impl_registry.lower(op, interval_t, types.Number)
		def cuda_op_interval_scalar(context, builder, sig, args):
			x = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			y = context.cast(builder, args[1], sig.args[1], interval_t.__member_type__)
			return caller(context, builder, sig, args, x, y, x)._getvalue()

	def binary_iop_scalar_interval(op, caller):
		@impl_registry.lower(op, types.Number, interval_t)
		def cuda_op_scalar_interval(context, builder, sig, args):
			x = context.cast(builder, args[0], sig.args[0], interval_t.__member_type__)
			y = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			return caller(context, builder, sig, args, x, y, x)._getvalue()

	ops = [(operator.iadd, [add_interval_interval, add_interval_scalar, add_scalar_interval]),
		   (operator.isub, [sub_interval_interval, sub_interval_scalar, sub_scalar_interval]),
		   (operator.imul, [mul_interval_interval, mul_interval_scalar, mul_scalar_interval]),
		   #(operator.itruediv, [div_interval_interval, div_interval_scalar, div_scalar_interval]),
		   #(operator.ipow, [pow_interval_interval, pow_interval_scalar, pow_scalar_interval])
		   ]

	for op, callers in ops:
		decl_registry.register_global(op)(binary_op_template)

		interval_interval, interval_scalar, scalar_interval = callers
		binary_iop_interval_interval(op, interval_interval)
		binary_iop_interval_scalar(op, interval_scalar)
		binary_iop_scalar_interval(op, scalar_interval) 

	def bool_op_interval_interval(op, caller):
		@impl_registry.lower(op, interval_t, interval_t)
		def cuda_op_interval_interval(context, builder, sig, args):
			x = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			y = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			return caller(context, builder, sig, args, x, y)

	def bool_op_interval_scalar(op, caller):
		@impl_registry.lower(op, interval_t, types.Number)
		def cuda_op_interval_scalar(context, builder, sig, args):
			x = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			y = context.cast(builder, args[1], sig.args[1], interval_t.__member_type__)
			return caller(context, builder, sig, args, x, y)

	def bool_op_scalar_interval(op, caller):
		@impl_registry.lower(op, types.Number, interval_t)
		def cuda_op_scalar_interval(context, builder, sig, args):
			x = context.cast(builder, args[0], sig.args[0], interval_t.__member_type__)
			y = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			return caller(context, builder, sig, args, x, y)

	def eq_interval_interval(context, builder, sig, args, *intervals):
		def eq(x,y):
			return builder.and_(builder.fcmp_unordered("==", x.lo, y.lo), builder.fcmp_unordered("==", x.hi, y.hi))
		return eq(*intervals)

	def eq_interval_scalar(context, builder, sig, args, *intervals):
		def eq(x,y):
			return builder.and_(builder.fcmp_unordered("==", x.lo, y), builder.fcmp_unordered("==", x.hi, y))
		return eq(*intervals)

	def eq_scalar_interval(context, builder, sig, args, *intervals):
		def eq(x,y):
			return builder.and_(builder.fcmp_unordered("==", x, y.lo), builder.fcmp_unordered("==", x, y.hi))
		return eq(*intervals)

	def ne_interval_interval(context, builder, sig, args, *intervals):
		def ne(x,y):
			return builder.and_(builder.fcmp_unordered("!=", x.lo, y.lo), builder.fcmp_unordered("!=", x.hi, y.hi))
		return ne(*intervals)

	def ne_interval_scalar(context, builder, sig, args, *intervals):
		def ne(x,y):
			return builder.and_(builder.fcmp_unordered("!=", x.lo, y), builder.fcmp_unordered("!=", x.hi, y))
		return ne(*intervals)

	def ne_scalar_interval(context, builder, sig, args, *intervals):
		def ne(x,y):
			return builder.and_(builder.fcmp_unordered("!=", x, y.lo), builder.fcmp_unordered("!=", x, y.hi))
		return ne(*intervals)

	def gt_interval_interval(context, builder, sig, args, *intervals):
		def gt(x,y):
			return builder.fcmp_unordered(">=", x.lo, y.hi)
		return gt(*intervals)

	def gt_interval_scalar(context, builder, sig, args, *intervals):
		def gt(x,y):
			return builder.fcmp_unordered(">=", x.lo, y)
		return gt(*intervals)

	def gt_scalar_interval(context, builder, sig, args, *intervals):
		def gt(x,y):
			return builder.fcmp_unordered(">=", x, y.hi)
		return gt(*intervals)

	def ge_interval_interval(context, builder, sig, args, *intervals):
		def ge(x,y):
			return builder.fcmp_unordered(">", x.lo, y.hi)
		return ge(*intervals)

	def ge_interval_scalar(context, builder, sig, args, *intervals):
		def ge(x,y):
			return builder.fcmp_unordered(">", x.lo, y)
		return ge(*intervals)

	def ge_scalar_interval(context, builder, sig, args, *intervals):
		def ge(x,y):
			return builder.fcmp_unordered(">", x, y.hi)
		return ge(*intervals)

	def lt_interval_interval(context, builder, sig, args, *intervals):
		def lt(x,y):
			return builder.fcmp_unordered("<", x.hi, y.lo)
		return lt(*intervals)

	def lt_interval_scalar(context, builder, sig, args, *intervals):
		def lt(x,y):
			return builder.fcmp_unordered("<", x.hi, y)
		return lt(*intervals)

	def lt_scalar_interval(context, builder, sig, args, *intervals):
		def lt(x,y):
			return builder.fcmp_unordered("<", x, y.lo)
		return lt(*intervals)

	def le_interval_interval(context, builder, sig, args, *intervals):
		def le(x,y):
			return builder.fcmp_unordered("<=", x.hi, y.lo)
		return le(*intervals)

	def le_interval_scalar(context, builder, sig, args, *intervals):
		def le(x,y):
			return builder.fcmp_unordered("<=", x.hi, y)
		return le(*intervals)

	def le_scalar_interval(context, builder, sig, args, *intervals):
		def le(x,y):
			return builder.fcmp_unordered("<=", x, y.lo)
		return le(*intervals)

	class bool_op_template(ConcreteTemplate):
		cases = [signature(types.boolean, interval_t, interval_t),
				 signature(types.boolean, interval_t.__member_type__, interval_t),
				 signature(types.boolean, interval_t, interval_t.__member_type__)]

	ops = [(operator.eq, [eq_interval_interval, eq_interval_scalar, eq_scalar_interval]),
		   (operator.ne, [ne_interval_interval, ne_interval_scalar, ne_scalar_interval]),
		   (operator.gt, [gt_interval_interval, gt_interval_scalar, gt_scalar_interval]),
		   (operator.ge, [ge_interval_interval, ge_interval_scalar, ge_scalar_interval]),
		   (operator.lt, [lt_interval_interval, lt_interval_scalar, lt_scalar_interval]),
		   (operator.le, [le_interval_interval, le_interval_scalar, le_scalar_interval])]

	for op, callers in ops:
		decl_registry.register_global(op)(bool_op_template)

		interval_interval, interval_scalar, scalar_interval = callers
		bool_op_interval_interval(op, interval_interval)
		bool_op_interval_scalar(op, interval_scalar)
		bool_op_scalar_interval(op, scalar_interval) 


def op_rd_32(context, op, sig = signature(types.float32, types.float32, types.float32), lib = mathfuncs):
	return context.get_function(getattr(lib, "f" + op + "_rd"), sig)

def op_ru_32(context, op, sig = signature(types.float32, types.float32, types.float32), lib = mathfuncs):
	return context.get_function(getattr(lib, "f" + op + "_ru"), sig)

def val_rd_32(context, builder, lo):
	double2float_rd = context.get_function(mathfuncs.double2float_rd, signature(types.float32, types.float64))
	lo = double2float_rd(builder, [lo])
	return lo

def val_ru_32(context, builder, hi):
	double2float_ru = context.get_function(mathfuncs.double2float_ru, signature(types.float32, types.float64))
	hi = double2float_ru(builder, [hi])
	return hi

def op_rd_64(context, op, sig = signature(types.float64, types.float64, types.float64), lib = operator):
	return context.get_function(getattr(lib, op), sig)

def op_ru_64(context, op, sig = signature(types.float64, types.float64, types.float64), lib = operator):
	return context.get_function(getattr(lib, op), sig)

def val_rd_64(context, builder, lo):
	nextafter = context.get_function(mathfuncs.nextafter, signature(types.float64, types.float64, types.float64))
	ninf = context.get_constant(types.float64, np.float64(-np.inf))
	lo = nextafter(builder, [lo, ninf])
	return lo

def val_ru_64(context, builder, hi):
	nextafter = context.get_function(mathfuncs.nextafter, signature(types.float64, types.float64, types.float64))
	inf = context.get_constant(types.float64, np.float64(np.inf))
	hi = nextafter(builder, [hi, inf])
	return hi

def constant(typ, val):
	def get_constant(context):
		return context.get_constant(typ, val)
	return get_constant

for typ, decl in zip([intervalf, intervald], [intervalf_t, intervald_t]):
	interval_factory(typ, decl)





from .intervaltype import intervalf2, intervald2, intervalf3, intervald3
from .intervaldecl import (intervalf2_t, intervalf3_t,
						   intervald2_t, intervald3_t)
from ..vector.vectordecl import (Vector2Type, Vector3Type,
								 float2_t, float3_t,
								 double2_t, double3_t)
from ..vector.vectorimpl import vec_decl, floating_vec_decl, libfunc_op_caller

def libfunc_op_caller(op,builder,rtrn,*args):
	for n, attr in enumerate(rtrn._fe_type.__member_names__):
		setattr(rtrn, attr, op(builder, [arg[n] for arg in args]))
	return rtrn._getvalue()

def libfunc_vec_helper_wrapper(cuda_op):
	def libfunc_helper(context, builder, sig, args):
		# print(sig)
		# print(args[0], dir(args[0]))
		types = [arg.__member_type__ for arg in sig.args]
		sig = signature(*([sig.return_type.__member_type__] + types))
		return context.get_function(cuda_op, sig), libfunc_op_caller
	return libfunc_helper

def libfunc_floating_helper_wrapper(cuda_op):
	def libfunc_helper(context, builder, sig, args):
		types = [sig.args[0].__member_type__ , sig.args[1]]
		sig = signature(*([sig.return_type.__member_type__] + types))
		return context.get_function(cuda_op, sig), libfunc_op_caller
	return libfunc_helper

def binary_op_factory_floatvec(op, vec_t, floatvec, op_helper):
	@impl_registry.lower(op, vec_t, floatvec)
	@impl_registry.lower(op, floatvec, vec_t)
	def cuda_op_vec_vec(context, builder, sig, args):
		a = cgutils.create_struct_proxy(sig.args[0])(context, builder, value = args[0])
		b = cgutils.create_struct_proxy(sig.args[1])(context, builder, value = args[1])
		out = cgutils.create_struct_proxy(vec_t)(context, builder)

		a_members = [getattr(a, attr) for attr in sig.return_type.__member_names__]
		b_members = [getattr(b, attr) for attr in sig.return_type.__member_names__]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,out,a_members,b_members)

def binary_op_factory_scalar(op, vec_t, op_helper):
	@impl_registry.lower(op, vec_t, types.Number)
	@impl_registry.lower(op, types.Number, vec_t)
	def cuda_op_vec_vec(context, builder, sig, args):
		a = sig.args[1] == sig.return_type
		b = 1 - a

		a = cgutils.create_struct_proxy(vec_t)(context, builder, value=args[a])
		b = args[b] #context.cast(builder, args[b], sig.args[b], sig.return_type.__member_type__.__member_type__)
		out = cgutils.create_struct_proxy(vec_t)(context, builder)

		a_members = [getattr(a, attr) for attr in sig.return_type.__member_names__]
		b_members = [b for attr in sig.return_type.__member_names__]

		op, caller = op_helper(context, builder, sig, args)
		return caller(op,builder,out,a_members,b_members)


vec2s = [intervalf2, intervald2]
vec3s = [intervalf3, intervald3]

vec2s_t = [intervalf2_t, intervald2_t]
vec3s_t = [intervalf3_t, intervald3_t]

floating2_t = [float2_t, double2_t]
floating3_t = [float3_t, double3_t]

vec_groups = [vec2s, vec3s]
vec_t_groups = [vec2s_t, vec3s_t]
Vecs = [Vector2Type, Vector3Type]
floating_groups = [floating2_t, floating3_t]

for vecs, vecs_t, Vec, floatings_t in zip(vec_groups, vec_t_groups, Vecs, floating_groups):
	for vec, vec_t, floating_t in zip(vecs, vecs_t, floatings_t):
		vec_decl(vec, vec_t, Vec)
		floating_vec_decl(vec, vec_t)

		class vec_op_template(ConcreteTemplate):
			cases = [signature(vec_t, floating_t, vec_t),
					 signature(vec_t, vec_t, floating_t)]

		ops = [operator.add, operator.sub, operator.mul, operator.truediv]
		for op in ops:
			decl_registry.register_global(op)(vec_op_template)
			binary_op_factory_floatvec(op, vec_t, floating_t, libfunc_vec_helper_wrapper(op))

		# decl_registry.register_global(min)(vec_op_template)
		decl_registry.register_global(max)(vec_op_template)

		# min_helper = libfunc_vec_helper_wrapper(mathfuncs.min)
		max_helper = libfunc_vec_helper_wrapper(mathfuncs.max)

		# binary_op_factory_floatvec(min, vec_t, min_helper)
		binary_op_factory_floatvec(max, vec_t, floating_t, max_helper)
		# binary_op_factory_floatvec(mathfuncs.min, vec_t, min_helper)
		# binary_op_factory_floatvec(mathfuncs.max, vec_t, floatvec, max_helper)

		#dot




		floating = floating_t.__member_type__

		class vec_op_template(ConcreteTemplate):
			cases = [signature(vec_t, floating, vec_t),
					 signature(vec_t, vec_t, floating)]

		# decl_registry.register_global(min)(vec_op_template)
		decl_registry.register_global(max)(vec_op_template)

		# min_helper = libfunc_floating_helper_wrapper(min)
		max_helper = libfunc_floating_helper_wrapper(max)

		# binary_op_factory_scalar(min, vec_t, min_helper)
		binary_op_factory_scalar(max, vec_t, max_helper)
		# binary_op_factory_scalar(mathfuncs.min, vec_t, min_helper)
		# binary_op_factory_scalar(mathfuncs.max, vec_t, floating, max_helper)




