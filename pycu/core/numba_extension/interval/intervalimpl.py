from numba import types
from numba.core.extending import typeof_impl, type_callable
from numba.core.typing import signature

from numba.core.extending import models, register_model, lower_builtin
from numba.core import cgutils

from numba.cuda.cudadecl import registry as decl_registry
from numba.cuda.cudaimpl import registry as impl_registry
from numba.core.typing.templates import ConcreteTemplate, AttributeTemplate

from llvmlite import ir

import operator
from .intervaltype import intervalf, intervald
from .intervaldecl import intervalf_t, intervald_t, IntervalVectorType

from ..vector.vectordecl import VectorType

from .. import mathfuncs

import numpy as np

##########################################
#Define Interval Type
##########################################

def interval_factory(interval, interval_t):
	if interval == intervalf:
		float_t = ir.FloatType()
		ir_interval_t = ir.LiteralStructType([float_t, float_t])

		const_nan = constant(interval_t._member_t, np.float32(np.nan))
		const_inf = constant(interval_t._member_t, np.float32(np.inf))
		const_n_inf = constant(interval_t._member_t, np.float32(-np.inf))
		const_zero = constant(interval_t._member_t, np.float32(0))
		const_one = constant(interval_t._member_t, np.float32(1))
		const_n_one = constant(interval_t._member_t, np.float32(-1))
		const_n_pi_half_lo = constant(interval_t._member_t, -np.int32(0x3fc90fda).view(np.float32))
		const_pi_half_lo = constant(interval_t._member_t, np.int32(0x3fc90fda).view(np.float32))
		const_pi_half = constant(interval_t._member_t, np.int32(0x3fc90fdb).view(np.float32))
		const_pi_half_hi = constant(interval_t._member_t, np.int32(0x3fc90fdc).view(np.float32))
		const_pi_lo = constant(interval_t._member_t, np.int32(0x40490fda).view(np.float32))
		const_pi  = constant(interval_t._member_t, np.float32(np.pi))
		const_pi_hi = constant(interval_t._member_t, np.int32(0x40490fdc).view(np.float32))
		const_pi2_lo = constant(interval_t._member_t, np.int32(0x40c90fda).view(np.float32))
		const_pi2 = constant(interval_t._member_t, np.int32(0x40c90fdb).view(np.float32))
		const_pi2_hi = constant(interval_t._member_t, np.int32(0x40c90fdc).view(np.float32))

		cast = lambda context, builder, val : context.cast(builder, val, types.float32, types.float64)

		op_rd = op_rd_32
		op_ru = op_ru_32
		val_rd = val_rd_32
		val_ru = val_ru_32

		# const_pi  = constant(interval_t._member_t, np.float32(np.pi))
		# const_nan = constant(interval_t._member_t, np.float32(np.nan))
		# const_inf = constant(interval_t._member_t, np.float32(np.inf))
		# const_zero = constant(interval_t._member_t, np.float32(0))
		# const_one = constant(interval_t._member_t, np.float32(1))
		# const_n_one = constant(interval_t._member_t, np.float32(-1))

		# cast = lambda context, builder, val : context.cast(builder, val, types.float32, types.float64)

		# op_rd = op_rd_32
		# op_ru = op_ru_32
		# val_rd = val_rd_32
		# val_ru = val_ru_32

	elif interval == intervald:
		double_t = ir.DoubleType()
		ir_interval_t = ir.LiteralStructType([double_t, double_t])

		const_nan = constant(interval_t._member_t, np.float64(np.nan))
		const_inf = constant(interval_t._member_t, np.float64(np.inf))
		const_n_inf = constant(interval_t._member_t, np.float64(-np.inf))
		const_zero = constant(interval_t._member_t, np.float64(0))
		const_one = constant(interval_t._member_t, np.float64(1))
		const_n_one = constant(interval_t._member_t, np.float64(-1))
		const_n_pi_half_lo = constant(interval_t._member_t, -np.int64(0x3ff921fb54442d17).view(np.float64))
		const_pi_half_lo = constant(interval_t._member_t, np.int64(0x3ff921fb54442d17).view(np.float64))
		const_pi_half = constant(interval_t._member_t, np.int64(0x3ff921fb54442d18).view(np.float64))
		const_pi_half_hi = constant(interval_t._member_t, np.int64(0x3ff921fb54442d19).view(np.float64))
		const_pi_lo = constant(interval_t._member_t, np.int64(0x400921fb54442d17).view(np.float64))
		const_pi  = constant(interval_t._member_t, np.float64(np.pi))
		const_pi_hi = constant(interval_t._member_t, np.int64(0x400921fb54442d19).view(np.float64))
		const_pi2_lo = constant(interval_t._member_t, np.int64(0x401921fb54442d17).view(np.float64))
		const_pi2 = constant(interval_t._member_t, np.int64(0x401921fb54442d18).view(np.float64))
		const_pi2_hi = constant(interval_t._member_t, np.int64(0x401921fb54442d19).view(np.float64))

		cast = lambda context, builder, val : val

		op_rd = op_rd_64
		op_ru = op_ru_64
		val_rd = val_rd_64
		val_ru = val_ru_64

		# const_pi  = constant(interval_t._member_t, np.float64(np.pi))
		# const_nan = constant(interval_t._member_t, np.float64(np.nan))
		# const_inf = constant(interval_t._member_t, np.float64(np.inf))
		# const_zero = constant(interval_t._member_t, np.float64(0))
		# const_one = constant(interval_t._member_t, np.float64(1))
		# const_n_one = constant(interval_t._member_t, np.float64(-1))

		# cast = lambda context, builder, val : val

		# op_rd = op_rd_64
		# op_ru = op_ru_64
		# val_rd = val_rd_64
		# val_ru = val_ru_64

	@typeof_impl.register(interval)
	def typeof_interval(val, c):
		return interval_t

	@type_callable(interval)
	def type_interval(context):
		def typer(lo_hi):
			if isinstance(lo_hi, types.Number):
				return interval_t
			else:
				raise ValueError(f"Input to {interval.__name__} not understood")
		return typer

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
			members = [("lo", interval_t._member_t),
					   ("hi", interval_t._member_t) #,
					   # ("width", interval_t._member_t)
					   ]
			models.StructModel.__init__(self, dmm, fe_type, members)

	##########################################
	#Initializer/Constructor Methods
	##########################################

	#initialize: interval(lo, hi)
	@lower_builtin(interval, types.Number, types.Number)
	def impl_interval(context, builder, sig, args):
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
		out.lo = context.cast(builder, args[0], sig.args[0], interval_t._member_t)
		out.hi = context.cast(builder, args[1], sig.args[1], interval_t._member_t)
		return out._getvalue()

	#initialize: interval(lo_hi)
	@lower_builtin(interval, types.Number)
	def impl_vec(context, builder, sig, args):
		out = cgutils.create_struct_proxy(sig.return_type)(context, builder)
		out.lo = context.cast(builder, args[0], sig.args[0], interval_t._member_t)
		out.hi = context.cast(builder, args[0], sig.args[0], interval_t._member_t)
		return out._getvalue()

	##########################################
	#Define Interval Attributes
	##########################################

	@decl_registry.register_attr
	class Interval_attrs(AttributeTemplate):
		key = interval_t

		def resolve_lo(self, mod):
			return interval_t._member_t

		def resolve_hi(self, mod):
			return interval_t._member_t

		# def resolve_width(self, mod):
		# 	return interval_t._member_t

	@impl_registry.lower_getattr(interval_t, 'lo')
	def interval_get_lo(context, builder, sig, args):
		interval = cgutils.create_struct_proxy(sig)(context, builder, value = args)
		return interval.lo

	@impl_registry.lower_getattr(interval_t, 'hi')
	def interval_get_hi(context, builder, sig, args):
		interval = cgutils.create_struct_proxy(sig)(context, builder, value = args)
		return interval.hi

	# @impl_registry.lower_getattr(interval_t, 'width')
	# def interval_get_width(context, builder, sig, args):
	# 	sub_ru = op_ru(context, 'sub')
	# 	interval = cgutils.create_struct_proxy(sig)(context, builder, value = args)
	# 	return sub_ru(builder, [interval.hi, interval.lo])

	def unary_op_interval(op, caller):
		@impl_registry.lower(op, interval_t)
		def cuda_op_interval_interval(context, builder, sig, args):
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			return caller(context, builder, sig, args, f)

	class unary_op_template(ConcreteTemplate):
		cases = [signature(interval_t, interval_t)]

	def unary_impl(unary_op):
		def unary_interval(context, builder, sig, args, *intervals):
			op = context.get_function(unary_op, signature(interval_t._member_t, interval_t._member_t))
			def unary(f):
				out = cgutils.create_struct_proxy(interval_t)(context, builder)

				out.lo = op(builder, [f.lo])
				out.hi = op(builder, [f.hi])

				return out._getvalue()
			return unary(*intervals)
		return unary_interval

	def neg_interval(context, builder, sig, args, *intervals):
		neg_ = context.get_function(operator.neg, signature(interval_t._member_t, interval_t._member_t))

		def neg(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = neg_(builder, [f.hi])
			out.hi = neg_(builder, [f.lo])

			return out._getvalue()
		return neg(*intervals)

	def abs_interval(context, builder, sig, args, *intervals):
		zero = const_zero(context)

		abs_ = context.get_function(abs, signature(*([interval_t._member_t]*2)))
		max_ = context.get_function(max, signature(*([interval_t._member_t]*3)))
		neg_ = context.get_function(operator.neg, signature(*([interval_t]*2)))

		def absolute(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			positive = builder.fcmp_unordered('>=', f.lo, zero)
			with builder.if_else(positive) as (true, false):
				with true:
					out.lo = f.lo
					out.hi = f.hi
				with false:
					negative = builder.fcmp_unordered('<', f.hi, zero)
					with builder.if_else(negative) as (true, false):
						with true:
							temp = neg_(builder, [f._getvalue()])
							temp_ = cgutils.create_struct_proxy(interval_t)(context, builder, value = temp)
							out.lo = temp_.lo
							out.hi = temp_.hi
						with false:
							out.lo = zero
							out.hi = max_(builder, [abs_(builder, [f.lo]), abs_(builder, [f.hi])])
			return out._getvalue()
		return absolute(*intervals)

	def sign_interval(context, builder, sig, args, *intervals):
		sign_ = context.get_function(mathfuncs.sign, signature(*([interval_t._member_t]*2)))
		def sign(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)
			out.lo = sign_(builder, [f.lo])
			out.hi = sign_(builder, [f.hi])
			return out._getvalue()
		return sign(*intervals)


	ops = [(operator.neg, neg_interval),
		   (abs, abs_interval),
		   (round, unary_impl(mathfuncs.round))]

	for op, caller in ops:
		decl_registry.register_global(op)(unary_op_template)
		unary_op_interval(op, caller)
		# print(op, interval_t)
	#fabsf
	#absf

	# unary_op_interval(mathfuncs.abs, abs_interval)

	for op in ['floor', 'ceil']: #'round', 
		op1 = getattr(mathfuncs, op)
		if interval == intervalf:
			op2 = getattr(mathfuncs, op + 'f')
		else:
			op2 = op1
		unary_op_interval(op1, unary_impl(op2))

	unary_op_interval(mathfuncs.sign, sign_interval)

	def exp_factory(op):
		def exp_interval(context, builder, sig, args, *intervals):
			exp_ = context.get_function(op, signature(types.float64, types.float64))

			def exp(f):
				out = cgutils.create_struct_proxy(interval_t)(context, builder)

				#round ops for these functions do not exist so a cast/round method is used
				exp_lo = exp_(builder, [cast(context, builder, f.lo)])
				exp_hi = exp_(builder, [cast(context, builder, f.hi)])

				#value is rounded after op. For double, nextafter is used to force rounding as float128 isn't supported
				out.lo = val_rd(context, builder, exp_lo)
				out.hi = val_ru(context, builder, exp_hi)

				return out._getvalue()
			return exp(*intervals)
		return exp_interval

	for ext in ['', '10', '2']:  #'m1' -> expm1 = exp(x) - 1
		op = getattr(mathfuncs, f'exp{ext}')
		unary_op_interval(op, exp_factory(op))
	# expf
	# exp10f
	# exp2f

	def log_factory(op):
		def log_interval(context, builder, sig, args, *intervals):
			nan = const_nan(context)
			zero = const_zero(context)

			log_ = context.get_function(op, signature(types.float64, types.float64))

			def log(f):
				out = cgutils.create_struct_proxy(interval_t)(context, builder)

				negative = builder.fcmp_unordered('<', f.hi, zero)
				with builder.if_else(negative) as (true, false):
					with true:
						out.lo = nan
						out.hi = nan
					with false:
						split_zero = builder.fcmp_unordered('<=', f.lo, zero)
						with builder.if_else(split_zero) as (true, false):
							with true:
								log_hi = log_(builder, [cast(context, builder, f.hi)])

								out.lo = zero
								out.hi = val_ru(context, builder, log_hi)
							with false:
								log_lo = log_(builder, [cast(context, builder, f.lo)])
								log_hi = log_(builder, [cast(context, builder, f.hi)])

								out.lo = val_rd(context, builder, log_lo)
								out.hi = val_ru(context, builder, log_hi)

				return out._getvalue()
			return log(*intervals)
		return log_interval

	for ext in ['', '10', '2']: #1p -> log1p = log(1 + x)
		op = getattr(mathfuncs, f'log{ext}')
		unary_op_interval(op, log_factory(op))
	# logf
	# log10f
	# log2f

	def sqrt_interval(context, builder, sig, args, *intervals):
		zero = const_zero(context)
		nan = const_nan(context)

		sqrt_rd = op_rd(context, 'sqrt', signature(*([sig.args[0]._member_t]*2)), mathfuncs)
		sqrt_ru = op_ru(context, 'sqrt', signature(*([sig.args[0]._member_t]*2)), mathfuncs)

		def sqrt(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			negative = builder.fcmp_unordered('<', f.hi, zero)
			with builder.if_else(negative) as (true, false):
				with true:
					out.lo = nan
					out.hi = nan
				with false:
					split_zero = builder.fcmp_unordered('<=', f.lo, zero)
					with builder.if_else(split_zero) as (true, false):
						with true:
							out.lo = zero
							out.hi = sqrt_ru(builder, [f.hi])
						with false:
							out.lo = sqrt_rd(builder, [f.lo])
							out.hi = sqrt_ru(builder, [f.hi])

			return out._getvalue()
		return sqrt(*intervals)

	unary_op_interval(mathfuncs.sqrt, sqrt_interval)
	#sqrtf
	#rsqrt
	#cbrt
	#rcbrt

	def trig_mod_interval_interval(context, builder, sig, args, *intervals):
		zero = const_zero(context)

		div_ = context.get_function(operator.truediv, signature(*([interval_t._member_t]*3)))
		ceil_ = context.get_function(mathfuncs.ceil, signature(*([interval_t._member_t]*2)))
		floor_ = context.get_function(mathfuncs.floor, signature(*([interval_t._member_t]*2)))

		mul_ = context.get_function(operator.mul, signature(*([interval_t, interval_t, interval_t._member_t])))
		sub_ = context.get_function(operator.sub, signature(*([interval_t]*3)))

		def trig_fmod(f,g):
			den = builder.alloca(f.lo.type, 1)
			with builder.if_else(builder.fcmp_unordered('<', f.lo, zero)) as (true, false):
				with true:
					builder.store(g.hi, den)
				with false:
					builder.store(g.lo, den)

			n = cgutils.alloca_once_value(builder, div_(builder, [f.lo, builder.load(den)]))

			with builder.if_else(builder.fcmp_unordered('<', builder.load(n), zero)) as (true, false):
				with true:
					builder.store(ceil_(builder, [builder.load(n)]), n)
				with false:
					builder.store(floor_(builder, [builder.load(n)]), n)

			out = cgutils.create_struct_proxy(interval_t)(context, builder)
			out = sub_(builder, [f._getvalue(), mul_(builder, [g._getvalue(), builder.load(n)])])
			out = cgutils.create_struct_proxy(interval_t)(context, builder, value = out)

			return out
		return trig_fmod(*intervals)

	def trig_shift_negative_interval(context, builder, sig, args, *intervals):
		zero = const_zero(context)
		inf = const_inf(context)
		n_inf = const_n_inf(context)
		pi2_lo = const_pi2_lo(context)

		neg_ = context.get_function(operator.neg, signature(*([interval_t._member_t]*2)))
		div_ = context.get_function(operator.truediv, signature(*([interval_t._member_t]*3)))
		ceil_ = context.get_function(mathfuncs.ceil, signature(*([interval_t._member_t]*2)))
		mul_ = context.get_function(operator.mul, signature(*([interval_t._member_t]*3)))
		add_ = context.get_function(operator.add, signature(*([interval_t._member_t]*3)))

		def trig_shift(f):
			with builder.if_then(builder.fcmp_unordered("<", f.lo, zero)):
				with builder.if_else(builder.fcmp_unordered("==", f.lo, n_inf)) as (true, false):
					with true:
						f.lo = zero
						f.hi = inf
					with false:
						n = ceil_(builder, [div_(builder, [neg_(builder, [f.lo]), pi2_lo])]) 
						shift = mul_(builder, [n, pi2_lo])
						f.lo = add_(builder, [f.lo, shift])
						f.hi = add_(builder, [f.hi, shift])
			return f
		return trig_shift(*intervals)

	def cos_interval(context, builder, sig, args, *intervals):
		#https://github.com/mauriciopoppe/interval-arithmetic/blob/bc9e779/src/operations/trigonometric.ts#L240
		one = const_one(context)
		n_one = const_n_one(context)

		cos_ = context.get_function(mathfuncs.cos, signature(types.float64, types.float64))
		max_ = context.get_function(max, signature(*([interval_t._member_t]*3)))

		sub_ru = op_ru(context, 'sub')

		inactive = context.get_constant(types.uint32, np.uint32(0))
		active = context.get_constant(types.uint32, np.uint32(-1))

		sub_ = context.get_function(operator.sub, signature(*([interval_t]*3)))
		neg_ = context.get_function(operator.neg, signature(*([interval_t]*2)))

		def cos(f):
			pi = cgutils.create_struct_proxy(interval_t)(context, builder)
			pi.lo = const_pi_lo(context)
			pi.hi = const_pi_hi(context)
			pi2 = cgutils.create_struct_proxy(interval_t)(context, builder)
			pi2.lo = const_pi2_lo(context)
			pi2.hi = const_pi2_hi(context)

			cos_body = builder.append_basic_block("cos_interval.body")
			cos_end = builder.append_basic_block("cos_interval.end")

			negate = cgutils.alloca_once_value(builder, inactive)

			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			#use goto branch to emulate recursion
			builder.branch(cos_body)
			with builder.goto_block(cos_body):
				width = sub_ru(builder, [f.hi, f.lo])
				with builder.if_else(builder.fcmp_unordered('>=', width, pi2.lo)) as (true, false):
					with true:
						out.lo = n_one
						out.hi = one
					with false:
						temp1 = trig_shift_negative_interval(context, builder, sig, args, f)
						temp2 = trig_mod_interval_interval(context, builder, sig, args, temp1, pi2)
						f.lo = temp2.lo
						f.hi = temp2.hi
						with builder.if_else(builder.fcmp_unordered('>=', f.lo, pi.hi)) as (true, false):
							with true:
								temp3 = sub_(builder, [f._getvalue(), pi._getvalue()])
								temp3 = cgutils.create_struct_proxy(interval_t)(context, builder, value = temp3)

								f.lo = temp3.lo
								f.hi = temp3.hi

								#flip negate bit so that on next loop through cos_body the interval is negated
								builder.store(builder.not_(builder.load(negate)), negate)
								builder.branch(cos_body)
							with false:
								cos_lo = val_rd(context, builder, cos_(builder, [cast(context, builder, f.lo)]))
								cos_hi = val_ru(context, builder, cos_(builder, [cast(context, builder, f.hi)]))
								with builder.if_else(builder.fcmp_unordered("<=", f.hi, pi.lo)) as (true, false):
									with true:
										out.lo = cos_hi
										out.hi = cos_lo
									with false:
										with builder.if_else(builder.fcmp_unordered("<=", f.hi, pi2.lo)) as (true, false):
											with true:
												out.lo = n_one
												out.hi = max_(builder, [cos_lo, cos_hi])
											with false:
												out.lo = n_one
												out.hi = one
				builder.branch(cos_end)
			builder.position_at_end(cos_end)

			with builder.if_then(builder.icmp_unsigned('==', builder.load(negate), active)):
				temp4 = neg_(builder, [out._getvalue()])
				temp4 = cgutils.create_struct_proxy(interval_t)(context, builder, value = temp4)
				out.lo = temp4.lo
				out.hi = temp4.hi

			return out._getvalue()

		return cos(*intervals)

	def sin_interval(context, builder, sig, args, *intervals):
		sub_ = context.get_function(operator.sub, signature(*([interval_t]*3)))
		cos_ = context.get_function(mathfuncs.cos, signature(*([interval_t]*2)))

		def sin(f):
			pi_half = cgutils.create_struct_proxy(interval_t)(context, builder)
			pi_half.lo = const_pi_half_lo(context)
			pi_half.hi = const_pi_half_hi(context)

			return cos_(builder, [sub_(builder, [f._getvalue(), pi_half._getvalue()])])
		return sin(*intervals)

	# def tan_interval(context, builder, sig, args, *intervals):
		# inf = const_inf(context)
		# n_inf = const_n_inf(context)

		# pi_half_lo = const_pi_half_lo(context)
		# n_pi_half_lo = const_n_pi_half_lo(context)

		# subf = context.get_function(operator.sub, signature(*([interval_t._member_t]*3)))
		# tanf = context.get_function(mathfuncs.tan, signature(types.float64, types.float64))

		# def tan(x,y):
		# 	pi = cgutils.create_struct_proxy(interval_t)(context, builder)
		# 	pi.lo = const_pi_lo(context)
		# 	pi.hi = const_pi_hi(context)

		# 	z = cgutils.create_struct_proxy(interval_t)(context, builder)
		# 	shift_negative_interval(context, builder, sig, args, x, z)
		# 	trig_fmod_interval_interval(context, builder, sig, args, z, pi, x)

		# 	with builder.if_then(builder.fcmp_unordered(">=", x.lo, pi_half_lo)):
		# 		sub_interval_interval(context, builder, sig, args, x, pi, z)

		# 	split_pi_half = builder.or_(builder.fcmp_unordered("<=", z.lo, n_pi_half_lo), builder.fcmp_unordered(">=", z.hi, pi_half_lo))
		# 	with builder.if_else(split_pi_half) as (true, false):
		# 		with true:
		# 			y.lo = n_inf
		# 			y.hi = inf
		# 		with false:
		# 			tan_lo = tanf(builder, [cast(context, builder, z.lo)])
		# 			tan_hi = tanf(builder, [cast(context, builder, z.hi)])

		# 			y.lo = val_rd(context, builder, tan_lo)
		# 			y.hi = val_ru(context, builder, tan_hi)
		# 	return y
		# return tan(*intervals)

	def sinh_interval(context, builder, sig, args, *intervals):
		sinh_ = context.get_function(mathfuncs.sinh, signature(types.float64, types.float64))

		def sinh(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			sinh_lo = sinh_(builder, [cast(context, builder, f.lo)])
			sinh_hi = sinh_(builder, [cast(context, builder, f.hi)])

			out.lo = val_rd(context, builder, sinh_lo)
			out.hi = val_ru(context, builder, sinh_hi)

			return out._getvalue()
		return sinh(*intervals)

	def cosh_interval(context, builder, sig, args, *intervals):
		zero = const_zero(context)
		one = const_one(context)

		max_ = context.get_function(max, signature(*([interval_t._member_t]*3)))
		cosh_ = context.get_function(mathfuncs.cosh, signature(types.float64, types.float64))

		def cosh(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			cosh_lo = cosh_(builder, [cast(context, builder, f.lo)])
			cosh_hi = cosh_(builder, [cast(context, builder, f.hi)])

			with builder.if_else(builder.fcmp_unordered('<', f.hi, zero)) as (true, false):
				with true:
					out.lo = val_rd(context, builder, cosh_hi)
					out.hi = val_ru(context, builder, cosh_lo)
				with false:
					with builder.if_else(builder.fcmp_unordered('>', f.lo, zero)) as (true, false):
						with true:
							out.lo = val_rd(context, builder, cosh_lo)
							out.hi = val_ru(context, builder, cosh_hi)
						with false:
							out.lo = one
							out.hi = max_(builder, [val_ru(context, builder, cosh_lo), val_ru(context, builder, cosh_hi)])

			return out._getvalue()
		return cosh(*intervals)

	def tanh_interval(context, builder, sig, args, *intervals):
		tanh_ = context.get_function(mathfuncs.tanh, signature(types.float64, types.float64))

		def tanh(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			tanh_lo = tanh_(builder, [cast(context, builder, f.lo)])
			tanh_hi = tanh_(builder, [cast(context, builder, f.hi)])

			out.lo = val_rd(context, builder, tanh_lo)
			out.hi = val_ru(context, builder, tanh_hi)

			return out._getvalue()
		return tanh(*intervals)

	def asin_interval(context, builder, sig, args, *intervals):
		nan = const_nan(context)
		one = const_one(context)
		n_one = const_n_one(context)

		max_ = context.get_function(max, signature(*([interval_t._member_t]*3)))
		min_ = context.get_function(min, signature(*([interval_t._member_t]*3)))
		asin_ = context.get_function(mathfuncs.asin, signature(types.float64, types.float64))

		def asin(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			invalid = builder.or_(builder.fcmp_unordered('<', f.hi, n_one), builder.fcmp_unordered('<', one, f.lo))
			with builder.if_else(invalid) as (true, false):
				with true:
					y.lo = nan
					y.hi = nan
				with false:
					lo = max_(builder, [f.lo, n_one])
					hi = min_(builder, [f.hi, one])

					asin_lo = asin_(builder, [cast(context, builder, lo)])
					asin_hi = asin_(builder, [cast(context, builder, hi)])

					y.lo = val_rd(context, builder, asin_lo)
					y.hi = val_ru(context, builder, asin_hi)

			return out._getvalue()
		return asin(*intervals)

	def acos_interval(context, builder, sig, args, *intervals):
		nan = const_nan(context)
		one = const_one(context)
		n_one = const_n_one(context)

		max_ = context.get_function(max, signature(*([interval_t._member_t]*3)))
		min_ = context.get_function(min, signature(*([interval_t._member_t]*3)))
		acos_ = context.get_function(mathfuncs.acos, signature(types.float64, types.float64))

		def acos(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			invalid = builder.or_(builder.fcmp_unordered('<', x.hi, n_one), builder.fcmp_unordered('<', one, x.lo))
			with builder.if_else(invalid) as (true, false):
				with true:
					out.lo = nan
					out.hi = nan
				with false:
					lo = max_(builder, [x.lo, n_one])
					hi = min_(builder, [x.hi, one])

					acos_lo = acos_(builder, [cast(context, builder, lo)])
					acos_hi = acos_(builder, [cast(context, builder, hi)])

					out.lo = val_rd(context, builder, acos_lo)
					out.hi = val_ru(context, builder, acos_hi)

			return out._getvalue()
		return acos(*intervals)

	def atan_interval(context, builder, sig, args, *intervals):
		atan_ = context.get_function(mathfuncs.atan, signature(types.float64, types.float64))

		def atan(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			atan_lo = atan_(builder, [cast(context, builder, f.lo)])
			atan_hi = atan_(builder, [cast(context, builder, f.hi)])

			out.lo = val_rd(context, builder, atan_lo)
			out.hi = val_ru(context, builder, atan_hi)

			return out._getvalue()
		return atan(*intervals)

	def asinh_interval(context, builder, sig, args, *intervals):
		asinh_ = context.get_function(mathfuncs.asinh, signature(types.float64, types.float64))

		def asinh(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			asinh_lo = asinh_(builder, [cast(context, builder, f.lo)])
			asinh_hi = asinh_(builder, [cast(context, builder, f.hi)])

			y.lo = val_rd(context, builder, asinh_lo)
			y.hi = val_ru(context, builder, asinh_hi)

			return out._getvalue()
		return asinh(*intervals)

	def acosh_interval(context, builder, sig, args, *intervals):
		nan = const_nan(context)
		one = const_one(context)

		max_ = context.get_function(max, signature(*([interval_t._member_t]*3)))
		acosh_ = context.get_function(mathfuncs.acosh, signature(types.float64, types.float64))

		def acosh(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			with builder.if_else(builder.fcmp_unordered('<', f.hi, one)) as (true, false):
				with true:
					out.lo = nan
					out.hi = nan
				with false:
					lo = max_(builder, [context.get_constant(types.float64, np.float64(1)), cast(context, builder, f.lo)])
					hi = cast(context, builder, f.hi)

					acosh_lo = acosh_(builder, [lo])
					acosh_hi = acosh_(builder, [hi])

					out.lo = val_rd(context, builder, acosh_lo)
					out.hi = val_ru(context, builder, acosh_hi)

			return out._getvalue()
		return acosh(*intervals)

	def atanh_interval(context, builder, sig, args, *intervals):
		nan = const_nan(context)
		one = const_one(context)
		n_one = const_n_one(context)

		min_ = context.get_function(min, signature(*([interval_t._member_t]*3)))
		max_ = context.get_function(max, signature(*([interval_t._member_t]*3)))
		atanh_ = context.get_function(mathfuncs.atanh, signature(types.float64, types.float64))

		def atanh(f):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			invalid = builder.or_(builder.fcmp_unordered('<', f.hi, n_one), builder.fcmp_unordered('<', one, f.lo))
			with builder.if_else(invalid) as (true, false):
				with true:
					out.lo = nan
					out.hi = nan
				with false:
					lo = max_(builder, [f.lo, n_one])
					hi = min_(builder, [f.hi, one])

					atanh_lo = atanh_(builder, [cast(context, builder, lo)])
					atanh_hi = atanh_(builder, [cast(context, builder, hi)])

					out.lo = val_rd(context, builder, atanh_lo)
					out.hi = val_ru(context, builder, atanh_hi)

			return out._getvalue()
		return atanh(*intervals)

	ops = [("cos", cos_interval),
		   ("sin", sin_interval),
		   # ("tan", tan_interval),
		   ("cosh", cosh_interval),
		   ("sinh", sinh_interval),
		   ("tanh", tanh_interval),
		   ("acos", acos_interval),
		   ("asin", asin_interval),
		   ("atan", atan_interval),
		   ("acosh", acosh_interval),
		   ("asinh", asinh_interval),
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
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			return caller(context, builder, sig, args, f, g)

	def binary_op_interval_scalar(op, caller, scalar_type = types.Number): #scalar_type = interval_t._member_t
		@impl_registry.lower(op, interval_t, scalar_type)
		def cuda_op_interval_scalar(context, builder, sig, args):
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			g = context.cast(builder, args[1], sig.args[1], interval_t._member_t) #scalar_type
			return caller(context, builder, sig, args, f, g)

	def binary_op_scalar_interval(op, caller, scalar_type = types.Number): #scalar_type = interval_t._member_t
		@impl_registry.lower(op, scalar_type, interval_t)
		def cuda_op_scalar_interval(context, builder, sig, args):
			f = context.cast(builder, args[0], sig.args[0], interval_t._member_t) #scalar_type
			g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			return caller(context, builder, sig, args, f, g)

	class binary_op_template(ConcreteTemplate):
		cases = [signature(interval_t, interval_t, interval_t),
				 signature(interval_t, interval_t._member_t, interval_t),
				 signature(interval_t, interval_t, interval_t._member_t)]

	def add_interval_interval(context, builder, sig, args, *intervals):
		add_rd = op_rd(context, 'add')
		add_ru = op_ru(context, 'add')

		def add(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = add_rd(builder, [f.lo, g.lo])
			out.hi = add_ru(builder, [f.hi, g.hi])

			return out._getvalue()
		return add(*intervals)

	def add_interval_scalar(context, builder, sig, args, *intervals):
		add_rd = op_rd(context, 'add')
		add_ru = op_ru(context, 'add')

		def add(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = add_rd(builder, [f.lo, g])
			out.hi = add_ru(builder, [f.hi, g])

			return out._getvalue()
		return add(*intervals)

	def add_scalar_interval(context, builder, sig, args, *intervals):
		add_ = context.get_function(operator.add, signature(*[interval_t, interval_t, interval_t._member_t]))

		def add(f, g):
			return add_(builder, [g._getvalue(), f])
		return add(*intervals)

	def sub_interval_interval(context, builder, sig, args, *intervals):
		sub_rd = op_rd(context, 'sub')
		sub_ru = op_ru(context, 'sub')

		def sub(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = sub_rd(builder, [f.lo, g.hi])
			out.hi = sub_ru(builder, [f.hi, g.lo])

			return out._getvalue()
		return sub(*intervals)

	def sub_interval_scalar(context, builder, sig, args, *intervals):
		sub_rd = op_rd(context, 'sub')
		sub_ru = op_ru(context, 'sub')

		def sub(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = sub_rd(builder, [f.lo, g])
			out.hi = sub_ru(builder, [f.hi, g])

			return out._getvalue()
		return sub(*intervals)

	def sub_scalar_interval(context, builder, sig, args, *intervals):
		sub_rd = op_rd(context, 'sub')
		sub_ru = op_ru(context, 'sub')

		def sub(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = sub_rd(builder, [f, g.hi])
			out.hi = sub_ru(builder, [f, g.lo])

			return out._getvalue()
		return sub(*intervals)

	def mul_interval_interval(context, builder, sig, args, *intervals):
		mul_rd = op_rd(context, 'mul')
		mul_ru = op_ru(context, 'mul')
		min_ = context.get_function(min, signature(*[interval_t._member_t]*3))
		max_ = context.get_function(max, signature(*[interval_t._member_t]*3))

		def mul(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			lo = mul_rd(builder, [f.lo, g.lo])
			hi = mul_ru(builder, [f.lo, g.lo])
			for pair in [[f.lo, g.hi], [f.hi, g.lo], [f.hi, g.hi]]:
				lo = min_(builder, [lo, mul_rd(builder, pair)])
				hi = max_(builder, [hi, mul_ru(builder, pair)])
			out.lo = lo
			out.hi = hi

			return out._getvalue()
		return mul(*intervals)

	def mul_interval_scalar(context, builder, sig, args, *intervals):
		mul_rd = op_rd(context, 'mul')
		mul_ru = op_ru(context, 'mul')
		min_ = context.get_function(min, signature(*[interval_t._member_t]*3))
		max_ = context.get_function(max, signature(*[interval_t._member_t]*3))

		def mul(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			lo = mul_rd(builder, [f.lo, g])
			hi = mul_ru(builder, [f.hi, g])

			out.lo = min_(builder, [lo, hi])
			out.hi = max_(builder, [lo, hi])

			return out._getvalue()
		return mul(*intervals)

	def mul_scalar_interval(context, builder, sig, args, *intervals):
		mul_ = context.get_function(operator.mul, signature(*[interval_t, interval_t, interval_t._member_t]))

		def mul(f,g):
			return mul_(builder, [g._getvalue(), f])
		return mul(*intervals)

	def div_interval_interval(context, builder, sig, args, *intervals):
		zero = const_zero(context)
		inf = const_inf(context)
		n_inf = const_n_inf(context)

		div_rd = op_rd(context, 'div')
		div_ru = op_ru(context, 'div')
		min_ = context.get_function(min, signature(*[interval_t._member_t]*3))
		max_ = context.get_function(max, signature(*[interval_t._member_t]*3))

		def div(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			split_zero = builder.and_(builder.fcmp_unordered('<=', g.lo, zero), builder.fcmp_unordered('<=', zero, g.hi))
			with builder.if_else(split_zero) as (true, false):
				with true:
					out.lo = n_inf
					out.hi = inf
				with false:
					lo = div_rd(builder, [f.lo, g.lo])
					hi = div_ru(builder, [f.lo, g.lo])
					for pair in [[f.lo, g.hi], [f.hi, g.lo], [f.hi, g.hi]]:
						lo = min_(builder, [lo, div_rd(builder, pair)])
						hi = max_(builder, [hi, div_ru(builder, pair)])
					out.lo = lo
					out.hi = hi

			return out._getvalue()
		return div(*intervals)

	def div_interval_scalar(context, builder, sig, args, *intervals):
		zero = const_zero(context)
		inf = const_inf(context)
		n_inf = const_n_inf(context)

		div_rd = op_rd(context, 'div')
		div_ru = op_ru(context, 'div')

		def div(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			with builder.if_else(builder.fcmp_unordered('<', g, zero)) as (true, false):
				with true:
					out.lo = div_rd(builder, [f.hi, g])
					out.hi = div_ru(builder, [f.lo, g])
				with false:
					with builder.if_else(builder.fcmp_unordered('>', g, zero)) as (true, false):
						with true:
							out.lo = div_rd(builder, [f.lo, g])
							out.hi = div_ru(builder, [f.hi, g])
						with false:
							out.lo = n_inf
							out.hi = inf

			return out._getvalue()
		return div(*intervals)

	def div_scalar_interval(context, builder, sig, args, *intervals):
		div_ = context.get_function(operator.div, signature(*[interval_t]*3))

		def div(f,g):
			temp = cgutils.create_struct_proxy(interval_t)(context, builder)
			temp.lo = f
			temp.hi = f

			return div_(builder, [temp._getvalue(), g._getvalue()])
		return div(*intervals)

	# def pow_interval_interval(context, builder, sig, args, *intervals):
		# log = context.get_function(mathfuncs.log, signature(*([interval_t]*2)))
		# mul = context.get_function(operator.mul, signature(*([interval_t]*3)))
		# exp = context.get_function(mathfuncs.exp, signature(*([interval_t]*2)))
		# def pow(x,y,z):
		# 	a = cgutils.create_struct_proxy(interval_t)(context, builder, value = log(builder, [x._getvalue()]))
		# 	b = cgutils.create_struct_proxy(interval_t)(context, builder, value = mul(builder, [a._getvalue(), y._getvalue()]))
		# 	c = cgutils.create_struct_proxy(interval_t)(context, builder, value = exp(builder, [b._getvalue()]))
		# 	z.lo = c.lo
		# 	z.hi = c.hi
		# 	return z
		# return pow(*intervals)

	# def pow_interval_int(context, builder, sig, args, *intervals):
		# zero_i = context.get_constant(types.int64, 0)
		# one_i = context.get_constant(types.int64, 1)

		# zero = const_zero(context)
		# # one = const_one(context)

		# # one = context.get_constant(, 1)

		# sub_ = context.get_function(operator.sub, signature(*([types.int64]*3)))


		# def pow(x,n):
		# 	out = cgutils.create_struct_proxy(interval_t)(context, builder)

		# 	with builder.if_else(builder.icmp_signed('==', n, zero_i)) as (true, false):
		# 		with true:
		# 			out.lo = zero
		# 			out.hi = zero
		# 		with false:
		# 			pow_body = builder.append_basic_block("pow_interval_int.body")
		# 			pow_end = builder.append_basic_block("pow_interval_int.end")

		# 			m = cgutils.alloca_once_value(builder, n)

		# 			builder.branch(pow_body)
		# 			with builder.goto_block(pow_body):
		# 				with builder.if_then(builder.icmp_signed('>', builder.load(m), one_i)):
		# 					builder.store(sub_(builder, [builder.load(m), one_i]), m)

		# 					builder.branch(pow_body)

		# 				out.lo = zero
		# 				out.hi = zero



		# 			# 	# # n = 1
		# 			# 	# # x = interval(self.lo, self.hi)
		# 			# 	# 	# x = 1 / self
		# 			# 	# 	# n = -n

		# 			# 	# # while n > 1:
		# 			# 	# # 	if n % 2 == 0:
		# 			# 	# # 		x = x * x
		# 			# 	# # 		n = n / 2
		# 			# 	# # 	else:
		# 			# 	# # 		y = x * y
		# 			# 	# # 		x = x * x
		# 			# 	# # 		n = (n - 1) / 2
		# 			# 	# # return x * y

		# 				builder.branch(pow_end)
		# 			builder.position_at_end(pow_end)

		# 	return out._getvalue()
		# return pow(*intervals)




			# # negate = cgutils.alloca_once_value(builder, inactive)

			# # out = cgutils.create_struct_proxy(interval_t)(context, builder)

			# #use goto branch to emulate recursion
			# builder.branch(cos_body)
			# with builder.goto_block(cos_body):
			# 	width = sub_ru(builder, [f.hi, f.lo])
			# 	with builder.if_else(builder.fcmp_unordered('>=', width, pi2.lo)) as (true, false):
			# 		# with true:
			# 		# 	out.lo = n_one
			# 		# 	out.hi = one
			# 		# with false:
			# 		# 	temp1 = trig_shift_negative_interval(context, builder, sig, args, f)
			# 		# 	temp2 = trig_mod_interval_interval(context, builder, sig, args, temp1, pi2)
			# 		# 	f.lo = temp2.lo
			# 		# 	f.hi = temp2.hi
			# 		# 	with builder.if_else(builder.fcmp_unordered('>=', f.lo, pi.hi)) as (true, false):
			# 		# 		with true:
			# 		# 			temp3 = sub_(builder, [f._getvalue(), pi._getvalue()])
			# 		# 			temp3 = cgutils.create_struct_proxy(interval_t)(context, builder, value = temp3)

			# 		# 			f.lo = temp3.lo
			# 		# 			f.hi = temp3.hi

			# 		# 			#flip negate bit so that on next loop through cos_body the interval is negated
			# 		# 			builder.store(builder.not_(builder.load(negate)), negate)
			# 		# 			builder.branch(cos_body)
			# 		# 		with false:
			# 		# 			cos_lo = val_rd(context, builder, cos_(builder, [cast(context, builder, f.lo)]))
			# 		# 			cos_hi = val_ru(context, builder, cos_(builder, [cast(context, builder, f.hi)]))
			# 		# 			with builder.if_else(builder.fcmp_unordered("<=", f.hi, pi.lo)) as (true, false):
			# 		# 				with true:
			# 		# 					out.lo = cos_hi
			# 		# 					out.hi = cos_lo
			# 		# 				with false:
			# 		# 					with builder.if_else(builder.fcmp_unordered("<=", f.hi, pi2.lo)) as (true, false):
			# 		# 						with true:
			# 		# 							out.lo = n_one
			# 		# 							out.hi = max_(builder, [cos_lo, cos_hi])
			# 		# 						with false:
			# 		# 							out.lo = n_one
			# 		# 							out.hi = one
			# 	builder.branch(cos_end)
			# builder.position_at_end(cos_end)


















	# def pow_interval_float(context, builder, sig, args, *intervals):
		# log = context.get_function(mathfuncs.log, signature(*([interval_t]*2)))
		# mul = context.get_function(operator.mul, sig)
		# exp = context.get_function(mathfuncs.exp, signature(*([interval_t]*2)))
		# def pow(x,y,z):
		# 	# return exp(log(x)*y)
		# 	a = cgutils.create_struct_proxy(interval_t)(context, builder, value = log(builder, [x._getvalue()]))
		# 	b = cgutils.create_struct_proxy(interval_t)(context, builder, value = mul(builder, [a._getvalue(), y]))
		# 	c = cgutils.create_struct_proxy(interval_t)(context, builder, value = exp(builder, [b._getvalue()]))
		# 	z.lo = c.lo
		# 	z.hi = c.hi
		# 	return z
		# return pow(*intervals)

	# def pow_scalar_interval(context, builder, sig, args, *intervals):
		# def pow(x,y,z):
		# 	return z
		# return pow(*intervals)







	def mod_interval_interval(context, builder, sig, args, *intervals):
		zero = const_zero(context)

		# div_ = context.get_function(operator.truediv, signature(*([interval_t._member_t]*3)))
		# ceil_ = context.get_function(mathfuncs.ceil, signature(*([interval_t._member_t]*2)))
		# floor_ = context.get_function(mathfuncs.floor, signature(*([interval_t._member_t]*2)))

		# mul_ = context.get_function(operator.mul, signature(*([interval_t, interval_t, interval_t._member_t])))
		# sub_ = context.get_function(operator.sub, signature(*([interval_t]*3)))

		div_ = context.get_function(operator.truediv, signature(*([interval_t]*3)))
		ceil_ = context.get_function(mathfuncs.ceil, signature(*([interval_t]*2)))
		floor_ = context.get_function(mathfuncs.floor, signature(*([interval_t]*2)))

		mul_ = context.get_function(operator.mul, signature(*([interval_t, interval_t, interval_t])))
		sub_ = context.get_function(operator.sub, signature(*([interval_t]*3)))

		def mod(f,g):
			# den = builder.alloca(f.lo.type, 1)
			# with builder.if_else(builder.fcmp_unordered('<', f.lo, zero)) as (true, false):
			# 	with true:
			# 		builder.store(g.hi, den)
			# 	with false:
			# 		builder.store(g.lo, den)

			# n = cgutils.alloca_once_value(builder, div_(builder, [f.lo, builder.load(den)]))
			# # builder.store(floor_(builder, [builder.load(n)]), n)

			# # with builder.if_else(builder.fcmp_unordered('<', builder.load(n), zero)) as (true, false):
			# # 	with true:
			# # 		builder.store(ceil_(builder, [builder.load(n)]), n)
			# # 	with false:
			# # 		builder.store(floor_(builder, [builder.load(n)]), n)

			# return sub_(builder, [f._getvalue(), mul_(builder, [g._getvalue(), builder.load(n)])])

			n = floor_(builder, [div_(builder, [f._getvalue(), g._getvalue()])])
			return sub_(builder, [f._getvalue(), mul_(builder, [n, g._getvalue()])])

		return mod(*intervals)

	def mod_interval_scalar(context, builder, sig, args, *intervals):
		mod_ = context.get_function(mathfuncs.mod, signature(*([interval_t]*3)))

		def mod(x, y):
			temp = cgutils.create_struct_proxy(interval_t)(context, builder)
			temp.lo = y
			temp.hi = y

			return mod_(builder, [x._getvalue(), temp._getvalue()])
		return mod(*intervals)

	def mod_scalar_interval(context, builder, sig, args, *intervals):
		mod_ = context.get_function(mathfuncs.mod, signature(*([interval_t]*3)))

		def mod(x, y):
			temp = cgutils.create_struct_proxy(interval_t)(context, builder)
			temp.lo = x
			temp.hi = x

			return mod_(builder, [temp._getvalue(), y._getvalue()])
		return mod(*intervals)





	def fmod_interval_interval(context, builder, sig, args, *intervals):
		pass

	def fmod_interval_scalar(context, builder, sig, args, *intervals):
		#https://stackoverflow.com/questions/31057473/calculating-the-modulo-of-two-intervals

		zero = const_zero(context)
		one = const_one(context)

		sub_rd = op_rd(context, 'sub')
		sub_ru = op_ru(context, 'sub')

		abs_ = context.get_function(abs, signature(*([interval_t._member_t]*2)))
		neg_ = context.get_function(operator.neg, signature(*([interval_t._member_t]*2)))
		min_ = context.get_function(min, signature(*[interval_t._member_t]*3))
		max_ = context.get_function(max, signature(*[interval_t._member_t]*3))

		fmod_ = context.get_function(mathfuncs.fmod, signature(*([interval_t._member_t]*3)))

		def fmod(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			width = abs_(builder, [sub_ru(builder, [f.hi, f.lo])])
			g_pos = abs_(builder, [g])
			g_neg = neg_(builder, [g_pos])

			lo = fmod_(builder, [f.lo, g])
			hi = fmod_(builder, [f.hi, g])

			below_zero = builder.fcmp_unordered('<', f.hi, zero)
			with builder.if_else(below_zero) as (true, false):
				#return -fmod([-f.lo,-f.hi], g)
				with true:
					lo_neg = neg_(builder, [hi])
					hi_neg = neg_(builder, [lo])
					modulo_width = builder.and_(builder.fcmp_unordered('<', width, g_pos),
												builder.fcmp_unordered('<=', lo_neg, hi_neg))
					with builder.if_else(modulo_width) as (true_, false_):
						with true_:
							out.lo = neg_(builder, [max_(builder, [lo_neg, hi_neg])])
							out.hi = neg_(builder, [min_(builder, [lo_neg, hi_neg])])
						#return [-g, 0]
						with false_:
							out.lo = g_neg
							out.hi = zero
				with false:
					split_zero = builder.fcmp_unordered('<', f.lo, zero)
					with builder.if_else(split_zero) as (true, false):
						#return [max(f.lo % g, -g), min(f.hi % g, g)]
						with true:
							out.lo = max_(builder, [g_neg, lo])
							out.hi = min_(builder, [g_pos, hi])
						with false:
							modulo_width = builder.and_(builder.fcmp_unordered('<', width, g_pos),
														builder.fcmp_unordered('<=', lo, hi))
							with builder.if_else(modulo_width) as (true, false):
								#return [f.lo % g, f.hi % g]
								with true:
									out.lo = min_(builder, [lo, hi])
									out.hi = max_(builder, [lo, hi])
								#return [0, g]
								with false:
									out.lo = zero
									out.hi = g_pos

			return out._getvalue()
		return fmod(*intervals)

	def min_interval_interval(context, builder, sig, args, *intervals):
		min_ = context.get_function(min, signature(*[interval_t._member_t]*3))

		def minimum(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = min_(builder, [f.lo, g.lo])
			out.hi = min_(builder, [f.hi, g.hi])

			return out._getvalue()
		return minimum(*intervals)

	def min_interval_scalar(context, builder, sig, args, *intervals):
		min_ = context.get_function(min, signature(*[interval_t._member_t]*3))

		def minimum(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = min_(builder, [f.lo, g])
			out.hi = min_(builder, [f.hi, g])

			return out._getvalue()
		return minimum(*intervals)

	def min_scalar_interval(context, builder, sig, args, *intervals):
		min_ = context.get_function(min, signature(*[interval_t, interval_t, interval_t._member_t]))

		def minimum(f,g):
			return min_(builder, [g._getvalue(), f])
		return minimum(*intervals)

	def max_interval_interval(context, builder, sig, args, *intervals):
		max_ = context.get_function(max, signature(*[interval_t._member_t]*3))

		def maximum(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = max_(builder, [f.lo, g.lo])
			out.hi = max_(builder, [f.hi, g.hi])

			return out._getvalue()
		return maximum(*intervals)

	def max_interval_scalar(context, builder, sig, args, *intervals):
		max_ = context.get_function(max, signature(*[interval_t._member_t]*3))

		def maximum(f,g):
			out = cgutils.create_struct_proxy(interval_t)(context, builder)

			out.lo = max_(builder, [f.lo, g])
			out.hi = max_(builder, [f.hi, g])

			return out._getvalue()
		return maximum(*intervals)

	def max_scalar_interval(context, builder, sig, args, *intervals):
		max_ = context.get_function(max, signature(*[interval_t, interval_t, interval_t._member_t]))

		def maximum(f,g):
			return max_(builder, [g._getvalue(), f])
		return maximum(*intervals)


	# def step_interval_interval(context, builder, sig, args, *intervals):
		# zero = context.get_constant(sig.return_type, 0)
		# one = context.get_constant(sig.return_type, 1)

		# lt = context.get_function(operator.lt, sig)
		# def step(edge, x):
		# 	out = cgutils.alloca_once_value(builder, zero)
		# 	with builder.if_then(lt(builder, [edge._getvalue(), x._getvalue()])):
		# 		builder.store(one, out)
		# 	return builder.load(out)
		# return step(*intervals)

	# def step_interval_scalar(context, builder, sig, args, *intervals):
		# zero = context.get_constant(sig.return_type, 0)
		# one = context.get_constant(sig.return_type, 1)

		# lt = context.get_function(operator.lt, sig)
		# def step(edge, x):
		# 	out = cgutils.alloca_once_value(builder, zero)
		# 	temp = lt(builder, [edge._getvalue(), x])
		# 	with builder.if_then(temp):
		# 		builder.store(one, out)
		# 	return builder.load(out)
		# return step(*intervals)

	# def step_scalar_interval(context, builder, sig, args, *intervals):
		# zero = context.get_constant(sig.return_type, 0)
		# one = context.get_constant(sig.return_type, 1)

		# lt = context.get_function(operator.lt, sig)
		# def step(edge, x):
		# 	out = cgutils.alloca_once_value(builder, zero)
		# 	with builder.if_then(lt(builder, [edge, x._getvalue()])):
		# 		builder.store(one, out)
		# 	return builder.load(out)
		# return step(*intervals)

	ops = [(operator.add, [add_interval_interval, add_interval_scalar, add_scalar_interval]),
		   (operator.sub, [sub_interval_interval, sub_interval_scalar, sub_scalar_interval]),
		   (operator.mul, [mul_interval_interval, mul_interval_scalar, mul_scalar_interval]),
		   (operator.truediv, [div_interval_interval, div_interval_scalar, div_scalar_interval]),
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

	op = mathfuncs.mod
	binary_op_interval_interval(op, mod_interval_interval)
	binary_op_interval_scalar(op, mod_interval_scalar)
	binary_op_scalar_interval(op, mod_scalar_interval)

	binary_op_interval_scalar(mathfuncs.fmod, fmod_interval_scalar)


	# binary_op_interval_interval(mathfuncs.step, step_interval_interval)
	# binary_op_interval_scalar(mathfuncs.step, step_interval_scalar)
	# binary_op_scalar_interval(mathfuncs.step, step_scalar_interval) 


	##############################################
	##############################################
	##############################################

	# class pow_op_template(ConcreteTemplate):
	# 	cases = [#signature(interval_t, interval_t, interval_t),
	# 			 # signature(interval_t, interval_t._member_t, interval_t),
	# 			 # signature(interval_t, interval_t, interval_t._member_t)
	# 			 signature(interval_t, interval_t, types.int64)]

	# def pow_op_interval_int(op, caller):
	# 	@impl_registry.lower(op, interval_t, types.Integer)
	# 	def cuda_op_interval_scalar(context, builder, sig, args):
	# 		f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
	# 		g = context.cast(builder, args[1], sig.args[1], types.int64)
	# 		return caller(context, builder, sig, args, f, g)


	# decl_registry.register_global(operator.pow)(pow_op_template)
	# # binary_op_interval_interval(operator.pow, pow_interval_interval)
	# pow_op_interval_int(operator.pow, pow_interval_int)
	# # binary_op_interval_scalar(operator.pow, pow_interval_float, types.Float)
	# # # binary_op_scalar_interval(operator.pow, pow_scalar_interval)

	##############################################
	##############################################
	##############################################

	# def binary_iop_interval_interval(op, caller):
	# 	@impl_registry.lower(op, interval_t, interval_t)
	# 	def cuda_op_interval_interval(context, builder, sig, args):
	# 		print(sig, args)

	# 		# f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
	# 		# g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
	# 		# return caller(context, builder, sig, args, f, g)

	# def binary_iop_interval_scalar(op, caller):
	# 	@impl_registry.lower(op, interval_t, types.Number)
	# 	def cuda_op_interval_scalar(context, builder, sig, args):
	# 		f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
	# 		g = context.cast(builder, args[1], sig.args[1], interval_t._member_t)
	# 		return caller(context, builder, sig, args, f, g)

	# def binary_iop_scalar_interval(op, caller):
	# 	@impl_registry.lower(op, types.Number, interval_t)
	# 	def cuda_op_scalar_interval(context, builder, sig, args):
	# 		f = context.cast(builder, args[0], sig.args[0], interval_t._member_t)
	# 		g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
	# 		return caller(context, builder, sig, args, f, g)

	# def iadd_interval_interval(context, builder, sig, args, *intervals):
	# 	pass
	# def iadd_interval_scalar(context, builder, sig, args, *intervals):
	# 	pass
	# def iadd_scalar_interval(context, builder, sig, args, *intervals):
	# 	pass

	# def isub_interval_interval(context, builder, sig, args, *intervals):
	# 	sub_rd = op_rd(context, 'sub')
	# 	sub_ru = op_ru(context, 'sub')

	# 	def sub(f,g):
	# 		# print(f,g)
	# 		# out = cgutils.create_struct_proxy(interval_t)(context, builder)

	# 		# out.lo = sub_rd(builder, [f.lo, g.hi])
	# 		# out.hi = sub_ru(builder, [f.hi, g.lo])

	# 		# f.lo = out.lo
	# 		# f.hi = out.hi

	# 		return f._getvalue()
	# 	return sub(*intervals)

	# def isub_interval_scalar(context, builder, sig, args, *intervals):
	# 	pass
	# def isub_scalar_interval(context, builder, sig, args, *intervals):
	# 	pass
	# def imul_interval_interval(context, builder, sig, args, *intervals):
	# 	pass
	# def imul_interval_scalar(context, builder, sig, args, *intervals):
	# 	pass
	# def imul_scalar_interval(context, builder, sig, args, *intervals):
	# 	pass
	# def idiv_interval_interval(context, builder, sig, args, *intervals):
	# 	pass
	# def idiv_interval_scalar(context, builder, sig, args, *intervals):
	# 	pass
	# def idiv_scalar_interval(context, builder, sig, args, *intervals):
	# 	pass
	# def pow_interval_interval(context, builder, sig, args, *intervals):
	# 	pass
	# def pow_interval_scalar(context, builder, sig, args, *intervals):
	# 	pass
	# def pow_scalar_interval(context, builder, sig, args, *intervals):
	# 	pass

	# ops = [#(operator.iadd, [iadd_interval_interval, iadd_interval_scalar, iadd_scalar_interval]),
	# 	   (operator.isub, [isub_interval_interval, isub_interval_scalar, isub_scalar_interval]),
	# 	   #(operator.imul, [imul_interval_interval, imul_interval_scalar, imul_scalar_interval]),
	# 	   #(operator.itruediv, [idiv_interval_interval, idiv_interval_scalar, idiv_scalar_interval]),
	# 	   ##(operator.ipow, [pow_interval_interval, pow_interval_scalar, pow_scalar_interval])
	# 	   ]

	ops = [operator.iadd,
		   operator.isub,
		   operator.imul,
		   operator.itruediv,
		   #operator.ipow
		   ]

	for op in ops:
		decl_registry.register_global(op)(binary_op_template)

		# interval_interval, interval_scalar, scalar_interval = callers
		# binary_iop_interval_interval(op, interval_interval)
		# binary_iop_interval_scalar(op, interval_scalar)
		# binary_iop_scalar_interval(op, scalar_interval) 

	def bool_op_interval_interval(op, caller):
		@impl_registry.lower(op, interval_t, interval_t)
		def cuda_op_interval_interval(context, builder, sig, args):
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			return caller(context, builder, sig, args, f, g)

	def bool_op_interval_scalar(op, caller):
		@impl_registry.lower(op, interval_t, types.Number)
		def cuda_op_interval_scalar(context, builder, sig, args):
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			g = context.cast(builder, args[1], sig.args[1], interval_t._member_t)
			return caller(context, builder, sig, args, f, g)

	def bool_op_scalar_interval(op, caller):
		@impl_registry.lower(op, types.Number, interval_t)
		def cuda_op_scalar_interval(context, builder, sig, args):
			f = context.cast(builder, args[0], sig.args[0], interval_t._member_t)
			g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			return caller(context, builder, sig, args, f, g)

	def eq_interval_interval(context, builder, sig, args, *intervals):
		def eq(f,g):
			return builder.and_(builder.fcmp_unordered("==", f.lo, g.lo), builder.fcmp_unordered("==", f.hi, g.hi))
		return eq(*intervals)

	def eq_interval_scalar(context, builder, sig, args, *intervals):
		def eq(f,g):
			return builder.and_(builder.fcmp_unordered("==", f.lo, g), builder.fcmp_unordered("==", f.hi, g))
		return eq(*intervals)

	def eq_scalar_interval(context, builder, sig, args, *intervals):
		def eq(f,g):
			return builder.and_(builder.fcmp_unordered("==", f, g.lo), builder.fcmp_unordered("==", f, g.hi))
		return eq(*intervals)

	def ne_interval_interval(context, builder, sig, args, *intervals):
		def ne(f,g):
			return builder.and_(builder.fcmp_unordered("!=", f.lo, g.lo), builder.fcmp_unordered("!=", f.hi, g.hi))
		return ne(*intervals)

	def ne_interval_scalar(context, builder, sig, args, *intervals):
		def ne(f,g):
			return builder.and_(builder.fcmp_unordered("!=", f.lo, g), builder.fcmp_unordered("!=", f.hi, g))
		return ne(*intervals)

	def ne_scalar_interval(context, builder, sig, args, *intervals):
		def ne(f,g):
			return builder.and_(builder.fcmp_unordered("!=", f, g.lo), builder.fcmp_unordered("!=", f, g.hi))
		return ne(*intervals)

	def gt_interval_interval(context, builder, sig, args, *intervals):
		def gt(f,g):
			return builder.fcmp_unordered(">=", f.lo, g.hi)
		return gt(*intervals)

	def gt_interval_scalar(context, builder, sig, args, *intervals):
		def gt(f,g):
			return builder.fcmp_unordered(">=", f.lo, g)
		return gt(*intervals)

	def gt_scalar_interval(context, builder, sig, args, *intervals):
		def gt(f,g):
			return builder.fcmp_unordered(">=", f, g.hi)
		return gt(*intervals)

	def ge_interval_interval(context, builder, sig, args, *intervals):
		def ge(f,g):
			return builder.fcmp_unordered(">", f.lo, g.hi)
		return ge(*intervals)

	def ge_interval_scalar(context, builder, sig, args, *intervals):
		def ge(f,g):
			return builder.fcmp_unordered(">", f.lo, g)
		return ge(*intervals)

	def ge_scalar_interval(context, builder, sig, args, *intervals):
		def ge(f,g):
			return builder.fcmp_unordered(">", f, g.hi)
		return ge(*intervals)

	def lt_interval_interval(context, builder, sig, args, *intervals):
		def lt(f,g):
			return builder.fcmp_unordered("<", f.hi, g.lo)
		return lt(*intervals)

	def lt_interval_scalar(context, builder, sig, args, *intervals):
		def lt(f,g):
			return builder.fcmp_unordered("<", f.hi, g)
		return lt(*intervals)

	def lt_scalar_interval(context, builder, sig, args, *intervals):
		def lt(f,g):
			return builder.fcmp_unordered("<", f, g.lo)
		return lt(*intervals)

	def le_interval_interval(context, builder, sig, args, *intervals):
		def le(f,g):
			return builder.fcmp_unordered("<=", f.hi, g.lo)
		return le(*intervals)

	def le_interval_scalar(context, builder, sig, args, *intervals):
		def le(f,g):
			return builder.fcmp_unordered("<=", f.hi, g)
		return le(*intervals)

	def le_scalar_interval(context, builder, sig, args, *intervals):
		def le(f,g):
			return builder.fcmp_unordered("<=", f, g.lo)
		return le(*intervals)

	class bool_op_template(ConcreteTemplate):
		cases = [signature(types.boolean, interval_t, interval_t),
				 signature(types.boolean, interval_t._member_t, interval_t),
				 signature(types.boolean, interval_t, interval_t._member_t)]

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


	def ternary_op_interval_interval_interval(op, caller):
		@impl_registry.lower(op, interval_t, interval_t, interval_t)
		def cuda_op_interval_interval(context, builder, sig, args):
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			h = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[2])
			return caller(context, builder, sig, args, f, g, h)

	def ternary_op_interval_interval_scalar(op, caller, scalar_t = types.Number):
		@impl_registry.lower(op, interval_t, interval_t, scalar_t)
		def cuda_op_interval_interval_scalar(context, builder, sig, args):
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			h = context.cast(builder, args[2], sig.args[2], interval_t._member_t)
			return caller(context, builder, sig, args, f, g, h)

	def ternary_op_interval_scalar_interval(op, caller, scalar_t = types.Number):
		@impl_registry.lower(op, interval_t, scalar_t, interval_t)
		def cuda_op_interval_scalar_interval(context, builder, sig, args):
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			g = context.cast(builder, args[1], sig.args[1], interval_t._member_t)
			h = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[2])
			return caller(context, builder, sig, args, f, g, h)

	def ternary_op_scalar_interval_interval(op, caller, scalar_t = types.Number):
		@impl_registry.lower(op, scalar_t, interval_t, interval_t)
		def cuda_op_scalar_interval_interval(context, builder, sig, args):
			f = context.cast(builder, args[0], sig.args[0], interval_t._member_t)
			g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			h = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[2])
			return caller(context, builder, sig, args, f, g, h)

	def ternary_op_interval_scalar_scalar(op, caller):
		@impl_registry.lower(op, interval_t, types.Number, types.Number)
		def cuda_op_interval_scalar_scalar(context, builder, sig, args):
			f = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[0])
			g = context.cast(builder, args[1], sig.args[1], interval_t._member_t)
			h = context.cast(builder, args[2], sig.args[2], interval_t._member_t)
			return caller(context, builder, sig, args, f, g, h)

	def ternary_op_scalar_interval_scalar(op, caller):
		@impl_registry.lower(op, types.Number, interval_t, types.Number)
		def cuda_op_scalar_interval_scalar(context, builder, sig, args):
			f = context.cast(builder, args[0], sig.args[0], interval_t._member_t)
			g = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[1])
			h = context.cast(builder, args[2], sig.args[2], interval_t._member_t)
			return caller(context, builder, sig, args, f, g, h)

	def ternary_op_scalar_scalar_interval(op, caller):
		@impl_registry.lower(op, types.Number, types.Number, interval_t)
		def cuda_op_scalar_scalar_interval(context, builder, sig, args):
			f = context.cast(builder, args[0], sig.args[0], interval_t._member_t)
			g = context.cast(builder, args[1], sig.args[1], interval_t._member_t)
			h = cgutils.create_struct_proxy(interval_t)(context, builder, value = args[2])
			return caller(context, builder, sig, args, f, g, h)

	def clamp_interval_scalar_scalar(context, builder, sig, args, *intervals):
		min_ = context.get_function(min, signature(*[interval_t, interval_t, interval_t._member_t]))
		max_ = context.get_function(max, signature(*[interval_t, interval_t, interval_t._member_t]))

		def clamp(f,lo,hi):
			# return f._getvalue()
			return min_(builder, [max_(builder, [f._getvalue(), lo]), hi])
		return clamp(*intervals)

	def mix_interval_interval_scalar(context, builder, sig, args, *intervals):
		sub = context.get_function(operator.sub, signature(*[interval_t._member_t, interval_t._member_t, interval_t._member_t]))
		mul = context.get_function(operator.mul, signature(*[interval_t, interval_t, interval_t._member_t]))
		add = context.get_function(operator.add, signature(*[interval_t, interval_t, interval_t]))

		one = context.get_constant(interval_t._member_t, 1)

		def mix(x, y, a):
			#return x*(1.0 - a) + y*a
			return add(builder, [mul(builder, [x._getvalue(), sub(builder,[one, a])]), mul(builder, [y._getvalue(), a])])
		return mix(*intervals)

	def mix_interval_interval_interval(context, builder, sig, args, *intervals):
		sub = context.get_function(operator.sub, signature(*[interval_t, interval_t._member_t, interval_t]))
		mul = context.get_function(operator.mul, signature(*[interval_t, interval_t, interval_t]))
		add = context.get_function(operator.add, signature(*[interval_t, interval_t, interval_t]))

		one = context.get_constant(interval_t._member_t, 1)

		def mix(x, y, a):
			#return x*(1.0 - a) + y*a
			return add(builder, [mul(builder, [x._getvalue(), sub(builder,[one, a._getvalue()])]), mul(builder, [y._getvalue(), a._getvalue()])])
		return mix(*intervals)




	# class ternary_op_template_interval_scalar_scalar(ConcreteTemplate):
	# 	cases = [signature(interval_t, interval_t._member_t, interval_t._member_t)]

	# class ternary_op_template_interval_interval_scalar(ConcreteTemplate):
	# 	cases = [signature(interval_t, interval_t, interval_t._member_t)]

	# decl_registry.register_global(mathfuncs.clamp)(ternary_op_template_interval_scalar_scalar)
	ternary_op_interval_scalar_scalar(mathfuncs.clamp, clamp_interval_scalar_scalar)

	ternary_op_interval_interval_scalar(mathfuncs.mix, mix_interval_interval_scalar)
	ternary_op_interval_interval_interval(mathfuncs.mix, mix_interval_interval_interval)






	# def lerp_interval_scalar_scalar(context, builder, sig, args, *intervals):
	# 	pass


	# ternary_floating_ops = ["lerp"] #"clamp", "lerp", "param", "smooth_step"



	# def lerp_interval_interval_interval(context, builder, sig, args, *intervals):
		# add = context.get_function(operator.add, signature(*[sig.return_type for n in range(3)]))
		# sub = context.get_function(operator.sub, signature(*[sig.return_type for n in range(3)]))
		# mul = context.get_function(operator.mul, signature(*[sig.return_type for n in range(3)]))

		# def lerp(start,end,t):
		# 	out = cgutils.create_struct_proxy(interval_t)(context, builder)
		# 	sub_interval_interval(context, builder, sig, args, end, start, out)
		# 	temp = cgutils.create_struct_proxy(interval_t)(context, builder)
		# 	mul_interval_interval(context, builder, sig, args, t, out, temp)
		# 	add_interval_interval(context, builder, sig, args, start, temp, out)
		# 	return out
		# 	# return add(builder, [start, mul(builder, [t, sub(builder, [end, start])])])
		# return lerp(*intervals)

	# ternary_op_interval_interval_interval(mathfuncs.lerp, lerp_interval_interval_interval)

	# def clamp_interval_interval_interval(context, builder, sig, args, *intervals):
		# # minf = context.get_function(min, signature([interval_t, interval_t, interval_t._member_t]))
		# minf = context.get_function(min, signature([interval_t, interval_t, interval_t]))
		# maxf = context.get_function(max, signature([interval_t, interval_t, interval_t]))

		# def clamp(f,lo,hi):
		# 	return minf(builder, [maxf(builder, [f, lo]), hi])

		# # maximum = context.get_function(mathfuncs.fmax, signature(*[sig.return_type for n in range(3)]))
		# # minimum = context.get_function(mathfuncs.fmin, signature(*[sig.return_type for n in range(3)]))
		# # def clamp(f,lo,hi):
		# # 	out = cgutils.create_struct_proxy(interval_t)(context, builder)
		# # 	out.lo = minimum(builder, [maximum(builder, [f.lo, lo]), hi])
		# # 	out.hi = minimum(builder, [maximum(builder, [f.hi, lo]), hi])
		# # 	return out

	# ternary_op_interval_interval_interval(mathfuncs.clamp, clamp_interval_interval_interval)





	# for op in ternary_floating_ops:
	# 	mathop = getattr(mathfuncs, op)
	# ternary_op_interval_interval_interval(mathfuncs.lerp, lerp_interval_interval_interval)
		# ternary_op_interval_interval_scalar(mathop, )
		# ternary_op_interval_scalar_interval(mathop, )
		# ternary_op_scalar_interval_interval(mathop, )
		# ternary_op_interval_scalar_scalar(mathop, )
		# ternary_op_scalar_interval_scalar(mathop, )
		# ternary_op_scalar_scalar_interval(mathop, )


	# libfunc_helper_wrapper(libop)


def op_rd_32(context, op, sig = signature(types.float32, types.float32, types.float32), lib = mathfuncs):
	return context.get_function(getattr(lib, "f" + op + "_rd"), sig)

def op_ru_32(context, op, sig = signature(types.float32, types.float32, types.float32), lib = mathfuncs):
	return context.get_function(getattr(lib, "f" + op + "_ru"), sig)

def val_rd_32(context, builder, lo):
	double2float_rd = context.get_function(mathfuncs.double2float_rd, signature(types.float32, types.float64))
	return double2float_rd(builder, [lo])

def val_ru_32(context, builder, hi):
	double2float_ru = context.get_function(mathfuncs.double2float_ru, signature(types.float32, types.float64))
	return double2float_ru(builder, [hi])

def op_rd_64(context, op, sig = signature(types.float64, types.float64, types.float64), lib = mathfuncs):
	return context.get_function(getattr(lib, "d" + op + "_rd"), sig)

def op_ru_64(context, op, sig = signature(types.float64, types.float64, types.float64), lib = mathfuncs):
	return context.get_function(getattr(lib, "d" + op + "_ru"), sig)

def val_rd_64(context, builder, lo):
	nextafter = context.get_function(mathfuncs.nextafter, signature(types.float64, types.float64, types.float64))
	return nextafter(builder, [lo, context.get_constant(types.float64, np.float64(-np.inf))])

def val_ru_64(context, builder, hi):
	nextafter = context.get_function(mathfuncs.nextafter, signature(types.float64, types.float64, types.float64))
	return nextafter(builder, [hi, context.get_constant(types.float64, np.float64(np.inf))])

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
from ..vector.vectorimpl import (vec_decl, allocate_vector,
								 vectorize_vector, vectorize_member, vectorize_scalar_cast,
								 libfunc_vectorized_op_caller_wrapper, libfunc_reduction_op_caller_wrapper,
								 shift_binary_op_caller, dot_binary_op_caller, length_reduction_op_caller,
								 unary_op_vectorized_wrapper, unary_op_wrapper,
								 binary_op_vectorized_vector_vector_wrapper, binary_op_vectorized_vector_other_wrapper,
								 binary_op_vectorized_other_vector_wrapper, binary_op_vectorized_wrapper,
								 binary_op_vector_vector_wrapper, binary_op_vector_other_wrapper, binary_op_other_vector_wrapper,
								 ternary_op_vectorized_vec_vec_vec_wrapper,
								 ternary_op_vectorized_vec_vec_other_wrapper, ternary_op_vectorized_vec_other_vec_wrapper, ternary_op_vectorized_other_vec_vec_wrapper,
								 ternary_op_vectorized_vec_other_other_wrapper, ternary_op_vectorized_other_vec_other_wrapper, ternary_op_vectorized_other_other_vec_wrapper)

unary_floating_ops = ["sin", "cos", "tan",
					  "sinh", "cosh", "tanh",
					  "asin", "acos", "atan",
					  "asinh", "acosh", "atanh",
					  "sqrt", #"rsqrt", "cbrt", "rcbrt",
					  "exp" , "exp10", "exp2", #"expm1",
					  "log", "log10", "log2", #"log1p", "logb",
					  "floor", "ceil"]

binary_floating_ops = ["mod"] #"step"

ternary_floating_ops = ["clamp", "mix", "lerp", "param", "smooth_step"]

# def allocate_vector(context, builder, vec_t):
# 	OutProxy = cgutils.create_struct_proxy(vec_t, 'data')
# 	#convert numba type to ir type
# 	datamodel = context.data_model_manager[OutProxy._fe_type]

# 	out_ptr = cgutils.alloca_once(builder, datamodel.get_data_type(), size = 1)
# 	out = OutProxy(context, builder, ref = out_ptr)
# 	return out

def interval_vec_decl(vec, vec_t, Vec, floating_vec_t):
	floating_t = floating_vec_t._member_t

	#initialize: vec(x)
	@lower_builtin(vec, types.Number)
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			member = cgutils.create_struct_proxy(vec_t._member_t)(context, builder)
			member.lo = context.cast(builder, args[0], sig.args[0], floating_t)
			member.hi = context.cast(builder, args[0], sig.args[0], floating_t)
			setattr(out, attr, member._getvalue())

		return out._getpointer()

	#initialize: vec(x,y,...)
	@lower_builtin(vec, *[types.Number]*len(vec_t._members))
	def impl_vec(context, builder, sig, args):
		out = allocate_vector(context, builder, sig.return_type)

		for n, attr in enumerate(sig.return_type._fields):
			member = cgutils.create_struct_proxy(vec_t._member_t)(context, builder)
			member.lo = context.cast(builder, args[n], sig.args[n], floating_t)
			member.hi = context.cast(builder, args[n], sig.args[n], floating_t)
			setattr(out, attr, member._getvalue())

		return out._getpointer()

	if Vec == Vector2Type:
		#interval, scalar
		#scalar, interval
		pass

	if Vec == Vector3Type:
		#interval, interval, scalar
		#interval, scalar, interval
		#scalar, interval, interval
		#interval, scalar, scalar
		#scalar, interval, scalar
		#scalar, scalar, interval
		pass

		# #initialize: vec(x, vec2)
		# @lower_builtin(vec, type(vec_t._member_t), Vector2Type)
		# def impl_vec(context, builder, sig, args):
		# 	vec = cgutils.create_struct_proxy(sig.args[1], 'data')(context, builder, ref = args[1])
		# 	vals = [context.cast(builder, args[0], sig.args[0], sig.return_type._member_t),
		# 			context.cast(builder, vec.x, sig.args[1]._member_t, sig.return_type._member_t),
		# 			context.cast(builder, vec.y, sig.args[1]._member_t, sig.return_type._member_t)]

		# 	out = allocate_vector(context, builder, sig.return_type)

		# 	for n, attr in enumerate(sig.return_type._fields):
		# 		setattr(out, attr, vals[n])

		# 	return out._getpointer()

		# #initialize: vec(vec2, x)
		# @lower_builtin(vec, Vector2Type, type(vec_t._member_t))
		# def impl_vec(context, builder, sig, args):
		# 	vec = cgutils.create_struct_proxy(sig.args[0], 'data')(context, builder, ref = args[0])
		# 	vals = [context.cast(builder, vec.x, sig.args[0]._member_t, sig.return_type._member_t),
		# 			context.cast(builder, vec.y, sig.args[0]._member_t, sig.return_type._member_t),
		# 			context.cast(builder, args[1], sig.args[1], sig.return_type._member_t)]

		# 	out = allocate_vector(context, builder, sig.return_type)

		# 	for n, attr in enumerate(sig.return_type._fields):
		# 		setattr(out, attr, vals[n])

		# 	return out._getpointer()

	vec_pair = (vec_t, vectorize_vector)
	member_pair = (vec_t._member_t, vectorize_member)
	floating_vec_pair = (floating_vec_t, vectorize_vector)
	floating_pair = (types.Number, vectorize_scalar_cast(floating_t))

	#####
	#+, -, *, /
	#####
		# +=, -=, *=, /= (iadd,isub,imul,idiv)

	class vec_op_template(ConcreteTemplate):
		cases = [signature(vec_t, vec_t, vec_t),

				 signature(vec_t, vec_t._member_t, vec_t),
				 signature(vec_t, vec_t, vec_t._member_t),

				 signature(vec_t, floating_vec_t, vec_t),
				 signature(vec_t, vec_t, floating_vec_t),

				 signature(vec_t, floating_vec_t, vec_t._member_t),
				 signature(vec_t, vec_t._member_t, floating_vec_t),

				 signature(vec_t, floating_t, vec_t),
				 signature(vec_t, vec_t, floating_t)]

	for op in [operator.add, operator.sub, operator.mul, operator.truediv]:
		decl_registry.register_global(op)(vec_op_template)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)

		#op(self, self)
		binary_op_vectorized_vector_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair)
		#op(self, member)
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair)
		#op(self, floating3)
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)
		#op(self, scalar)
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)
		#op(floating3, member)
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, member_pair, floating_vec_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, member_pair, floating_vec_pair)

  #[operator.iadd, operator.isub, operator.imul, operator.itruediv]


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

				 signature(vec_t, floating_vec_t, vec_t),
				 signature(vec_t, vec_t, floating_vec_t),

				 signature(vec_t, floating_t, vec_t),
				 signature(vec_t, vec_t, floating_t),

				 #reduction
				 signature(vec_t._member_t, vec_t)]

	for op in [min, max]:
		decl_registry.register_global(op)(vec_op_template)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		libfunc_reduction_op_caller = libfunc_reduction_op_caller_wrapper(op)

		#vector
		binary_op_vectorized_vector_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair)
		#member
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair)
		#float3/double3
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)
		#scalar
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)

		#reduction
		unary_op_vectorized_wrapper(op, libfunc_reduction_op_caller, vec_pair)

	#####
	#sum
	#####

	class vec_op_template(ConcreteTemplate):
		cases = [signature(vec_t._member_t, vec_t)]

	decl_registry.register_global(sum)(vec_op_template)
	libfunc_reduction_op_caller = libfunc_reduction_op_caller_wrapper(operator.add)
	unary_op_vectorized_wrapper(sum, libfunc_reduction_op_caller, vec_pair)

	#####
	#dot
	#####

	binary_op_vector_vector_wrapper(mathfuncs.dot, dot_binary_op_caller, vec_t)
	binary_op_vector_vector_wrapper(mathfuncs.dot, dot_binary_op_caller, vec_t, floating_vec_t)
	binary_op_vector_vector_wrapper(mathfuncs.dot, dot_binary_op_caller, floating_vec_t, vec_t)

	#####
	#shift
	#####

	for scalar_type in [types.uint8, types.uint16, types.uint32, types.uint64, types.int8, types.int16, types.int32, types.int64]:
		binary_op_vector_other_wrapper(mathfuncs.shift, shift_binary_op_caller, vec_t, scalar_type)

	#####
	#neg, abs
	#####

	class vec_unary_op(ConcreteTemplate):
		cases = [signature(vec_t, vec_t)]

	for op in [operator.neg, abs]:
		decl_registry.register_global(op)(vec_unary_op)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		unary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, vec_pair)

	#####
	#unary functions
	#####

	#float/double vector
	for op in unary_floating_ops:
		op = getattr(mathfuncs, op)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)
		unary_op_vectorized_wrapper(op, libfunc_vectorized_op_caller, vec_pair)

	#####
	#length
	#####

	unary_op_wrapper(mathfuncs.length, length_reduction_op_caller, vec_t)

	#####
	#round
	#####

	class vec_unary_op(ConcreteTemplate):
		cases = [signature(vec_t, vec_t)]

	decl_registry.register_global(round)(vec_unary_op)

	libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(round)
	unary_op_vectorized_wrapper(round, libfunc_vectorized_op_caller, vec_pair)

	#####
	#binary functions
	#####

	for op in binary_floating_ops:
		op = getattr(mathfuncs, op)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)

		binary_op_vectorized_vector_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair)
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair)
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)
		binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)
		binary_op_vectorized_other_vector_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)

	#fmod only supports intervals as the first argument
	op = mathfuncs.fmod

	libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)

	binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)
	binary_op_vectorized_vector_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)

	#####
	#ternary functions
	#####

	for op in ternary_floating_ops:
		op = getattr(mathfuncs, op)

		libfunc_vectorized_op_caller = libfunc_vectorized_op_caller_wrapper(op)

		ternary_op_vectorized_vec_vec_vec_wrapper(op, libfunc_vectorized_op_caller, vec_pair)

		ternary_op_vectorized_vec_vec_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair)
		ternary_op_vectorized_vec_vec_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair)

		ternary_op_vectorized_vec_other_vec_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)
		ternary_op_vectorized_vec_other_vec_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair)

		ternary_op_vectorized_other_vec_vec_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)
		ternary_op_vectorized_other_vec_vec_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair)

		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair, member_pair)
		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair, floating_vec_pair)
		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, member_pair, floating_pair)
		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair, member_pair)
		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair, floating_vec_pair)
		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_vec_pair, floating_pair)
		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair, member_pair)
		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair, floating_vec_pair)
		ternary_op_vectorized_vec_other_other_wrapper(op, libfunc_vectorized_op_caller, vec_pair, floating_pair, floating_pair)

		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, member_pair, vec_pair, member_pair)
		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, member_pair, vec_pair, floating_vec_pair)
		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, member_pair, vec_pair, floating_pair)
		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, floating_vec_pair, vec_pair, member_pair)
		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, floating_vec_pair, vec_pair, floating_vec_pair)
		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, floating_vec_pair, vec_pair, floating_pair)
		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, floating_pair, vec_pair, member_pair)
		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, floating_pair, vec_pair, floating_vec_pair)
		ternary_op_vectorized_other_vec_other_wrapper(op, libfunc_vectorized_op_caller, floating_pair, vec_pair, floating_pair)

		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, member_pair, member_pair, vec_pair)
		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, member_pair, floating_vec_pair, vec_pair)
		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, member_pair, floating_pair, vec_pair)
		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, floating_vec_pair, member_pair, vec_pair)
		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, floating_vec_pair, floating_vec_pair, vec_pair)
		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, floating_vec_pair, floating_pair, vec_pair)
		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, floating_pair, member_pair, vec_pair)
		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, floating_pair, floating_vec_pair, vec_pair)
		ternary_op_vectorized_other_other_vec_wrapper(op, libfunc_vectorized_op_caller, floating_pair, floating_pair, vec_pair)

def vec_2_typer_wrapper(vec_t, floating_vec_t): #, Vec, FloatingVec
	floating_t = floating_vec_t._member_t
	# print(floating_t, type(floating_t), types.Number)
	def vec_2_typer(*attrs):
		#initialize: vec()
		if len(attrs) == 0:
			return vec_t
		#initialize: vec(x) or vec(vec)
		elif len(attrs) == 1:
			vec_or_input_t_or_scalar = attrs
			if isinstance(vec_or_input_t_or_scalar, type(vec_t._member_t)) or isinstance(vec_or_input_t_or_scalar, types.Number) or isinstance(vec_or_input_t_or_scalar, Vec):
				return vec_t
			else:
				raise ValueError(f"Input to {vec_t.__class__.__name__} not understood")
		#initialize: vec2(x,y)
		elif len(attrs) == 2:
			if all([isinstance(attr, type(vec_t._member_t)) or isinstance(attr, types.Number) for attr in attrs]):
				return vec_t
			else:
				raise ValueError(f"Input to {vec_t.__class__.__name__} not understood")
	return vec_2_typer

def vec_3_typer_wrapper(vec_t, floating_vec_t): #, Vec, FloatingVec
	floating_t = floating_vec_t._member_t
	def vec_3_typer(*attrs):
		#initialize: vec()
		if len(attrs) == 0:
			return vec_t
		#initialize: vec(x) or vec(vec)
		elif len(attrs) == 1:
			vec_or_input_t_or_scalar = attrs
			if isinstance(vec_or_input_t_or_scalar, type(vec_t._member_t)) or isinstance(vec_or_input_t_or_scalar, types.Number) or isinstance(vec_or_input_t_or_scalar, Vec):
				return vec_t
			else:
				raise ValueError(f"Input to {vec_t.__class__.__name__} not understood")
		#vec3(vec2,y) or vec3(x,vec2)
		elif len(attrs) == 2:
			vec_or_input_t_or_scalar_a, vec_or_input_t_or_scalar_b = attrs
			if (isinstance(vec_or_input_t_or_scalar_a, type(vec_t._member_t)) or isinstance(vec_or_input_t_or_scalar_a, types.Number)) and isinstance(vec_or_input_t_or_scalar_b, Vector2Type):
				return vec_t
			elif isinstance(vec_or_input_t_or_scalar_a, Vector2Type) and (isinstance(vec_or_input_t_or_scalar_b, type(vec_t._member_t)) or isinstance(vec_or_input_t_or_scalar_b, types.Number)):
				return vec_t
			else:
				raise ValueError(f"Input to {vec_t.__class__.__name__} not understood")
		# #initialize: vec3(x,y,z)
		elif len(attrs) == 3:
			if all([isinstance(attr, type(vec_t._member_t)) or isinstance(attr, types.Number) for attr in attrs]):
				return vec_t
			else:
				raise ValueError(f"Input to {vec_t.__class__.__name__} not understood")
	return vec_3_typer

vec2s = [intervalf2, intervald2]
vec3s = [intervalf3, intervald3]

vec2s_t = [intervalf2_t, intervald2_t]
vec3s_t = [intervalf3_t, intervald3_t]

floating2_t = [float2_t, double2_t]
floating3_t = [float3_t, double3_t]

vec_groups = [vec2s, vec3s]
vec_t_groups = [vec2s_t, vec3s_t]
Vecs = [Vector2Type, Vector3Type]
vec_typer_wrappers = [vec_2_typer_wrapper, vec_3_typer_wrapper]
floating_vec_groups = [floating2_t, floating3_t]

for vecs, vecs_t, Vec, vec_typer_wrapper, floatings_vec_t in zip(vec_groups, vec_t_groups, Vecs, vec_typer_wrappers, floating_vec_groups):
	for vec, vec_t, floating_vec_t in zip(vecs, vecs_t, floatings_vec_t):
		vec_decl(vec, vec_t, Vec, vec_typer_wrapper(vec_t, floating_vec_t))
		interval_vec_decl(vec, vec_t, Vec, floating_vec_t)
