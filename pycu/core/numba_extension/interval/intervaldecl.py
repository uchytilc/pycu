from ..vector.vectordecl import FloatVectorType, DoubleVectorType, Vector2Type, Vector3Type, float3_t, double3_t

from .intervaltype import (intervalf, intervald,
						   intervalf2, intervalf3,
						   intervald2, intervald3)

from numba import types

class IntervalType(types.Type):
	_fields = ['lo', 'hi']
	def __init__(self, name = 'interval'):
		super().__init__(name = name)

	@property
	def members(self):
		return [(name, self._member_t) for name in self._fields]

	# @property
	# def bitwidth(self):
	# 	return self._member_t.bitwidth

	def typer_round(self, context):
		return self
	def typer_ceil(self, context):
		return self
	def typer_floor(self, context):
		return self
	def typer_abs(self, context, other):
		return self
	def typer_sin(self, context):
		return self
	def typer_cos(self, context):
		return self
	def typer_tan(self, context):
		return self
	def typer_asin(self, context):
		return self
	def typer_acos(self, context):
		return self
	def typer_atan(self, context):
		return self
	def typer_sinh(self, context):
		return self
	def typer_cosh(self, context):
		return self
	def typer_tanh(self, context):
		return self
	def typer_asinh(self, context):
		return self
	def typer_acosh(self, context):
		return self
	def typer_atanh(self, context):
		return self
	# def typer_sinpi(self, context):
	# 	return self
	# def typer_cospi(self, context):
	# 	return self
	# # def typer_atan2(self, context):
	# #	return 
	# # def typer_sincos(self, context):
	# #	return 
	# # def typer_sincospi(self, context):
	# #	return 

	def typer_sqrt(self, context):
		return self
	# def typer_rsqrt(self, context):
	# 	return self
	# def typer_cbrt(self, context):
	# 	return self
	# def typer_rcbrt(self, context):
	# 	return self
	def typer_exp(self, context):
		return self
	def typer_exp10(self, context):
		return self
	def typer_exp2(self, context):
		return self
	# def typer_expm1(self, context):
	# 	return self
	def typer_log(self, context):
		return self
	# def typer_log1p(self, context):
	# 	return self
	def typer_log10(self, context):
		return self
	def typer_log2(self, context):
		return self
	# def typer_logb(self, context):
	# 	return self

	def typer_fmod(self, context, *other):
		return self
	def typer_mod(self, context, *other):
		return self


	def typer_clamp(self, context, *other):
		return self

	def typer_sign(self, context, *other):
		return self

	def typer_mix(self, context, *other):
		return self
	def typer_lerp(self, context, *other):
		return self

	def typer_min(self, context, *other):
		return self
	def typer_max(self, context, *other):
		return self

	def typer_step(self, context, *other):
		return self._member_t


class IntervalfType(IntervalType):
	_impl_t = intervalf
	_member_t = types.float32
	def __init__(self, name = 'intervalf'):
		super().__init__(name = name)

class IntervaldType(IntervalType):
	_impl_t = intervald
	_member_t = types.float64
	def __init__(self, name = 'intervald'):
		super().__init__(name = name)

	# def typer_fabs(self, context):
	# 	return self

	# def typer_fmin(self, context, *other):
	# 	return self
	# def typer_fmax(self, context, *other):
	# 	return self


intervalf_t = IntervalfType()
intervald_t = IntervaldType()
interval_t = intervalf_t




class IntervalVectorType:
	pass

class IntervalfVectorType(IntervalVectorType, FloatVectorType):
	_member_t = intervalf_t
	def __init__(self, name = 'intervalfvec'):
		super().__init__(name = name)

	def typer_step(self, context, *other):
		return float3_t

class IntervaldVectorType(IntervalVectorType, DoubleVectorType):
	_member_t = intervald_t
	def __init__(self, name = 'intervaldvec'):
		super().__init__(name = name)

	def typer_step(self, context, *other):
		return double3_t

class Intervalf2Type(Vector2Type, IntervalfVectorType):
	_impl_t = intervalf2
	def __init__(self, name = 'intervalf2'):
		super().__init__(name = name)

class Intervald2Type(Vector2Type, IntervaldVectorType):
	_impl_t = intervald2
	def __init__(self, name = 'intervald2'):
		super().__init__(name = name)


intervalf2_t = Intervalf2Type()
intervald2_t = Intervald2Type()
interval2_t = intervalf2_t


class Intervalf3Type(Vector3Type, IntervalfVectorType):
	_impl_t = intervalf3
	_vec2_t = intervalf2_t
	def __init__(self, name = 'intervalf3'):
		super().__init__(name = name)

class Intervald3Type(Vector3Type, IntervaldVectorType):
	_impl_t = intervald3
	_vec2_t = intervald2_t
	def __init__(self, name = 'intervald3'):
		super().__init__(name = name)

intervalf3_t = Intervalf3Type()
intervald3_t = Intervald3Type()
interval3_t = intervalf3_t
