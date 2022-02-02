from numba import types

from .vectortype import (char2, short2, int2, long2,
						uchar2, ushort2, uint2, ulong2,
						float2, double2,
						char3, short3, int3, long3,
						uchar3, ushort3, uint3, ulong3,
						float3, double3)

#To add typer behaviour for a given function and vector type add typer_FUNCTION and define the return type for the function

#_input_type
	#not happy with this name


class VectorType(types.Type):
	def __init__(self, name = 'vec'):
		super().__init__(name = name)

	@property
	def _members(self):
		return [(name, self._member_t) for name in self._fields]

	def typer_sum(self, context):
		return self._member_t

	def typer_dot(self, context, other):
		return self._member_t

	def typer_min(self, context, *other):
		#if unary max is called a scalar is returned
		if not other:
			return self._member_t
		return self

	def typer_max(self, context, *other):
		#if unary min is called a scalar is returned
		if not other:
			return self._member_t
		return self

	def typer_shift(self, context, *other):
		return self

vector_t = VectorType()

class NumberVectorType(VectorType):
	def __init__(self, name = 'number_vector'):
		super().__init__(name = name)

class IntegerVectorType(NumberVectorType):
	def __init__(self, name = 'integer_vector'):
		super().__init__(name = name)

class CharVectorType(IntegerVectorType):
	_member_t = types.int8
	def __init__(self, name = 'char_vector'):
		super().__init__(name = name)

class ShortVectorType(IntegerVectorType):
	_member_t = types.int16
	def __init__(self, name = 'short_vector'):
		super().__init__(name = name)

class IntVectorType(IntegerVectorType):
	_member_t = types.int32
	def __init__(self, name = 'int_vector'):
		super().__init__(name = name)

	def typer_abs(self, context):
		return self

class LongVectorType(IntegerVectorType):
	_member_t = types.int64
	def __init__(self, name = 'long_vector'):
		super().__init__(name = name)

	def typer_llabs(self, context):
		return self

	def typer_llmin(self, context, *other):
		if not other:
			return self._member_t
		return self

	def typer_llmax(self, context, *other):
		if not other:
			return self._member_t
		return self

class UCharVectorType(IntegerVectorType):
	_member_t = types.uint8
	def __init__(self, name = 'uchar_vector'):
		super().__init__(name = name)

class UShortVectorType(IntegerVectorType):
	_member_t = types.uint16
	def __init__(self, name = 'ushort_vector'):
		super().__init__(name = name)

class UIntVectorType(IntegerVectorType):
	_member_t = types.uint32
	def __init__(self, name = 'uint_vector'):
		super().__init__(name = name)

	def typer_umin(self, context, *other):
		if not other:
			return self._member_t
		return self

	def typer_umax(self, context, *other):
		if not other:
			return self._member_t
		return self

class ULongVectorType(IntegerVectorType):
	_member_t = types.uint64
	def __init__(self, name = 'ulong_vector'):
		super().__init__(name = name)

	def typer_ullmin(self, context, *other):
		if not other:
			return self._member_t
		return self

	def typer_ullmax(self, context, *other):
		if not other:
			return self._member_t
		return self


class FloatingVectorType(NumberVectorType):
	def __init__(self, name = 'floating_vector'):
		super().__init__(name = name)

	def typer_round(self, context):
		return self
	def typer_floor(self, context):
		return self
	def typer_ceil(self, context):
		return self

	def typer_length(self, context):
		return self._member_t

	def typer_abs(self, context):
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
	def typer_lerp(self, context, *other):
		return self
	def typer_param(self, context, *other):
		return self
	def typer_smooth_step(self, context, *other):
		return self

	# def typer_step(self, context, *other):
	# 	return self




class FloatVectorType(FloatingVectorType):
	_member_t = types.float32
	def __init__(self, name = 'float_vector'):
		super().__init__(name = name)

	# def typer_fabsf(self, context):
	# 	return self

	# def typer_fminf(self, context, *other):
	# 	if not other:
	# 		return self._member_t
	# 	return self
	# def typer_fmaxf(self, context, *other):
	# 	if not other:
	# 		return self._member_t
	# 	return self

	# def typer_sinf(self, context):
	# 	return self
	# def typer_cosf(self, context):
	# 	return self
	# def typer_tanf(self, context):
	# 	return self
	# def typer_asinf(self, context):
	# 	return self
	# def typer_acosf(self, context):
	# 	return self
	# def typer_atanf(self, context):
	# 	return self
	# def typer_sinhf(self, context):
	# 	return self
	# def typer_coshf(self, context):
	# 	return self
	# def typer_tanhf(self, context):
	# 	return self
	# def typer_asinhf(self, context):
	# 	return self
	# def typer_acoshf(self, context):
	# 	return self
	# def typer_atanhf(self, context):
	# 	return self
	# # def typer_sinpif(self, context):
	# #	return 
	# # def typer_cospif(self, context):
	# #	return 
	# # def typer_atan2f(self, context):
	# #	return 
	# # def typer_sincosf(self, context):
	# #	return 
	# # def typer_sincospif(self, context):
	# #	return 
	# def typer_sqrtf(self, context):
	# 	return self
	# def typer_rsqrtf(self, context):
	# 	return self
	# def typer_cbrtf(self, context):
	# 	return self
	# def typer_rcbrtf(self, context):
	# 	return self
	# def typer_expf(self, context):
	# 	return self
	# def typer_exp10f(self, context):
	# 	return self
	# def typer_exp2f(self, context):
	# 	return self
	# def typer_expm1f(self, context):
	# 	return self
	# def typer_logf(self, context):
	# 	return self
	# def typer_log1pf(self, context):
	# 	return self
	# def typer_log10f(self, context):
	# 	return self
	# def typer_log2f(self, context):
	# 	return self
	# def typer_logbf(self, context):
	# 	return self

	# def typer_fast_sinf(self, context):
	# 	return self
	# def typer_fast_cosf(self, context):
	# 	return self
	# def typer_fast_tanf(self, context):
	# 	return self
	# def typer_fast_expf(self, context):
	# 	return self
	# def typer_fast_exp10f(self, context):
	# 	return self
	# def typer_fast_logf(self, context):
	# 	return self
	# def typer_fast_log10f(self, context):
	# 	return self
	# def typer_fast_log2f(self, context):
	# 	return self

	# # def typer_fast_sincosf(self, context):
	# # 	return self


class DoubleVectorType(FloatingVectorType):
	_member_t = types.float64
	def __init__(self, name = 'double_vector'):
		super().__init__(name = name)

	# def typer_fabs(self, context):
	# 	return self

	# def typer_fmin(self, context, *other):
	# 	if not other:
	# 		return self._member_t
	# 	return self

	# def typer_fmax(self, context, *other):
	# 	if not other:
	# 		return self._member_t
	# 	return self


class Vector2Type(VectorType):
	_fields = ['x', 'y']
	def __init__(self,  name = 'vector2'):
		super().__init__(name = name)

class Vector3Type(VectorType):
	_fields = ['x', 'y', 'z']
	def __init__(self, name = 'vector3'):
		super().__init__(name = name)

vector2_t = Vector2Type()
vector3_t = Vector3Type()

class Char2Type(Vector2Type, CharVectorType):
	_impl_t = char2
	def __init__(self):
		super().__init__(name = 'char2')

class Short2Type(Vector2Type, ShortVectorType):
	_impl_t = short2
	def __init__(self):
		super().__init__(name = 'short2')

class Int2Type(Vector2Type, IntVectorType):
	_impl_t = int2
	def __init__(self):
		super().__init__(name = 'int2')

class Long2Type(Vector2Type, LongVectorType):
	_impl_t = long2
	def __init__(self):
		super().__init__(name = 'long2')

class UChar2Type(Vector2Type, UCharVectorType):
	_impl_t = uchar2
	def __init__(self):
		super().__init__(name = 'uchar2')

class UShort2Type(Vector2Type, UShortVectorType):
	_impl_t = ushort2
	def __init__(self):
		super().__init__(name = 'ushort2')

class UInt2Type(Vector2Type, UIntVectorType):
	_impl_t = uint2
	def __init__(self):
		super().__init__(name = 'uint2')

class ULong2Type(Vector2Type, ULongVectorType):
	_impl_t = ulong2
	def __init__(self):
		super().__init__(name = 'ulong2')

class Float2Type(Vector2Type, FloatVectorType):
	_impl_t = float2
	def __init__(self):
		super().__init__(name = 'float2')

class Double2Type(Vector2Type, DoubleVectorType):
	_impl_t = double2
	def __init__(self):
		super().__init__(name = 'double2')

char2_t = Char2Type() #char2 = Vec2(_member_t = types.uint32)
short2_t = Short2Type()
int2_t = Int2Type()
long2_t = Long2Type()
uchar2_t = UChar2Type()
ushort2_t = UShort2Type()
uint2_t = UInt2Type()
ulong2_t = ULong2Type()
float2_t = Float2Type()
double2_t = Double2Type()

class Char3Type(Vector3Type, CharVectorType):
	_impl_t = char3
	_vec2_t = uchar2_t
	def __init__(self):
		super().__init__(name = 'char3')

class Short3Type(Vector3Type, ShortVectorType):
	_impl_t = short3
	_vec2_t = ushort2_t
	def __init__(self):
		super().__init__(name = 'short3')

class Int3Type(Vector3Type, IntVectorType):
	_impl_t = int3
	_vec2_t = uint2_t
	def __init__(self):
		super().__init__(name = 'int3')

class Long3Type(Vector3Type, LongVectorType):
	_impl_t = long3
	_vec2_t = ulong2_t
	def __init__(self):
		super().__init__(name = 'long3')

class UChar3Type(Vector3Type, UCharVectorType):
	_impl_t = uchar3
	_vec2_t = char2_t
	def __init__(self):
		super().__init__(name = 'uchar3')

class UShort3Type(Vector3Type, UShortVectorType):
	_impl_t = ushort3
	_vec2_t = short2_t
	def __init__(self):
		super().__init__(name = 'ushort3')

class UInt3Type(Vector3Type, UIntVectorType):
	_impl_t = uint3
	_vec2_t = int2_t
	def __init__(self):
		super().__init__(name = 'uint3')

class ULong3Type(Vector3Type, ULongVectorType):
	_impl_t = ulong3
	_vec2_t = long2_t
	def __init__(self):
		super().__init__(name = 'ulong3')

class Float3Type(Vector3Type, FloatVectorType):
	_impl_t = float3
	_vec2_t = float2_t
	def __init__(self):
		super().__init__(name = 'float3')

class Double3Type(Vector3Type, DoubleVectorType):
	_impl_t = double3
	_vec2_t = double2_t
	def __init__(self):
		super().__init__(name = 'double3')

char3_t = Char3Type()
short3_t = Short3Type()
int3_t = Int3Type()
long3_t = Long3Type() #longlong3
uchar3_t = UChar3Type()
ushort3_t = UShort3Type()
uint3_t = UInt3Type()
ulong3_t = ULong3Type() #ulonglong3
float3_t = Float3Type()
double3_t = Double3Type()
