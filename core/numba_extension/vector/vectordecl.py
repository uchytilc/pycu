from numba import types

from .vectortype import (char2, short2, int2, long2,
						uchar2, ushort2, uint2, ulong2,
						float2, double2,
						char3, short3, int3, long3,
						uchar3, ushort3, uint3, ulong3,
						float3, double3)

#To add typer behaviour for a given function and vector type add typer_FUNCTION and define the return type for the function

class VectorType(types.Type):
	def __init__(self, name = 'vec'):
		super().__init__(name = name)

	@property
	def __members__(self):
		return [(name, self.__member_type__)for name in self.__member_names__]

	def typer_sum(self, context):
		return self.__member_type__

	def typer_dot(self, context, other):
		return self

	def typer_min(self, context, *other):
		#if unary max is called a scalar is returned
		if not other:
			return self.__member_type__
		return self

	def typer_max(self, context, *other):
		#if unary min is called a scalar is returned
		if not other:
			return self.__member_type__
		return self

vector_t = VectorType()

class NumberVectorType(VectorType):
	__input_type__ = types.Number
	def __init__(self, name = 'number_vector'):
		super().__init__(name = name)

class IntegerVectorType(NumberVectorType):
	def __init__(self, name = 'integer_vector'):
		super().__init__(name = name)

class CharVectorType(IntegerVectorType):
	__member_type__ = types.int8
	def __init__(self, name = 'char_vector'):
		super().__init__(name = name)

class ShortVectorType(IntegerVectorType):
	__member_type__ = types.int16
	def __init__(self, name = 'short_vector'):
		super().__init__(name = name)

class IntVectorType(IntegerVectorType):
	__member_type__ = types.int32
	def __init__(self, name = 'int_vector'):
		super().__init__(name = name)

	def typer_abs(self, context):
		return self

class LongVectorType(IntegerVectorType):
	__member_type__ = types.int64
	def __init__(self, name = 'long_vector'):
		super().__init__(name = name)

	def typer_llabs(self, context):
		return self

	def typer_llmin(self, context, *other):
		if not other:
			return self.__member_type__
		return self

	def typer_llmax(self, context, *other):
		if not other:
			return self.__member_type__
		return self

class UCharVectorType(IntegerVectorType):
	__member_type__ = types.uint8
	def __init__(self, name = 'uchar_vector'):
		super().__init__(name = name)

class UShortVectorType(IntegerVectorType):
	__member_type__ = types.uint16
	def __init__(self, name = 'ushort_vector'):
		super().__init__(name = name)

class UIntVectorType(IntegerVectorType):
	__member_type__ = types.uint32
	def __init__(self, name = 'uint_vector'):
		super().__init__(name = name)

	def typer_umin(self, context, *other):
		if not other:
			return self.__member_type__
		return self

	def typer_umax(self, context, *other):
		if not other:
			return self.__member_type__
		return self

class ULongVectorType(IntegerVectorType):
	__member_type__ = types.uint64
	def __init__(self, name = 'ulong_vector'):
		super().__init__(name = name)

	def typer_ullmin(self, context, *other):
		if not other:
			return self.__member_type__
		return self

	def typer_ullmax(self, context, *other):
		if not other:
			return self.__member_type__
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
		return self.__member_type__

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


class FloatVectorType(FloatingVectorType):
	__member_type__ = types.float32
	def __init__(self, name = 'float_vector'):
		super().__init__(name = name)

	# def typer_fabsf(self, context):
	# 	return self

	# def typer_fminf(self, context, *other):
	# 	if not other:
	# 		return self.__member_type__
	# 	return self
	# def typer_fmaxf(self, context, *other):
	# 	if not other:
	# 		return self.__member_type__
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
	__member_type__ = types.float64
	def __init__(self, name = 'double_vector'):
		super().__init__(name = name)

	# def typer_fabs(self, context):
	# 	return self

	# def typer_fmin(self, context, *other):
	# 	if not other:
	# 		return self.__member_type__
	# 	return self

	# def typer_fmax(self, context, *other):
	# 	if not other:
	# 		return self.__member_type__
	# 	return self




class Vector2Type(VectorType):
	__member_names__ = ['x', 'y']
	def __init__(self,  name = 'vec2'):
		super().__init__(name = name)

class Vector3Type(VectorType):
	__member_names__ = ['x', 'y', 'z']
	def __init__(self, name = 'vec3'):
		super().__init__(name = name)

vector2_t = Vector2Type()
vector3_t = Vector3Type()

class Char2Type(Vector2Type, CharVectorType):
	__impl_type__ = char2
	def __init__(self):
		super().__init__(name = 'char2')

class Short2Type(Vector2Type, ShortVectorType):
	__impl_type__ = short2
	def __init__(self):
		super().__init__(name = 'short2')

class Int2Type(Vector2Type, IntVectorType):
	__impl_type__ = int2
	def __init__(self):
		super().__init__(name = 'int2')

class Long2Type(Vector2Type, LongVectorType):
	__impl_type__ = long2
	def __init__(self):
		super().__init__(name = 'long2')

class UChar2Type(Vector2Type, UCharVectorType):
	__impl_type__ = uchar2
	def __init__(self):
		super().__init__(name = 'uchar2')

class UShort2Type(Vector2Type, UShortVectorType):
	__impl_type__ = ushort2
	def __init__(self):
		super().__init__(name = 'ushort2')

class UInt2Type(Vector2Type, UIntVectorType):
	__impl_type__ = uint2
	def __init__(self):
		super().__init__(name = 'uint2')

class ULong2Type(Vector2Type, ULongVectorType):
	__impl_type__ = ulong2
	def __init__(self):
		super().__init__(name = 'ulong2')

class Float2Type(Vector2Type, FloatVectorType):
	__impl_type__ = float2
	def __init__(self):
		super().__init__(name = 'float2')

class Double2Type(Vector2Type, DoubleVectorType):
	__impl_type__ = double2
	def __init__(self):
		super().__init__(name = 'double2')


char2_t = Char2Type() #char2 = Vec2(__member_type__ = types.uint32)
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
	__impl_type__ = char3
	__vec2_type__ = uchar2_t
	def __init__(self):
		super().__init__(name = 'char3')

class Short3Type(Vector3Type, ShortVectorType):
	__impl_type__ = short3
	__vec2_type__ = ushort2_t
	def __init__(self):
		super().__init__(name = 'short3')

class Int3Type(Vector3Type, IntVectorType):
	__impl_type__ = int3
	__vec2_type__ = uint2_t
	def __init__(self):
		super().__init__(name = 'int3')

class Long3Type(Vector3Type, LongVectorType):
	__impl_type__ = long3
	__vec2_type__ = ulong2_t
	def __init__(self):
		super().__init__(name = 'long3')

class UChar3Type(Vector3Type, UCharVectorType):
	__impl_type__ = uchar3
	__vec2_type__ = char2_t
	def __init__(self):
		super().__init__(name = 'uchar3')

class UShort3Type(Vector3Type, UShortVectorType):
	__impl_type__ = ushort3
	__vec2_type__ = short2_t
	def __init__(self):
		super().__init__(name = 'ushort3')

class UInt3Type(Vector3Type, UIntVectorType):
	__impl_type__ = uint3
	__vec2_type__ = int2_t
	def __init__(self):
		super().__init__(name = 'uint3')

class ULong3Type(Vector3Type, ULongVectorType):
	__impl_type__ = ulong3
	__vec2_type__ = long2_t
	def __init__(self):
		super().__init__(name = 'ulong3')

class Float3Type(Vector3Type, FloatVectorType):
	__impl_type__ = float3
	__vec2_type__ = float2_t
	def __init__(self):
		super().__init__(name = 'float3')

class Double3Type(Vector3Type, DoubleVectorType):
	__impl_type__ = double3
	__vec2_type__ = double2_t
	def __init__(self):
		super().__init__(name = 'double3')

char3_t = Char3Type()
short3_t = Short3Type()
int3_t = Int3Type()
long3_t = Long3Type()
uchar3_t = UChar3Type()
ushort3_t = UShort3Type()
uint3_t = UInt3Type()
ulong3_t = ULong3Type()
float3_t = Float3Type()
double3_t = Double3Type()
