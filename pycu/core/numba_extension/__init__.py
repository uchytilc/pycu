from .mathfuncs import *
from . import mathimpl

from .vector.vectortype import *
from .vector import vectorimpl

from .interval.intervaltype import *
from .interval import intervalimpl

from . import vecimpl

from ...driver.core.types import *

import ctypes
from numba.cuda.cudadrv.devicearray import DeviceNDArray

preparer.add_mapping(DeviceNDArray, CuNDArrayType)

class VectorType(Type):
	def __init__(self, dtype):
		self.dtype = dtype

class Vector2Type(VectorType):
	members = ['x', 'y']
	_cache = {}

	def as_ctypes(self):
		dtype = self.dtype.as_ctypes()

		key = (dtype,)
		c_Vector2 = self._cache.get(key, None)
		if c_Vector2 is None:
			class c_Vector2(ctypes.Structure):
				_fields_ = [(member, dtype) for member in self.members]

				def __init__(self, vector = []):
					super().__init__(*vector)

			self._cache[key] = c_Vector2
		return c_Vector2

class Vector3Type(VectorType):
	members = ['x', 'y','z']
	_cache = {}

	def as_ctypes(self):
		dtype = self.dtype.as_ctypes()

		key = (dtype,)
		c_Vector3 = self._cache.get(key, None)
		if c_Vector3 is None:
			class c_Vector3(ctypes.Structure):
				_fields_ = [(member, dtype) for member in self.members]

				def __init__(self, vector = []):
					super().__init__(*vector)

			self._cache[key] = c_Vector3
		return c_Vector3

class Vector4Type(VectorType):
	members = ['x', 'y','z','w']
	_cache = {}

	def as_ctypes(self):
		dtype = self.dtype.as_ctypes()

		key = (dtype,)
		c_Vector4 = self._cache.get(key, None)
		if c_Vector4 is None:
			class c_Vector4(ctypes.Structure):
				_fields_ = [(member, dtype) for member in self.members]

				def __init__(self, vector = []):
					super().__init__(*vector)

			self._cache[key] = c_Vector4
		return c_Vector4

class IntervalType(Type):
	members = ['lo', 'hi']
	_cache = {}
	def __init__(self, dtype):
		self.dtype = dtype

	def as_ctypes(self):
		dtype = self.dtype.as_ctypes()

		key = (dtype,)
		c_Interval = self._cache.get(key, None)
		if c_Interval is None:
			class c_Interval(ctypes.Structure):
				_fields_ = [(member, dtype) for member in self.members]
			self._cache[key] = c_Interval
		return c_Interval

char2_t = Vector2Type(int8_t)
short2_t = Vector2Type(int16_t)
int2_t = Vector2Type(int32_t)
long2_t = Vector2Type(int64_t)
uchar2_t = Vector2Type(uint8_t)
ushort2_t = Vector2Type(uint16_t)
uint2_t = Vector2Type(uint32_t)
ulong2_t = Vector2Type(uint64_t)
float2_t = Vector2Type(float32_t)
double2_t = Vector2Type(float64_t)

char3_t = Vector3Type(int8_t)
short3_t = Vector3Type(int16_t)
int3_t = Vector3Type(int32_t)
long3_t = Vector3Type(int64_t)
uchar3_t = Vector3Type(uint8_t)
ushort3_t = Vector3Type(uint16_t)
uint3_t = Vector3Type(uint32_t)
ulong3_t = Vector3Type(uint64_t)
float3_t = Vector3Type(float32_t)
double3_t = Vector3Type(float64_t)

intervalf_t = IntervalType(float32_t)
intervald_t = IntervalType(float64_t)

intervalf2_t = Vector2Type(intervalf_t)
intervald2_t = Vector2Type(intervald_t)
intervalf3_t = Vector3Type(intervalf_t)
intervald3_t = Vector3Type(intervald_t)

for vec, vec_t in zip([char2,   short2,   int2,   long2,   uchar2,   ushort2,   uint2,   ulong2,   float2,   double2],
					  [char2_t, short2_t, int2_t, long2_t, uchar2_t, ushort2_t, uint2_t, ulong2_t, float2_t, double2_t]):
	preparer.add_mapping(vec, vec_t)

for vec, vec_t in zip([char3,   short3,   int3,   long3,   uchar3,   ushort3,   uint3,   ulong3,   float3,   double3],
					  [char3_t, short3_t, int3_t, long3_t, uchar3_t, ushort3_t, uint3_t, ulong3_t, float3_t, double3_t]):
	preparer.add_mapping(vec, vec_t)

preparer.add_mapping(intervalf, intervalf_t)
preparer.add_mapping(intervald, intervald_t)

preparer.add_mapping(intervalf2, intervalf2_t)
preparer.add_mapping(intervald2, intervald2_t)

preparer.add_mapping(intervalf3, intervalf3_t)
preparer.add_mapping(intervald3, intervald3_t)
