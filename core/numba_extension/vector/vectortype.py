import numpy as np

#get get these working with numpy functions
	#https://stackoverflow.com/questions/43493256/python-how-to-implement-a-custom-class-compatible-with-numpy-functions

class vec:
	def __init__(self):
		pass

	@property
	def _fields_(self):
		return [(__memeber_name__, self.__member_type__) for __member_type__ in self.__member_names__]

	# def __repr__(self):
	# 	return f"{self.__class__.__name__}({', '.join([str(getattr(self, member)) for member in self.__member_names__])})"

	def __getitem__(self, idx):
		return getattr(self, self.__member_names__[idx])

	def __setitem__(self, idx, value):
		return setattr(self, self.__member_names__[idx], value)

	def __iter__(self):
		return iter([getattr(self, member) for member in self.__member_names__])

	def __array__(self, dtype):
		ary = np.empty(len(self.__member_names__), dtype = dtype)
		for n, member in enumerate(self.__member_names__):
			ary[n] = getattr(self, member)
		return ary

	# def __array_wrap__(self):
		# pass

class vec2(vec):
	__member_names__ = ['x', 'y']
	def __init__(self, x = 0., y = 0.):
		super().__init__()
		# self._x = None
		# self._y = None

		self.x = x
		self.y = y

	# @property
	# def x(self):
	# 	return self._x

	# @x.setter
	# def x(self, x):
	# 	if type(x) != self.__member_type__:
	# 		self._x = self.__member_type__(x)
	# 	else:
	# 		self._x = x

	# @property
	# def y(self):
	# 	return self._y

	# @y.setter
	# def y(self, y):
	# 	if type(y) != self.__member_type__:
	# 		self._y = self.__member_type__(y)
	# 	else:
	# 		self._y = y

	def __add__(self, other):
		vec = self.__class__
		return vec(self.x + other.x,
					self.y + other.y)

	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		vec = self.__class__
		return vec(self.x - other.x,
					self.y - other.y)

	def __rsub__(self, other):
		return -self + other

	def __mul__(self, other):
		vec = self.__class__
		return vec(self.x*other.x,
					self.y*other.y)

	def __rmul__(self, other):
		return self*other

	def __truediv__(self, other):
		vec = self.__class__
		return vec(self.x/other.x,
					self.y/other.y)

	def __abs__(self):
		vec = self.__class__
		return vec(abs(self.x),
					abs(self.y))

class vec3(vec):
	__member_names__ = ['x', 'y', 'z']
	def __init__(self, x = 0., y = 0., z = 0.):
		super().__init__()
		# self._x = None
		# self._y = None
		# self._z = None

		self.x = x
		self.y = y
		self.z = z

	# @property
	# def x(self):
	# 	return self._x
	
	# @x.setter
	# def x(self, x):
	# 	if type(x) != self.__member_type__:
	# 		self._x = self.__member_type__(x)
	# 	else:
	# 		self._x = x

	# @property
	# def y(self):
	# 	return self._y

	# @y.setter
	# def y(self, y):
	# 	if type(y) != self.__member_type__:
	# 		self._y = self.__member_type__(y)
	# 	else:
	# 		self._y = y

	# @property
	# def z(self):
	# 	return self._z

	# @z.setter
	# def z(self, z):
	# 	if type(z) != self.__member_type__:
	# 		self._z = self.__member_type__(z)
	# 	else:
	# 		self._z = z

	@property
	def xy(self):
		return self.__vec2__(self.x, self.y)

	@property
	def xz(self):
		return self.__vec2__(self.x, self.z)

	@property
	def yz(self):
		return self.__vec2__(self.y, self.z)

	def __add__(self, other):
		vec = self.__class__
		if isinstance(other, vec3):
			return vec(self.x + other.x,
						self.y + other.y,
						self.z + other.z)
		elif isinstance(other, vec2):
			return vec(self.x + other.x,
						self.y + other.y,
						self.z)

	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		vec = self.__class__
		if isinstance(other, vec3):
			return vec(self.x - other.x,
						self.y - other.y,
						self.z - other.z)
		if isinstance(other, vec2):
			return vec(self.x - other.x,
						self.y - other.y,
						self.z)

	def __rsub__(self, other):
		return -self + other

	def __mul__(self, other):
		vec = self.__class__
		if isinstance(other, vec3):
			return vec(self.x*other.x,
						self.y*other.y,
						self.z*other.z)
		if isinstance(other, vec2):
			return vec(self.x*other.x,
						self.y*other.y,
						self.z)

	def __rmul__(self, other):
		return self*other

	def __truediv__(self, other):
		vec = self.__class__
		if isinstance(other, vec3):
			return vec(self.x/other.x,
						self.y/other.y,
						self.z/other.z)
		elif isinstance(other, vec2):
			return vec(self.x/other.x,
						self.y/other.y,
						self.z)

	def __abs__(self):
		vec = self.__class__
		return vec(abs(self.x),
					abs(self.y),
					abs(self.z))

class char2(vec2):
	__member_type__ = np.int8
	def __init__(self, x, y):
		super().__init__(x,y)

class short2(vec2):
	__member_type__ = np.int16
	def __init__(self, x, y):
		super().__init__(x,y)

class int2(vec2):
	__member_type__ = np.int32
	def __init__(self, x, y):
		super().__init__(x,y)

class long2(vec2):
	__member_type__ = np.int64
	def __init__(self, x, y):
		super().__init__(x,y)

class uchar2(vec2):
	__member_type__ = np.uint8
	def __init__(self, x, y):
		super().__init__(x,y)

class ushort2(vec2):
	__member_type__ = np.uint16
	def __init__(self, x, y):
		super().__init__(x,y)

class uint2(vec2):
	__member_type__ = np.uint32
	def __init__(self, x, y):
		super().__init__(x,y)

class ulong2(vec2):
	__member_type__ = np.uint64
	def __init__(self, x, y):
		super().__init__(x,y)

class float2(vec2):
	__member_type__ = np.float32
	def __init__(self, x, y):
		super().__init__(x,y)

class double2(vec2):
	__member_type__ = np.float64
	def __init__(self, x, y):
		super().__init__(x,y)








class char3(vec3):
	__member_type__ = np.int8
	__vec2__ = char2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class short3(vec3):
	__member_type__ = np.int16
	__vec2__ = short2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class int3(vec3):
	__member_type__ = np.int32
	__vec2__ = int2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class long3(vec3):
	__member_type__ = np.int64
	__vec2__ = long2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class uchar3(vec3):
	__member_type__ = np.uint8
	__vec2__ = uchar2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class ushort3(vec3):
	__member_type__ = np.uint16
	__vec2__ = ushort2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class uint3(vec3):
	__member_type__ = np.uint32
	__vec2__ = uint2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class ulong3(vec3):
	__member_type__ = np.uint64
	__vec2__ = ulong2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class float3(vec3):
	__member_type__ = np.float32
	__vec2__ = float2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

class double3(vec3):
	__member_type__ = np.float64
	__vec2__ = double2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

# x = float3(1,2,3)
# print(x)
# y = float3(*x)
# print(y)
