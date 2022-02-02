from numba import types

class VecType(types.Type):
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

class Vec2Type(VecType):
	_fields = ['x', 'y']
	def __init__(self,  name = 'vec2'):
		super().__init__(name = name)

class Vec3Type(VecType):
	_fields = ['x', 'y', 'z']
	def __init__(self, name = 'vec3'):
		super().__init__(name = name)

vec2_t = Vec2Type()
vec3_t = Vec3Type()