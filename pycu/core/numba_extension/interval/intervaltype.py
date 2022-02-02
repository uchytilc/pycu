from ..vector.vectortype import vec2, vec3

import numpy as np
import math
# import mpmath as mp

#note: rounding floor/ceil are towards -/+ inf, up/down change directions if the number is - vs +

# def fadd_rd(x,y):
# 	return mp.fadd(x,y,rounding='f')

# def fadd_ru(x,y):
# 	return mp.fadd(x,y,rounding='c')

# def fsub_rd(x,y):
# 	return mp.fsub(x,y,rounding='f')

# def fsub_ru(x,y):
# 	return mp.fsub(x,y,rounding='c')

# def fmul_rd(x,y):
# 	return mp.fmul(x,y,rounding='f')

# def fmul_ru(x,y):
# 	return mp.fmul(x,y,rounding='c')

# def fdiv_rd(x,y):
# 	#if y == 0
# 		#return np.inf
# 	return mp.fdiv(x,y,rounding='f')

# def fdiv_ru(x,y):
# 	#if y == 0
# 		#return np.inf
# 	return mp.fdiv(x,y,rounding='c')

# def fsqrt_rd(x):
# 	return mp.sqrt(x,rounding='f')

# def fsqrt_ru(x):
# 	return mp.sqrt(x,rounding='c')

class _interval:
	__members__ = ['lo', 'hi']
	def __init__(self, lo, hi):
		# self._lo = 0
		# self._hi = 0

		self.lo = lo
		self.hi = hi

	def __array__(self):
		ary = np.empty(len(self.__members__), dtype = self.__member_type__)
		for n, member in enumerate(self.__members__):
			ary[n] = getattr(self, member)
		return ary

	# @property
	# def lo(self):
	# 	return self._lo

	# @lo.setter
	# def lo(self, lo):
	# 	self._lo = self.dtype(lo)

	# @property
	# def hi(self):
	# 	return self._hi

	# @hi.setter
	# def hi(self, hi):
	# 	self._hi = self.dtype(hi)

	@property
	def width(self):
		return fsub_ru(self.hi, self.lo)

	def __repr__(self):
		return f"interval({self.lo}, {self.hi})"

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.lo == other.lo and self.hi == other.hi
		else:
			return self.lo == other and self.hi == other

	def __ne__(self, other):
		return (not self.__eq__(other))

	def __gt__(self, other):
		if isinstance(other, self.__class__):
			return self.hi > other.lo
		else:
			return self.hi > other

	def __ge__(self, other):
		if isinstance(other, self.__class__):
			return self.hi >= other.lo
		else:
			return self.hi >= other

	def __lt__(self, other):
		if isinstance(other, self.__class__):
			return self.hi < other.lo
		else:
			return self.hi < other

	def __le__(self, other):
		if isinstance(other, self.__class__):
			return self.hi <= other.lo
		else:
			return self.hi <= other

	def __contains__(self, other):
		if isinstance(other, self.__class__):
			return self.lo <= other.lo and self.hi >= other.hi
		else:
			return self.lo <= other <= self.hi

	def __neg__(self):
		return interval(-self.hi, -self.lo)

	def __add__(self, other):
		if isinstance(other, self.__class__):
			lo = fadd_rd(self.lo, other.lo)
			hi = fadd_ru(self.hi, other.hi)
			return interval(lo, hi)
		else:
			lo = fadd_rd(self.lo, other)
			hi = fadd_ru(self.hi, other)
			return interval(lo, hi)

	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		if isinstance(other, self.__class__):
			lo = fsub_rd(self.lo, other.hi)
			hi = fsub_ru(self.hi, other.lo)
			return interval(lo, hi)
		else:
			lo = fsub_rd(self.lo, other)
			hi = fsub_ru(self.hi, other)
			return interval(lo, hi)

	def __rsub__(self, other):
		return -self + other

	def __mul__(self, other):
		if isinstance(other, self.__class__):
			#naive version avoids thread divergence so it is preferred
			lo = pymin([fmul_rd(self.lo, other.lo), fmul_rd(self.lo, other.hi), fmul_rd(self.hi, other.lo), fmul_rd(self.hi, other.hi)])
			hi = pymax([fmul_ru(self.lo, other.lo), fmul_ru(self.lo, other.hi), fmul_ru(self.hi, other.lo), fmul_ru(self.hi, other.hi)])
			return interval(lo, hi)
		else:
			lo = fmul_rd(other, self.lo)
			hi = fmul_ru(other, self.hi)
			return interval(pymin(lo, hi), pymax(lo, hi))

	def __rmul__(self, other):
		return self * other

	def __truediv__(self, other):
		pass
		# if isinstance(other, self.__class__):
		# 	if other.lo <= 0. and 0 <= other.hi:
		# 		return interval(-np.inf, np.inf)
		# 	elif self.hi < 0.:
		# 		if other.hi < 0.:
		# 			return interval(fdiv_rd(self.hi, other.lo), fdiv_ru(self.lo, other.hi))
		# 		else:
		# 			return interval(fdiv_rd(self.lo, other.lo), fdiv_ru(self.hi, other.hi))
		# 	elif self.hi < 0.:
		# 		if other.hi < 0.:
		# 			return interval(fdiv_rd(self.hi, other.hi), fdiv_ru(self.lo, other.hi))
		# 		else:
		# 			return interval(fdiv_rd(self.lo, other.lo), fdiv_ru(self.hi, other.lo))
		# 	else:
		# 		if other.hi < 0.:
		# 			return interval(fdiv_rd(self.hi, other.hi), fdiv_ru(self.lo, other.lo))
		# 		else:
		# 			return interval(fdiv_rd(self.lo, other.hi), fdiv_ru(self.hi, other.lo))
		# else:
		# 	try:
		# 		other = 1/other
		# 	except ZeroDivisionError:
		# 		other = np.inf
		# 	return self*other
		# 	# if other < 0.:
		# 	# 	return interval(fdiv_rd(self.hi, other), fdiv_ru(self.lo, other))
		# 	# elif other > 0.:
		# 	# 	return interval(fdiv_rd(self.lo, other), fdiv_ru(self.hi, other))
		# 	# else:
		# 	# 	return interval(-np.inf, np.inf)

		# # if isinstance(other, self.__class__):
		# # 	if other.lo <= 0. and 0 <= other.hi.:
		# # 		return interval(-np.inf, np.inf)
		# # 	elif self.hi < 0.:
		# # 		if other.hi < 0.:
		# # 			return interval(fdiv_rd(self.hi, other.lo), fdiv_ru(self.lo, other.hi))
		# # 		else:
		# # 			return interval(fdiv_rd(self.lo, other.lo), fdiv_ru(self.hi, other.hi))
		# # 	elif self.hi < 0.:
		# # 		if other.hi < 0.:
		# # 			return interval(fdiv_rd(self.hi, other.hi), fdiv_ru(self.lo, other.hi))
		# # 		else:
		# # 			return interval(fdiv_rd(self.lo, other.lo), fdiv_ru(self.hi, other.lo))
		# # 	else:
		# # 		if other.hi < 0.:
		# # 			return interval(fdiv_rd(self.hi, other.hi), fdiv_ru(self.lo, other.lo))
		# # 		else:
		# # 			return interval(fdiv_rd(self.lo, other.hi), fdiv_ru(self.hi, other.lo))
		# # else:
		# # 	if other < 0.:
		# # 		return interval(fdiv_rd(self.hi, other), fdiv_ru(self.lo, other))
		# # 	elif other > 0.:
		# # 		return interval(fdiv_rd(self.lo, other), fdiv_ru(self.hi, other))
		# # 	else:
		# # 		return interval(-np.inf, np.inf)

	def __rtruediv__(self, other):
		return interval(other, other)/self

	def __pow__(self, n):
		pass
		# if isinstance(n, int):
		# 	if n < 0:
		# 		x = 1 / self
		# 		n = -n
		# 	if n == 0:
		# 		return 1
		# 	y = 1
		# 	x = interval(self.lo, self.hi)
		# 	while n > 1:
		# 		if n % 2 == 0:
		# 			x = x * x
		# 			n = n / 2
		# 		else:
		# 			y = x * y
		# 			x = x * x
		# 			n = (n - 1) / 2
		# 	return x * y
		# # elif isinstance(n, self.__class__):
		# # 	return (n * self.log()).exp()

	def __abs__(self):
		if self.lo >= 0.:
			return self
		elif self.hi < 0.:
			return -self
		else:
			return interval(0., max(abs(self.lo), abs(self.hi)))


class intervalf(_interval):
	__member_type__ = np.float32

interval = intervalf

class intervald(_interval):
	__member_type__ = np.float64






class intervalf2(vec2):
	__member_type__ = intervalf
	def __init__(self, x, y):
		super().__init__(x,y)

	# def __array__(self):
	# 	ary = np.empty(len(self.__members__), dtype = self.__member_type__)
	# 	for n, member in enumerate(self.__members__):
	# 		ary[n] = getattr(self, member)
	# 	return ary

class intervalf3(vec3):
	__member_type__ = intervalf
	__vec2__ = intervalf2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)

	# def __array__(self):
	# 	ary = np.empty(len(self.__members__), dtype = self.__member_type__)
	# 	for n, member in enumerate(self.__members__):
	# 		ary[n] = getattr(self, member)
	# 	return ary


interval2 = intervalf2
interval3 = intervalf3

class intervald2(vec2):
	__member_type__ = intervald
	def __init__(self, x, y):
		super().__init__(x,y)

class intervald3(vec3):
	__member_type__ = intervald
	__vec2__ = intervald2
	def __init__(self, x, y, z):
		super().__init__(x,y,z)
