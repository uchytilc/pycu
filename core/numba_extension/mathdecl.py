from numba import types
from numba.cuda.cudadecl import (registry as cuda_registry)
from numba.core.typing.templates import CallableTemplate

# import pycu
# from pycu import core
from . import mathfuncs
from .cudadecl import register_op

#this defines the return types of all declared functions
#all core functions are redefined as CallableTemplates so that external types can define new lowering behaviour
#also extends functionallity of some existing functions allowing them to be called on additional input types

#TO DO
    #Add vectors to this list so they can be called using either
        #pycu.vec2
        #vec


rnd_extns = ["_rd", "_rn", "_ru", "_rz"]


def determine_floating_type(typ, context):
    rtrn = types.float32
    conversion = context.can_convert(typ, rtrn)
    #note: conversion == 4: 4 is enum for unsafe type conversion (i.e. loss of precision)
    if conversion == 4:
        rtrn = types.float64
        conversion = context.can_convert(typ, rtrn)
        if conversion == 4:
            raise TypeError(f"The input type {typ} results in an unsafe type cast thus it is not a supported input type for the function {op}")
    return rtrn

def unary_external(typ, op, context):
    try:
        return getattr(typ, "typer_" + op)(context)
    except AttributeError as e:
        raise TypeError(f"{e}\n{typ} does not support this operation")

#CallableTemplate for functions that return the same type as the input type
class UnaryInputType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        def typer(typ):
            if not typ.is_internal:
                return unary_external(typ, op, self.context)
            return typ
        return typer

def unary_op_input_type_factory(op):
    @cuda_registry.register
    class Cuda_op(UnaryInputType):
        key = getattr(mathfuncs, op)
    return Cuda_op

#CallableTemplate for functions that return either a float32 or a float64 based on arg input
class UnaryFloatingType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        def typer(typ):
            if not typ.is_internal:
                return unary_external(typ, op, self.context)
            return determine_floating_type(typ, self.context)
        return typer

def unary_op_floating_type_factory(op):
    @cuda_registry.register
    class Cuda_op(UnaryFloatingType):
        key = getattr(mathfuncs, op)
    return Cuda_op

#CallableTemplate for functions that return a fixed output regardless of arg input
class UnaryFixedType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        def typer(typ):
            if not typ.is_internal:
                return unary_external(typ, op, self.context)
            return rtrn
        return typer

def unary_op_fixed_type_factory(op, rtrn):
    @cuda_registry.register
    class Cuda_op(UnaryFixedType):
        key = getattr(mathfuncs, op)
    return Cuda_op

# register_op("abs", unary_op_input_type_factory)
register_op("llabs", unary_op_fixed_type_factory, types.int64)
register_op("fabsf", unary_op_fixed_type_factory, types.float32)
register_op("fabs", unary_op_fixed_type_factory, types.float64)

unary_ops = ["sin", "cos", "tan",
             "asin", "acos", "atan",
             "sinh", "cosh", "tanh",
             "asinh", "acosh", "atanh",
             "sinpi", "cospi",
             "sqrt", "rsqrt", "cbrt", "rcbrt",
             "lgamma", "tgamma",
             "j0", "j1", "y0", "y1",
             "exp" ,"exp10", "exp2", "expm1",
             "log", "log1p", "log10", "log2", "logb",
             "erf", "erfc", "erfcinv", "erfcx", "erfinv",
             "normcdf", "normcdfinv",
             "ceil", "floor", "trunc",
             "round", "rint", "nearbyint",
             "modf", "frexp"]
for op in unary_ops:
    opf = op + 'f'
    register_op(op, unary_op_floating_type_factory)
    register_op(opf, unary_op_fixed_type_factory, types.float32)

unary_ops = ['sqrt']
for op in unary_ops:
    for rnd in rnd_extns:
        fop = 'f' + op + rnd
        dop = 'd' + op + rnd
        register_op(fop, unary_op_fixed_type_factory, types.float32)
        register_op(dop, unary_op_fixed_type_factory, types.float64)

unary_ops = ["fast_sinf", "fast_cosf", "fast_tanf",
             "fast_expf", "fast_exp10f",
             "fast_logf", "fast_log10f", "fast_log2f",
             "saturatef", "frsqrt_rn"]
for opf in unary_ops:
    register_op(opf, unary_op_fixed_type_factory, types.float32)

unary_ops = ["llround", "llrint"]
for op in unary_ops:
    opf = op + 'f'
    register_op(op, unary_op_fixed_type_factory, types.int64)
    register_op(opf, unary_op_fixed_type_factory, types.int64)

unary_ops = ["ilogb"]
for op in unary_ops:
    opf = op + 'f'
    register_op(op, unary_op_fixed_type_factory, types.int32)
    register_op(opf, unary_op_fixed_type_factory, types.int32)

unary_ops = ["signbit","isnan","isinf"]
for op in unary_ops:
    opf = op + 'f'
    opd = op + 'd'
    register_op(opf, unary_op_fixed_type_factory, types.int32)
    register_op(opd, unary_op_fixed_type_factory, types.int32)

unary_ops = ["finite"]
for op in unary_ops:
    opf = op + 'f'
    opd = "is" + op + 'd'
    register_op(opd, unary_op_fixed_type_factory, types.int32)
    register_op(opf, unary_op_fixed_type_factory, types.int32)

unary_ops = ["sincos", "sincospi"]
for op in unary_ops:
    opf = op + 'f'
    register_op(op, unary_op_fixed_type_factory, types.void)
    register_op(opf, unary_op_fixed_type_factory, types.void)

unary_ops = ["fast_sincosf"]
for op in unary_ops:
    register_op(op, unary_op_fixed_type_factory, types.void)

unary_ops = ["brev","clz","popc","ffs"]
for op in unary_ops:
    opll = op + 'll'
    register_op(op, unary_op_fixed_type_factory, types.int32)
    register_op(opll, unary_op_fixed_type_factory, types.int64)

unary_groups = [["double2float","int2float","ll2float","uint2float","ull2float"],
                ["ll2double","ull2double"],
                ["double2int","float2int"],
                ["float2ll","double2ll"],
                ["float2uint","double2uint"],
                ["float2ull","double2ull"]]
typs = [types.float32, types.float64, types.int32, types.int64, types.uint32, types.uint64]
for unary_ops, typ in zip(unary_groups, typs):
    for op in unary_ops:
        for rnd in rnd_extns:
            op_rnd = op + rnd
            register_op(op_rnd, unary_op_fixed_type_factory, typ)

#TO DO
    # float2half_rn
    # half2float
    # hiloint2double
    # int2double_rn
    # uint2double_rn



def binary_external(typ_a, typ_b, op, context):
    try:
        return getattr(typ_a, "typer_" + op)(context, typ_b)
    except AttributeError:
        try:
            return getattr(typ_b, "typer_" + op)(context, typ_a)
        except AttributeError as e:
            raise TypeError(f"{e}\n{typ_a} and/or {typ_b} do not support this operation")

#CallableTemplate for functions that return either a float32 or a float64 based on arg input
class BinaryFloatingType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        def typer(typ_a, typ_b):
            if not typ_a.is_internal or not typ_b.is_internal:
                return binary_external(typ_a, typ_b, op, self.context)
            if ignore == 0:
                typ = typ_b
            elif ignore == 1:
                typ = typ_a
            else:
                typ = self.context.cunify_types(typ_a, typ_b)
            return determine_floating_type(typ, self.context)
        return typer

def binary_op_floating_type_factory(op, ignore = None):
    @cuda_registry.register
    class Cuda_op(BinaryFloatingType):
        key = getattr(mathfuncs, op)

    return Cuda_op

#CallableTemplate for functions that return a fixed output regardless of arg input
class BinaryFixedType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        rtrn = self.rtrn
        def typer(typ_a, typ_b):
            if not typ_a.is_internal or not typ_b.is_internal:
                return binary_external(typ_a, typ_b, op, self.context)
            return rtrn
        return typer

def binary_op_fixed_type_factory(op, _rtrn):
    @cuda_registry.register
    class Cuda_op(BinaryFixedType):
        key = getattr(mathfuncs, op)
        rtrn = _rtrn
    return Cuda_op

binary_ops = ["atan2", "fmod", "copysign", "remainder", "remquo",
              "fdim", "hypot", "nextafter", "pow"]
for op in binary_ops:
    opf = op + 'f'
    register_op(op, binary_op_floating_type_factory)
    register_op(opf, binary_op_fixed_type_factory, types.float32)

binary_ops = ["add", "div", "mul", "sqrt", "rcp"]
for op in binary_ops:
    for rnd in rnd_extns:
        fop = 'f' + op + rnd
        dop = 'd' + op + rnd
        register_op(fop, binary_op_fixed_type_factory, types.float32)
        register_op(dop, binary_op_fixed_type_factory, types.float64)

binary_ops = ["fsub"]
for op in binary_ops:
    for rnd in rnd_extns:
        fop = op + rnd
        register_op(fop, binary_op_fixed_type_factory, types.float32)

binary_ops = ["fast_powf", "fast_fdividef"]
for opf in binary_ops:
    register_op(opf, binary_op_fixed_type_factory, types.float32)

binary_ops = ["mulhi", "mul24", "hadd",
              "rhadd", "uhadd", "umul24",
              "umulhi", "urhadd"]
for op in binary_ops:
    register_op(op, binary_op_fixed_type_factory, types.int32)

binary_ops = ["mul64hi", "umul64hi"]
for op in binary_ops:
    register_op(op, binary_op_fixed_type_factory, types.int64)

binary_ops = ["yn", "jn"]
for op in binary_ops:
    opf = op + 'f'
    register_op(op, binary_op_floating_type_factory, 0) #ignore = 0
    register_op(op, binary_op_fixed_type_factory, types.float32)

binary_ops = ["scalbn", "ldexp", "powi"]
for op in binary_ops:
    opf = op + 'f'
    register_op(op, binary_op_floating_type_factory, 1) #ignore = 1
    register_op(opf, binary_op_fixed_type_factory, types.float32)





#unary/binary functions are used for functions like min and max
    #allows for max to be applied to a struct by itself
        #ex. max(float3)
    #as well as multiple structs
        #ex. max(float3, float3)

def unarybinary_external_typa(typ_a, typ_b, op, context):
    if typ_b is None:
        try:
            return getattr(typ_a, "typer_" + op)(context)
        except AttributeError as e:
            raise TypeError(f"{e}\n{typ_a} does not support this operation")
    else:
        try:
            return getattr(typ_a, "typer_" + op)(context, typ_b)
        except AttributeError as e:
            raise TypeError(f"{e}\n{typ_a} and/or {typ_b} do not support this operation")

def unarybinary_external_typb(typ_a, typ_b, op, context):
    try:
        return getattr(typ_b, "typer_" + op)(typ_a, context)
    except AttributeError as e:
        raise TypeError(f"{e}\n{typ_a} and/or {typ_b} do not support this operation")

class UnaryBinaryFloatingType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        def typer(typ_a, typ_b = None):
            if not typ_a.is_internal:
                return unarybinary_external_typa(typ_a, typ_b, op, self.context)
            if typ_b is not None:
                if not typ_b.is_internal:
                    return unarybinary_external_typb(typ_a, typ_b, op, self.context)
                return self.context.cunify_types(typ_a, typ_b)
            return typ_a
        return typer

def unarybinary_op_floating_type_factory(op):
    @cuda_registry.register
    class Cuda_op(UnaryBinaryFloatingType):
        key = getattr(mathfuncs, op)
    return Cuda_op

class UnaryBinaryFixedType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        def typer(typ_a, typ_b = None):
            if not typ_a.is_internal:
                return unarybinary_external_typa(typ_a, typ_b, op, self.context)
            if typ_b is not None and not typ_b.is_internal:
                return unarybinary_external_typb(typ_a, typ_b, op, self.context)
            return rtrn
        return typer

def unarybinary_op_fixed_type_factory(op, rtrn):
    @cuda_registry.register
    class Cuda_op(UnaryBinaryFixedType):
        key = getattr(mathfuncs, op)
    return Cuda_op

register_op("max", unarybinary_op_floating_type_factory) #types.int32
register_op("llmax", unarybinary_op_fixed_type_factory, types.int64)
register_op("umax", unarybinary_op_fixed_type_factory, types.uint32)
register_op("ullmax", unarybinary_op_fixed_type_factory, types.uint64)
register_op("fmaxf", unarybinary_op_fixed_type_factory, types.float32)
register_op("fmax", unarybinary_op_fixed_type_factory, types.float64)

register_op("min", unarybinary_op_floating_type_factory) #types.int32
register_op("llmin", unarybinary_op_fixed_type_factory, types.int64)
register_op("umin", unarybinary_op_fixed_type_factory, types.uint32)
register_op("ullmin", unarybinary_op_fixed_type_factory, types.uint64)
register_op("fminf", unarybinary_op_fixed_type_factory, types.float32)
register_op("fmin", unarybinary_op_fixed_type_factory, types.float64)



def ternary_external(typ_a, typ_b, typ_c, context):
    try:
        return getattr(typ_a, "typer_" + op)(context, typ_b, typ_c)
    except AttributeError:
        try:
            return getattr(typ_b, "typer_" + op)(context, typ_a, typ_c)
        except AttributeError:
            try:
                return getattr(typ_c, "typer_" + op)(context, typ_a, typ_b)
            except AttributeError as e:
                raise TypeError(f"{e}\n{typ_a}, {typ_b}, and/or {typ_c} do not support this operation")

#CallableTemplate for functions that return either a float32 or a float64 based on arg input
class TernaryFloatingType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        def typer(typ_a, typ_b):
            if not typ_a.is_internal or not typ_b.is_internal or not typ_c.is_internal:
                return ternary_external(typ_a, typ_b, typ_c, op, self.context)
            typ = self.context.cunify_types(self.context.cunify_types(typ_a, typ_b), typ_c)
            return determine_floating_type(typ, self.context)
        return typer

def ternary_op_floating_type_factory(op):
    @cuda_registry.register
    class Cuda_op(TernaryFloatingType):
        key = getattr(mathfuncs, op)
    return Cuda_op

#CallableTemplate for functions that return a fixed output regardless of arg input
class TernaryFixedType(CallableTemplate):
    def generic(self):
        op = self.key.__name__
        def typer(typ_a, typ_b, typ_c):
            if not typ_a.is_internal or not typ_b.is_internal or not typ_c.is_internal:
                return ternary_external(typ_a, typ_b, typ_c, op, self.context)
            return rtrn
        return typer

def ternary_op_fixed_type_factory(op, rtrn):
    @cuda_registry.register
    class Cuda_op(TernaryFixedType):
        key = getattr(mathfuncs, op)
    return Cuda_op

ternary_ops = ["fma"]
for op in ternary_ops:
    opf = op + 'f'
    register_op(opf, ternary_op_fixed_type_factory, types.float32)
    register_op(op, ternary_op_fixed_type_factory, types.float64) #ternary_op_floating_type_factory

for op in ternary_ops:
    for rnd in rnd_extns:
        opf = op + 'f' + rnd
        opd = op + rnd
        register_op(opf, ternary_op_fixed_type_factory, types.float32)
        register_op(opd, ternary_op_fixed_type_factory, types.float64)

register_op("sad", ternary_op_fixed_type_factory, types.int32)
register_op("usad", ternary_op_fixed_type_factory, types.uint32)
register_op("byte_perm", ternary_op_fixed_type_factory, types.int32)

