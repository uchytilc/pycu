try:
    from numba import types
    from numba.cuda.cudaimpl import (lower as cuda_lower)
    from numba.core.typing.templates import signature
    from numba.cuda import libdevice as numba_libdevice
    from llvmlite.llvmpy.core import Type
except:
    raise ImportError("Numba is not present")

# import pycu
# from pycu import core
from . import mathdecl, mathfuncs

rnd_extns = ["_rd", "_rn", "_ru", "_rz"]

###########
#UNARY OPS#
###########

#####
#abs
#####

def abs_op_factory(op):
    @cuda_lower(getattr(mathfuncs, op), types.Number)
    def cuda_op(context, builder, sig, args):
        fn = context.get_function(abs, signature(sig.return_type, sig.return_type))
        return fn(builder, [context.cast(builder, args[0], sig.args[0], sig.return_type)])

for op in ['llabs', 'fabsf', 'fabs']: #'abs', 
    abs_op_factory(op)

def unary_op_fixed_factory(op, typ):
    #specialized for when input type only supports fixed types
    @cuda_lower(getattr(mathfuncs, op), typ)
    def cuda_opf(context, builder, sig, args):
        return context.get_function(getattr(numba_libdevice, op), sig)(builder, args)

def unary_op_factory(op, opf):
    #specialized for when input and output type are the same
    @cuda_lower(getattr(mathfuncs, op), types.float64)
    def cuda_op(context, builder, sig, args):
        return context.get_function(getattr(numba_libdevice, op), sig)(builder, args)

    @cuda_lower(getattr(mathfuncs, op), types.float32)
    def cuda_op_float(context, builder, sig, args):
        return context.get_function(getattr(numba_libdevice, opf(op)), sig)(builder, args)

    #integer specializations
    @cuda_lower(getattr(mathfuncs, op), types.Integer)
    def cuda_op_integer(context, builder, sig, args):
        fname = op
        if sig.return_type != types.float64:
            fname = opf(op)
        x = context.cast(builder, args[0], sig.args[0], sig.return_type)
        fn = context.get_function(getattr(numba_libdevice, fname), signature(sig.return_type, sig.return_type))
        return fn(builder, [x])

def unary_opf_factory(op):
    #specialized for when input and output type are the same
    @cuda_lower(getattr(mathfuncs, op), types.float32)
    def cuda_opf(context, builder, sig, args):
        return context.get_function(getattr(numba_libdevice, op), sig)(builder, args)

    def cuda_opf_specialized(context, builder, sig, args):
        fn = context.get_function(getattr(numba_libdevice, op), signature(sig.return_type, sig.return_type))
        return fn(builder, [context.cast(builder, args[0], sig.args[0], sig.return_type)])

    cuda_lower(getattr(mathfuncs, op), types.float64)(cuda_opf)
    cuda_lower(getattr(mathfuncs, op), types.Integer)(cuda_opf)

#unary ops that follow the convention
    #op       = float64
    #op + 'f' = float32

ops = ["sin", "cos", "tan",
       "sinh", "cosh", "tanh",
       "asin", "acos", "atan",
       "asinh", "acosh", "atanh",
       "sinpi", "cospi",
       "sqrt", "rsqrt", "cbrt", "rcbrt",
       "lgamma", "tgamma",
       "j0", "j1", "y0", "y1",
       "exp" ,"exp10", "exp2", "expm1",
       "log", "log1p", "log10", "log2", "logb",
       "erf", "erfc", "erfcinv", "erfcx", "erfinv",
       "normcdf","normcdfinv",
       "ceil", "floor", "trunc",
       "round", "llround",
       "rint", "llrint",
       "ilogb","nearbyint"]

def opf(op):
    #function dictating how the input op name (which takes a double(s)) should be transformed to get the float variant
    return op + 'f'

for op in ops:
    unary_op_factory(op, opf)
    unary_opf_factory(opf(op))


#####################################
# "modf", "frexp"

#second arg is a ptr
    #resolve_modf(self, mod):
    #resolve_modff(self, mod):
#####################################


unary_ops = ["sqrt", "rcp"]
for op in unary_ops:
    for rnd in rnd_extns:
        unary_op_fixed_factory('f' + op + rnd, types.float32)
        unary_op_fixed_factory('d' + op + rnd, types.float64)

#unary ops that are float only

opfs = ["fast_sinf", "fast_cosf", "fast_tanf",
        "fast_expf", "fast_exp10f",
        "fast_logf", "fast_log10f", "fast_log2f",
        "saturatef", "frsqrt_rn"]

for op in opfs:
    unary_opf_factory(op)

#unary ops that follow the convention
    #op + 'd' = float64
    #op + 'f' = float32

ops = ["signbitd","isnand","isinfd"]

def opf(op):
    return op[:-1] + 'f'

for op in ops:
    unary_op_factory(op, opf)
    unary_opf_factory(opf(op))

#note: float variant doesn't have 'is' otherwise it conforms to above pattern
ops = ["isfinited"]

def opf(op):
    return op[2:-1] + 'f'

for op in ops:
    unary_op_factory(op, opf)
    unary_opf_factory(opf(op))

#####################################
# unary_ops = ["sincos", "sincospi"]
# unary_ops = ["fast_sincosf"]

#second and third args are ptrs (cast input to float)
    # resolve_sincos
    # resolve_sincosf
    # resolve_sincospi
    # resolve_sincospif
#####################################

#####################################
# unary_ops = ["brev","clz","popc","ffs"]
#####################################


def generate_cast_op(ops, typ):
    for op in ops:
        for rnd in rnd_extns:
            op_rnd = op + rnd
            unary_op_fixed_factory(op_rnd, typ)

generate_cast_op(["float2int", "float2ll", "float2uint", "float2ull"], types.float32)
generate_cast_op(["double2float", "double2int", "double2ll", "double2uint", "double2ull"], types.float64)
generate_cast_op(["int2float"], types.int32)
generate_cast_op(["ll2float", "ll2double"], types.int64)
generate_cast_op(["uint2float"], types.uint32)
generate_cast_op(["ull2float", "ull2double"], types.uint64)










############
#BINARY OPS#
############

def binary_op_fixed_factory(op, typ):
    #specialized for when input and output type are the same
    @cuda_lower(getattr(mathfuncs, op), typ, typ)
    def cuda_opf(context, builder, sig, args):
        return context.get_function(getattr(numba_libdevice, op), sig)(builder, args)

def binary_op_factory(op, opf):
    #specialized for when input and output type are the same
    @cuda_lower(getattr(mathfuncs, op), types.float64, types.float64)
    def cuda_op(context, builder, sig, args):
        return context.get_function(getattr(numba_libdevice, op), sig)(builder, args)

    @cuda_lower(getattr(mathfuncs, op), types.float32, types.float32)
    def cuda_op_float(context, builder, sig, args):
        return context.get_function(getattr(numba_libdevice, opf(op)), sig)(builder, args)

    #integer specializations
    @cuda_lower(getattr(mathfuncs, op), types.Integer, types.Integer)
    def cuda_op_integer(context, builder, sig, args):
        fname = op
        if sig.return_type != types.float64:
            fname = opf(op)
        x = context.cast(builder, args[0], sig.args[0], sig.return_type)
        y = context.cast(builder, args[1], sig.args[1], sig.return_type)
        fn = context.get_function(getattr(numba_libdevice, fname), signature(sig.return_type, sig.return_type))
        return fn(builder, [x, y])

def binary_opf_factory(op):
    #specialized for when input and output type are the same
    @cuda_lower(getattr(mathfuncs, op), types.float32, types.float32)
    def cuda_opf(context, builder, sig, args):
        return context.get_function(getattr(numba_libdevice, op), sig)(builder, args)

    def cuda_opf_specialized(context, builder, sig, args):
        fn = context.get_function(getattr(numba_libdevice, op), signature(sig.return_type, sig.return_type))
        return fn(builder, [context.cast(builder, args[0], sig.args[0], sig.return_type)])

    cuda_lower(getattr(mathfuncs, op), types.float64, types.float64)(cuda_opf)
    cuda_lower(getattr(mathfuncs, op), types.Integer, types.Integer)(cuda_opf)



binary_ops = ["atan2", "fmod", "copysign", "remainder", "fdim", "hypot", "nextafter", "pow"]

def opf(op):
    return op + 'f'

for op in binary_ops:
    binary_op_factory(op, opf)
    binary_opf_factory(opf(op))



#####################################
#"powi" #int as second arg
#"remquo"

#resolve_remquo  (floating, (floating, floating, *int))
#resolve_remquof (floating, (floating, floating, *int))
#####################################

binary_ops = ["add", "div", "mul"]
for op in binary_ops:
    for rnd in rnd_extns:
        binary_op_fixed_factory('f' + op + rnd, types.float32)
        binary_op_fixed_factory('d' + op + rnd, types.float64)

binary_ops = ["fsub"]
for op in binary_ops:
    for rnd in rnd_extns:
        binary_op_fixed_factory(op + rnd, types.float32)





#####################################
# binary_ops = ["fast_powf", "fast_fdividef"]
#####################################


#####################################
# binary_ops = ["mulhi", "mul24", "hadd",
#               "rhadd", "uhadd", "umul24",
#               "umulhi", "urhadd"]
#####################################


#####################################
# binary_ops = ["mul64hi", "umul64hi"]
#####################################


#####################################
# binary_ops = ["yn", "jn"]

#resolve_yn      (floating, (int, floating))
#resolve_ynf     (floating, (int, floating))
#resolve_jn      (floating, (int, floating))
#resolve_jnf     (floating, (int, floating))
#####################################


#####################################
# binary_ops = ["scalbn", "ldexp"]

#resolve_scalbn  (floating, (floating, int))
#resolve_scalbnf (floating, (floating, int))
#resolve_ldexp   (floating, (floating, int))
#resolve_ldexpf  (floating, (floating, int))
#####################################










def minmax_op_factory(op, minmax):
    @cuda_lower(getattr(mathfuncs, op), types.Number, types.Number)
    def cuda_op(context, builder, sig, args):
        fn = context.get_function(minmax, signature(sig.return_type, sig.return_type, sig.return_type))
        args = [context.cast(builder, arg, fromtyp, sig.return_type) for arg, fromtyp in zip(args, sig.args)]
        return fn(builder, args)

for op in ["min", "llmin", "fmin", "fminf", "umin", "ullmin"]:
    minmax_op_factory(op, min)

for op in ["max", "llmax", "fmax", "fmaxf", "umax", "ullmax"]:
    minmax_op_factory(op, max)




#############
#TERNARY OPS#
#############

#resolve_fma(self, mod):
#resolve_fma_rd(self, mod):
#resolve_fma_rn(self, mod):
#resolve_fma_ru(self, mod):
#resolve_fma_rz(self, mod):

#resolve_fmaf(self, mod):
#resolve_fmaf_rd(self, mod):
#resolve_fmaf_rn(self, mod):
#resolve_fmaf_ru(self, mod):
#resolve_fmaf_rz(self, mod):

# sad
# usad
# byte_perm




#use numba's for this
    # 3.236. __nv_nan
    # 3.237. __nv_nanf
