from numba import types
from numba.cuda.cudadecl import (registry as cuda_registry)
from numba.core.typing.templates import AttributeTemplate

import pycu

_api = {}

def register_op(func, factory, *args):
    # if not isinstance(func, str):
    #     func = func.__name__
    _api[func] = types.Function(factory(func, *args))

@cuda_registry.register_attr
class PyCuModuleTemplate(AttributeTemplate):
    key = types.Module(pycu)

    def __getattr__(self, attr):
        def resolve_attr(mod):
            #note: len("resolve_") = 8
            return _api[attr[8:]]

        if attr[8:] in _api:
            return resolve_attr
        else:
            raise AttributeError(f"module 'core' has no attribute '{attr[8:]}'")

    # def resolve_abs(self, mod):
    #     return api["abs"]
    # def resolve_llabs(self, mod):
    #     return api["llabs"]
    # def resolve_fabs(self, mod):
    #     return api["fabs"]
    # def resolve_fabsf(self, mod):
    #     return api["fabsf"]



    # def resolve_sin(self, mod):
    #     return api["sin"]
    # def resolve_sinf(self, mod):
    #     return api["sinf"]
    # def resolve_cos(self, mod):
    #     return api["cos"]
    # def resolve_cosf(self, mod):
    #     return api["cosf"]
    # def resolve_tan(self, mod):
    #     return api["tan"]
    # def resolve_tanf(self, mod):
    #     return api["tanf"]
    # def resolve_asin(self, mod):
    #     return api["asin"]
    # def resolve_asinf(self, mod):
    #     return api["asinf"]
    # def resolve_acos(self, mod):
    #     return api["acos"]
    # def resolve_acosf(self, mod):
    #     return api["acof"]
    # def resolve_atan(self, mod):
    #     return api["atan"]
    # def resolve_atanf(self, mod):
    #     return api["atanf"]
    # def resolve_sinh(self, mod):
    #     return api["sinh"]
    # def resolve_sinhf(self, mod):
    #     return api["sinhf"]
    # def resolve_cosh(self, mod):
    #     return api["cosh"]
    # def resolve_coshf(self, mod):
    #     return api["coshf"]
    # def resolve_tanh(self, mod):
    #     return api["tanh"]
    # def resolve_tanhf(self, mod):
    #     return api["tanhf"]
    # def resolve_asinh(self, mod):
    #     return api["asinh"]
    # def resolve_asinhf(self, mod):
    #     return api["asinhf"]
    # def resolve_acosh(self, mod):
    #     return api["acosh"]
    # def resolve_acoshf(self, mod):
    #     return api["acoshf"]
    # def resolve_atanh(self, mod):
    #     return api["atanh"]
    # def resolve_atanhf(self, mod):
    #     return api["atanhf"]
    # def resolve_sinpi(self, mod):
    #     return api["sinpi"]
    # def resolve_sinpif(self, mod):
    #     return api["sinpif"]
    # def resolve_cospi(self, mod):
    #     return api["cospi"]
    # def resolve_cospif(self, mod):
    #     return api["cospif"]
    # def resolve_sqrt(self, mod):
    #     return api["sqrt"]
    # def resolve_sqrtf(self, mod):
    #     return api["sqrtf"]
    # def resolve_rsqrt(self, mod):
    #     return api["rsqrt"]
    # def resolve_rsqrtf(self, mod):
    #     return api["rsqrtf"]
    # def resolve_lgamma(self, mod):
    #     return api["lgamma"]
    # def resolve_lgammaf(self, mod):
    #     return api["lgammaf"]
    # def resolve_tgamma(self, mod):
    #     return api["tgamma"]
    # def resolve_tgammaf(self, mod):
    #     return api["tgammaf"]
    # def resolve_j0(self, mod):
    #     return api["j0"]
    # def resolve_j0f(self, mod):
    #     return api["j0f"]
    # def resolve_j1(self, mod):
    #     return api["j1"]
    # def resolve_j1f(self, mod):
    #     return api["j1f"]
    # def resolve_y0(self, mod):
    #     return api["y0"]
    # def resolve_y0f(self, mod):
    #     return api["y0f"]
    # def resolve_y1(self, mod):
    #     return api["y1"]
    # def resolve_y1f(self, mod):
    #     return api["y1f"]
    # def resolve_exp(self, mod):
    #     return api["exp"]
    # def resolve_expf(self, mod):
    #     return api["expf"]
    # def resolve_exp10(self, mod):
    #     return api["exp10"]
    # def resolve_exp10f(self, mod):
    #     return api["exp10f"]
    # def resolve_exp2(self, mod):
    #     return api["exp2"]
    # def resolve_exp2f(self, mod):
    #     return api["exp2f"]
    # def resolve_expm1(self, mod):
    #     return api["expm1"]
    # def resolve_expm1f(self, mod):
    #     return api["expm1f"]
    # def resolve_log(self, mod):
    #     return api["log"]
    # def resolve_logf(self, mod):
    #     return api["logf"]
    # def resolve_log1p(self, mod):
    #     return api["log1p"]
    # def resolve_log1pf(self, mod):
    #     return api["log1pf"]
    # def resolve_log10(self, mod):
    #     return api["log10"]
    # def resolve_log10f(self, mod):
    #     return api["log10f"]
    # def resolve_log2(self, mod):
    #     return api["log2"]
    # def resolve_log2f(self, mod):
    #     return api["log2f"]
    # def resolve_logb(self, mod):
    #     return api["logb"]
    # def resolve_logbf(self, mod):
    #     return api["logbf"]
    # def resolve_erf(self, mod):
    #     return api["erf"]
    # def resolve_erff(self, mod):
    #     return api["erff"]
    # def resolve_erfc(self, mod):
    #     return api["erfc"]
    # def resolve_erfcf(self, mod):
    #     return api["erfcf"]
    # def resolve_erfcinv(self, mod):
    #     return api["erfcinv"]
    # def resolve_erfcinvf(self, mod):
    #     return api["erfcinvf"]
    # def resolve_erfcx(self, mod):
    #     return api["erfcx"]
    # def resolve_erfcxf(self, mod):
    #     return api["erfcxf"]
    # def resolve_erfinv(self, mod):
    #     return api["erfinv"]
    # def resolve_erfinvf(self, mod):
    #     return api["erfinvf"]
    # def resolve_normcdf(self, mod):
    #     return api["normcdf"]
    # def resolve_normcdff(self, mod):
    #     return api["normcdff"]
    # def resolve_normcdfinv(self, mod):
    #     return api["normcdfinv"]
    # def resolve_normcdfinvf(self, mod):
    #     return api["normcdfinvf"]
    # def resolve_ceil(self, mod):
    #     return api["ceil"]
    # def resolve_ceilf(self, mod):
    #     return api["ceilf"]
    # def resolve_floor(self, mod):
    #     return api["floor"]
    # def resolve_floorf(self, mod):
    #     return api["floorf"]
    # def resolve_trunc(self, mod):
    #     return api["trunc"]
    # def resolve_truncf(self, mod):
    #     return api["truncf"]
    # def resolve_round(self, mod):
    #     return api["round"]
    # def resolve_roundf(self, mod):
    #     return api["roundf"]
    # def resolve_rint(self, mod):
    #     return api["rint"]
    # def resolve_rintf(self, mod):
    #     return api["rintf"]
    # def resolve_nearbyint(self, mod):
    #     return api["nearbyint"]
    # def resolve_nearbyintf(self, mod):
    #     return api["nearbyintf"]
