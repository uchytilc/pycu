# #TO DO
# 	#impliment compiler options like arch
# 	#use some compiler options as additional keys when storing ptx
# 	#edit open_nvrtclib to include wind32

#Add pycu to default include headers
	#load from pycu/include



# #test_cuda_driver.py
# 	#https://github.com/numba/numba/blob/c5461dd295663559b55f7e59280df9317062887c/numba/cuda/tests/cudadrv/test_cuda_driver.py
# #test_cuda_memory.py
# 	# https://github.com/numba/numba/blob/e5364a3da418cffdb0de36238d0bb1346322118b/numba/cuda/tests/cudadrv/test_cuda_memory.py

limits_h = climits = """
	#pragma once

	#if defined _WIN32 || defined _WIN64
	  #define __WORDSIZE 32
	#else
	  #if defined __x86_64__ && !defined __ILP32__
	    #define __WORDSIZE 64
	  #else
	    #define __WORDSIZE 32
	  #endif
	#endif
	#define MB_LEN_MAX  16
	#define CHAR_BIT    8
	#define SCHAR_MIN   (-128)
	#define SCHAR_MAX   127
	#define UCHAR_MAX   255
	#ifdef __CHAR_UNSIGNED__
	  #define CHAR_MIN   0
	  #define CHAR_MAX   UCHAR_MAX
	#else
	  #define CHAR_MIN   SCHAR_MIN
	  #define CHAR_MAX   SCHAR_MAX
	#endif
	#define SHRT_MIN    (-32768)
	#define SHRT_MAX    32767
	#define USHRT_MAX   65535
	#define INT_MIN     (-INT_MAX - 1)
	#define INT_MAX     2147483647
	#define UINT_MAX    4294967295U
	#if __WORDSIZE == 64
	  # define LONG_MAX  9223372036854775807L
	#else
	  # define LONG_MAX  2147483647L
	#endif
	#define LONG_MIN    (-LONG_MAX - 1L)
	#if __WORDSIZE == 64
	  #define ULONG_MAX  18446744073709551615UL
	#else
	  #define ULONG_MAX  4294967295UL
	#endif
	#define LLONG_MAX  9223372036854775807LL
	#define LLONG_MIN  (-LLONG_MAX - 1LL)
	#define ULLONG_MAX 18446744073709551615ULL;
"""

################################
# stddef_h ="""
# #pragma once
# #include <climits>
# namespace __jitify_stddef_ns {
#   typedef unsigned long size_t;
#   typedef   signed long ptrdiff_t;
# } // namespace __jitify_stddef_ns
# namespace std { using namespace __jitify_stddef_ns; }
# using namespace __jitify_stddef_ns;
# """

stddef_h = """
	#pragma once
	#include <climits>
	typedef unsigned long size_t;
	typedef   signed long ptrdiff_t;
"""
################################

stdio_h = """
	#pragma once
	#include <stddef.h>

	#define FILE int
	int fflush ( FILE * stream );
	int fprintf ( FILE * stream, const char * format, ... );
"""

# math_h = """
	# #pragma once

	# #if __cplusplus >= 201103L
	# #define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\
	# 	inline double      f(double x)         { return ::f(x); } \\
	# 	inline float       f##f(float x)       { return ::f(x); } \\
	# 	/*inline long double f##l(long double x) { return ::f(x); }*/ \\
	# 	inline float       f(float x)          { return ::f(x); } \\
	# 	/*inline long double f(long double x)    { return ::f(x); }*/
	# #else
	# #define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\
	# 	inline double      f(double x)         { return ::f(x); } \\
	# 	inline float       f##f(float x)       { return ::f(x); } \\
	# 	/*inline long double f##l(long double x) { return ::f(x); }*/
	# #endif
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(cos)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(sin)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(tan)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(acos)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(asin)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(atan)
	# template<typename T> inline T atan2(T y, T x) { return ::atan2(y, x); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(cosh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(sinh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(tanh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(exp)
	# template<typename T> inline T frexp(T x, int* exp) { return ::frexp(x, 
	# exp); }
	# template<typename T> inline T ldexp(T x, int  exp) { return ::ldexp(x, 
	# exp); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(log)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(log10)
	# template<typename T> inline T modf(T x, T* intpart) { return ::modf(x, 
	# intpart); }
	# template<typename T> inline T pow(T x, T y) { return ::pow(x, y); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(sqrt)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(ceil)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(floor)
	# template<typename T> inline T fmod(T n, T d) { return ::fmod(n, d); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(fabs)
	# template<typename T> inline T abs(T x) { return ::abs(x); }
	# #if __cplusplus >= 201103L
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(acosh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(asinh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(atanh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(exp2)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(expm1)
	# template<typename T> inline int ilogb(T x) { return ::ilogb(x); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(log1p)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(log2)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(logb)
	# template<typename T> inline T scalbn (T x, int n)  { return ::scalbn(x, 
	# n); }
	# template<typename T> inline T scalbln(T x, long n) { return ::scalbn(x, 
	# n); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(cbrt)
	# template<typename T> inline T hypot(T x, T y) { return ::hypot(x, y); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(erf)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(erfc)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(tgamma)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(lgamma)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(trunc)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(round)
	# template<typename T> inline long lround(T x) { return ::lround(x); }
	# template<typename T> inline long long llround(T x) { return ::llround(x); 
	# }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)
	# template<typename T> inline long lrint(T x) { return ::lrint(x); }
	# template<typename T> inline long long llrint(T x) { return ::llrint(x); 
	# }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(nearbyint)
	# // TODO: remainder, remquo, copysign, nan, nextafter, nexttoward, fdim,
	# // fmax, fmin, fma
	# #endif
	# #undef DEFINE_MATH_UNARY_FUNC_WRAPPER

	# #define M_PI 3.14159265358979323846
	# // Note: Global namespace already includes CUDA math funcs
# """

# math_h = """
	# #pragma once
	# namespace __jitify_math_ns {
	# #if __cplusplus >= 201103L
	# #define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\
	# 	inline double      f(double x)         { return ::f(x); } \\
	# 	inline float       f##f(float x)       { return ::f(x); } \\
	# 	/*inline long double f##l(long double x) { return ::f(x); }*/ \\
	# 	inline float       f(float x)          { return ::f(x); } \\
	# 	/*inline long double f(long double x)    { return ::f(x); }*/
	# #else
	# #define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\
	# 	inline double      f(double x)         { return ::f(x); } \\
	# 	inline float       f##f(float x)       { return ::f(x); } \\
	# 	/*inline long double f##l(long double x) { return ::f(x); }*/
	# #endif
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(cos)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(sin)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(tan)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(acos)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(asin)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(atan)
	# template<typename T> inline T atan2(T y, T x) { return ::atan2(y, x); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(cosh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(sinh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(tanh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(exp)
	# template<typename T> inline T frexp(T x, int* exp) { return ::frexp(x, 
	# exp); }
	# template<typename T> inline T ldexp(T x, int  exp) { return ::ldexp(x, 
	# exp); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(log)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(log10)
	# template<typename T> inline T modf(T x, T* intpart) { return ::modf(x, 
	# intpart); }
	# template<typename T> inline T pow(T x, T y) { return ::pow(x, y); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(sqrt)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(ceil)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(floor)
	# template<typename T> inline T fmod(T n, T d) { return ::fmod(n, d); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(fabs)
	# template<typename T> inline T abs(T x) { return ::abs(x); }
	# #if __cplusplus >= 201103L
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(acosh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(asinh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(atanh)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(exp2)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(expm1)
	# template<typename T> inline int ilogb(T x) { return ::ilogb(x); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(log1p)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(log2)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(logb)
	# template<typename T> inline T scalbn (T x, int n)  { return ::scalbn(x, 
	# n); }
	# template<typename T> inline T scalbln(T x, long n) { return ::scalbn(x, 
	# n); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(cbrt)
	# template<typename T> inline T hypot(T x, T y) { return ::hypot(x, y); }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(erf)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(erfc)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(tgamma)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(lgamma)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(trunc)
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(round)
	# template<typename T> inline long lround(T x) { return ::lround(x); }
	# template<typename T> inline long long llround(T x) { return ::llround(x); 
	# }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)
	# template<typename T> inline long lrint(T x) { return ::lrint(x); }
	# template<typename T> inline long long llrint(T x) { return ::llrint(x); 
	# }
	# DEFINE_MATH_UNARY_FUNC_WRAPPER(nearbyint)
	# // TODO: remainder, remquo, copysign, nan, nextafter, nexttoward, fdim,
	# // fmax, fmin, fma
	# #endif
	# #undef DEFINE_MATH_UNARY_FUNC_WRAPPER
	# } // namespace __jitify_math_ns
	# namespace std { using namespace __jitify_math_ns; }
	# #define M_PI 3.14159265358979323846
	# // Note: Global namespace already includes CUDA math funcs
	# //using namespace __jitify_math_ns;;
# """




stdint_h = '''
#pragma once
#include <climits>

//namespace __jitify_stdint_ns {

typedef signed char      int8_t;
typedef signed short     int16_t;
typedef signed int       int32_t;
typedef signed long long int64_t;
typedef signed char      int_fast8_t;
typedef signed short     int_fast16_t;
typedef signed int       int_fast32_t;
typedef signed long long int_fast64_t;
typedef signed char      int_least8_t;
typedef signed short     int_least16_t;
typedef signed int       int_least32_t;
typedef signed long long int_least64_t;
typedef signed long long intmax_t;
typedef signed long      intptr_t; //optional
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char      uint_fast8_t;
typedef unsigned short     uint_fast16_t;
typedef unsigned int       uint_fast32_t;
typedef unsigned long long uint_fast64_t;
typedef unsigned char      uint_least8_t;
typedef unsigned short     uint_least16_t;
typedef unsigned int       uint_least32_t;
typedef unsigned long long uint_least64_t;
typedef unsigned long long uintmax_t;
typedef unsigned long      uintptr_t; //optional
#define INT8_MIN    SCHAR_MIN
#define INT16_MIN   SHRT_MIN
#define INT32_MIN   INT_MIN
#define INT64_MIN   LLONG_MIN
#define INT8_MAX    SCHAR_MAX
#define INT16_MAX   SHRT_MAX
#define INT32_MAX   INT_MAX
#define INT64_MAX   LLONG_MAX
#define UINT8_MAX   UCHAR_MAX
#define UINT16_MAX  USHRT_MAX
#define UINT32_MAX  UINT_MAX
#define UINT64_MAX  ULLONG_MAX
#define INTPTR_MIN  LONG_MIN
#define INTMAX_MIN  LLONG_MIN
#define INTPTR_MAX  LONG_MAX
#define INTMAX_MAX  LLONG_MAX
#define UINTPTR_MAX ULONG_MAX
#define UINTMAX_MAX ULLONG_MAX
#define PTRDIFF_MIN INTPTR_MIN
#define PTRDIFF_MAX INTPTR_MAX
#define SIZE_MAX    UINT64_MAX

// } // namespace __jitify_stdint_ns
// namespace std { using namespace __jitify_stdint_ns; }
// using namespace __jitify_stdint_ns;;
'''


math_constants_h = """
#if !defined(__MATH_CONSTANTS_H__)
#define __MATH_CONSTANTS_H__

/* single precision constants */
#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_NAN_F            __int_as_float(0x7fffffff)
#define CUDART_MIN_DENORM_F     __int_as_float(0x00000001)
#define CUDART_MAX_NORMAL_F     __int_as_float(0x7f7fffff)
#define CUDART_NEG_ZERO_F       __int_as_float(0x80000000)
#define CUDART_ZERO_F           0.0f
#define CUDART_ONE_F            1.0f
#define CUDART_SQRT_HALF_F      0.707106781f
#define CUDART_SQRT_HALF_HI_F   0.707106781f
#define CUDART_SQRT_HALF_LO_F   1.210161749e-08f
#define CUDART_SQRT_TWO_F       1.414213562f
#define CUDART_THIRD_F          0.333333333f
#define CUDART_PIO4_F           0.785398163f
#define CUDART_PIO2_F           1.570796327f
#define CUDART_3PIO4_F          2.356194490f
#define CUDART_2_OVER_PI_F      0.636619772f
#define CUDART_SQRT_2_OVER_PI_F 0.797884561f
#define CUDART_PI_F             3.141592654f
#define CUDART_L2E_F            1.442695041f
#define CUDART_L2T_F            3.321928094f
#define CUDART_LG2_F            0.301029996f
#define CUDART_LGE_F            0.434294482f
#define CUDART_LN2_F            0.693147181f
#define CUDART_LNT_F            2.302585093f 
#define CUDART_LNPI_F           1.144729886f
#define CUDART_TWO_TO_M126_F    1.175494351e-38f
#define CUDART_TWO_TO_126_F     8.507059173e37f
#define CUDART_NORM_HUGE_F      3.402823466e38f
#define CUDART_TWO_TO_23_F      8388608.0f
#define CUDART_TWO_TO_24_F      16777216.0f
#define CUDART_TWO_TO_31_F      2147483648.0f
#define CUDART_TWO_TO_32_F      4294967296.0f
#define CUDART_REMQUO_BITS_F    3
#define CUDART_REMQUO_MASK_F    (~((~0)<<CUDART_REMQUO_BITS_F))
#define CUDART_TRIG_PLOSS_F     105615.0f

/* double precision constants */
#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN              __longlong_as_double(0xfff8000000000000ULL)
#define CUDART_NEG_ZERO         __longlong_as_double(0x8000000000000000ULL)
#define CUDART_MIN_DENORM       __longlong_as_double(0x0000000000000001ULL)
#define CUDART_ZERO             0.0
#define CUDART_ONE              1.0
#define CUDART_SQRT_TWO         1.4142135623730951e+0
#define CUDART_SQRT_HALF        7.0710678118654757e-1
#define CUDART_SQRT_HALF_HI     7.0710678118654757e-1
#define CUDART_SQRT_HALF_LO   (-4.8336466567264567e-17)
#define CUDART_THIRD            3.3333333333333333e-1
#define CUDART_TWOTHIRD         6.6666666666666667e-1
#define CUDART_PIO4             7.8539816339744828e-1
#define CUDART_PIO4_HI          7.8539816339744828e-1
#define CUDART_PIO4_LO          3.0616169978683830e-17
#define CUDART_PIO2             1.5707963267948966e+0
#define CUDART_PIO2_HI          1.5707963267948966e+0
#define CUDART_PIO2_LO          6.1232339957367660e-17
#define CUDART_3PIO4            2.3561944901923448e+0
#define CUDART_2_OVER_PI        6.3661977236758138e-1
#define CUDART_PI               3.1415926535897931e+0
#define CUDART_PI_HI            3.1415926535897931e+0
#define CUDART_PI_LO            1.2246467991473532e-16
#define CUDART_SQRT_2PI         2.5066282746310007e+0
#define CUDART_SQRT_2PI_HI      2.5066282746310007e+0
#define CUDART_SQRT_2PI_LO    (-1.8328579980459167e-16)
#define CUDART_SQRT_PIO2        1.2533141373155003e+0
#define CUDART_SQRT_PIO2_HI     1.2533141373155003e+0
#define CUDART_SQRT_PIO2_LO   (-9.1642899902295834e-17)
#define CUDART_SQRT_2OPI        7.9788456080286536e-1
#define CUDART_L2E              1.4426950408889634e+0
#define CUDART_L2E_HI           1.4426950408889634e+0
#define CUDART_L2E_LO           2.0355273740931033e-17
#define CUDART_L2T              3.3219280948873622e+0
#define CUDART_LG2              3.0102999566398120e-1
#define CUDART_LG2_HI           3.0102999566398120e-1
#define CUDART_LG2_LO         (-2.8037281277851704e-18)
#define CUDART_LGE              4.3429448190325182e-1
#define CUDART_LGE_HI           4.3429448190325182e-1
#define CUDART_LGE_LO           1.09831965021676510e-17
#define CUDART_LN2              6.9314718055994529e-1
#define CUDART_LN2_HI           6.9314718055994529e-1
#define CUDART_LN2_LO           2.3190468138462996e-17
#define CUDART_LNT              2.3025850929940459e+0
#define CUDART_LNT_HI           2.3025850929940459e+0
#define CUDART_LNT_LO         (-2.1707562233822494e-16)
#define CUDART_LNPI             1.1447298858494002e+0
#define CUDART_LN2_X_1024       7.0978271289338397e+2
#define CUDART_LN2_X_1025       7.1047586007394398e+2
#define CUDART_LN2_X_1075       7.4513321910194122e+2
#define CUDART_LG2_X_1024       3.0825471555991675e+2
#define CUDART_LG2_X_1075       3.2360724533877976e+2
#define CUDART_TWO_TO_23        8388608.0
#define CUDART_TWO_TO_52        4503599627370496.0
#define CUDART_TWO_TO_53        9007199254740992.0
#define CUDART_TWO_TO_54        18014398509481984.0
#define CUDART_TWO_TO_M54       5.5511151231257827e-17
#define CUDART_TWO_TO_M1022     2.22507385850720140e-308
#define CUDART_TRIG_PLOSS       2147483648.0
#define CUDART_DBL2INT_CVT      6755399441055744.0

#endif /* !__MATH_CONSTANTS_H__ */
"""

nvrtc_headers = {"limits.h":limits_h,
				 "climits":climits,
				 "stddef.h":stddef_h,
				 "stdio.h":stdio_h,
				 "stdint.h":stdint_h,
				 "stdint":stdint_h,
				 "math_constants.h":math_constants_h}
