#ifndef NDARRAY_STRUCT_H_
#define NDARRAY_STRUCT_H_

#include <stdint.h>

template<typename T>
typedef struct {
    void*    meminfo = 0;
    void*    parent = 0;
    ssize_t  nitems;
    ssize_t  itemsize = sizeof(T);
    T*       data;
    size_t   dim;
    // intptr_t shape; #c_ssize_t
    // intptr_t strides;
} ndarray;


// [(f'shape{dim_ext[(ax + 22)%25]*(ax//25 + 1)}', ctypes.c_ssize_t) for ax in range(ndim)] + \
// [(f'stirdes{dim_ext[(ax + 22)%25]*(ax//25 + 1)}', ctypes.c_ssize_t) for ax in range(ndim)]



// npy_intp shape_and_strides[];

// NPY_INTP
// The enumeration value for a signed integer type which is the same size as a (void *) pointer. This is the type used by all arrays of indices.

		//https://github.com/numba/numba/blob/7e8538140ce3f8d01a5273a39233b5481d8b20b1/numba/cuda/compiler.py
            // meminfo = ctypes.c_void_p(0)
            // parent = ctypes.c_void_p(0)
            // nitems = c_intp(devary.size)
            // itemsize = c_intp(devary.dtype.itemsize)
            // data = ctypes.c_void_p(driver.device_pointer(devary))
            // kernelargs.append(meminfo)
            // kernelargs.append(parent)
            // kernelargs.append(nitems)
            // kernelargs.append(itemsize)
            // kernelargs.append(data)
            // for ax in range(devary.ndim):
            //     kernelargs.append(c_intp(devary.shape[ax]))
            // for ax in range(devary.ndim):
            //     kernelargs.append(c_intp(devary.strides[ax]))



	// # fields = [("meminfo", pointer(int8)), #pointer(int8) #pointer(void)
	// # 		  ("parent", pointer(int8)), #pointer(int8) #pointer(void)
	// # 		  ("nitems", ssize_t),
	// # 		  ("itemsize", ssize_t),
	// # 		  ("data", pointer(dtype)), #pointer(float32) #pointer(void)
	// # 		  ("shape", ShapeStride(ndim)), #Pointer(ShapeStride(ndim)) #Array(ndim, ssize_t)
	// # 		  ("strides", ShapeStride(ndim))] #Pointer(ShapeStride(ndim)) #Array(ndim, ssize_t)


#endif