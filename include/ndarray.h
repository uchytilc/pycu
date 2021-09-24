#ifndef NDARRAY_H
#define NDARRAY_H

template<typename T, int DIM = 1>
struct NDArray{
    void*    meminfo = 0;
    void*    parent = 0;
    ssize_t  nitems;
    ssize_t  itemsize = sizeof(T);
    T*       data;
    ssize_t  shape[DIM];
    ssize_t  strides[DIM];
};

#endif