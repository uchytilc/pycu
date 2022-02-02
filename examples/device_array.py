import pycu
from pycu import autoinit
import numpy as np

nbytes = 1 << 10
d_buff = pycu.device_buffer(nbytes)
print(d_buff)

# d_buff.copy_from_host()

size = 100
dtype = np.float32
d_ary = pycu.device_array(size, dtype)
print(d_ary)

# d_ary.copy_from_host()

shape = (8,8,8)
dtype = np.int32
d_nd_ary = pycu.device_ndarray(shape, dtype)
print(d_nd_ary)

# d_nd_ary.copy_from_host()

# pycu.to_device_ndarray
# pycu.to_device
# pycu.to_device_array