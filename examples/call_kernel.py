import numpy as np
import pycu

from pycu import autoinit

nvrtc_options = {"std":"c++11"}
kernel = pycu.compile_file("kernel.cu", "kernel", nvrtc_options = nvrtc_options)

d_out = pycu.device_array(32)

kernel[1,32](d_out)
