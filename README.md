# PyCu

A Python package

Driver API bindings

CUDA made easy

## Intro

PyCu provides ctypes bindings to the CUDA Driver API along with a few other CUDA related libraries, each of which can be imported separately.

To access the direct API's of the Driver, NVRTC, or NVVM simply import any of the following

```
from pycu import driver
from pycu import nvrtc
from pycu import nvvm
```

To access a higher level API for each you can import "core" from the above libraries

```
from pycu.driver import core as driver
from pycu.nvrtc import core as nvrtc
from pycu.nvvm import core as nvvm
```

If you want to import it all import pycu

```
import pycu
```

pycu also has its own "core" which contains Jitity, a jit compiler that uses NVRTC and NVVM to compile kernel functions.

```
from pycu import core as pycu
```

Currently ```import pycu``` will auto import all of the above packages including ```pycy.core``` but this can be turned off (and may be turned of by defualt in the future) if only a select few of the functionalities are needed. This is a project I started developing in order to work on something else. Almost the entirity of the Driver API currently has a corrisponding python binding. The missing bindings are primarily related to the lesser used features (at least for me) like the graph API. There are a number of incomplete features as of right now, almost all related to the higher level Python API. They are mostly added when I need them in the other project.

## Examples

There is currently one example located here https://github.com/uchytilc/Recursive-Grid-Based-Integration demonstrating how to compile and call kernels written in C, create device arrays, and query GPU properties.
