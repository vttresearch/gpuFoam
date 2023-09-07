# topaz - An expression template library for generic GPU/CPU ranges


## Overview

The library provides a minimal range implementation which supports both GPU and CPU computations. The library is header only,
and can be compiled with the Nvidia cuda compiler (nvcc) or with the Gnu compiled (gcc). If compiled with nvcc all the
operations run on the gpu.

Although, the library is intended to be as dependency-free as possible, the following libraries are still required:

- C++14
- [Boost v1.67+](https://www.boost.org/) Extension to standard library. Required when compiled with gcc.
- [thrust](https://thrust.github.io/) Cuda extensions to the standard library. Required when compiled with nvcc.


## Credits
topaz is adapted from the [newton](https://github.com/jaredhoberock/newton) by Jared Hoberock. 
