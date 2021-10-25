import os
import cffi
import inspect

ffi = cffi.FFI()
debug = False
use_openmp = True
directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

with open('%s/cffi.h' % directory) as my_header:
    ffi.cdef(my_header.read())

with open('%s/cffi.c' % directory) as my_source:
    if debug:
        ffi.set_source(
            '_cffi',
            my_source.read(),
            extra_compile_args=['-pedantic', '-Wall', '-g', '-O0'],
        )
    else:
        if use_openmp:
            ffi.set_source(
                '_cffi',
                my_source.read(),
                extra_compile_args=['-fopenmp', '-D use_openmp', '-O3', '-march=native'],
                extra_link_args=['-fopenmp'],
            )
        else:
            ffi.set_source('_cffi',
                           my_source.read(),
                           extra_compile_args=['-O3', '-march=native'],
                           )

ffi.compile()
from _cffi import *