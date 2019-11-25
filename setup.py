from distutils.core import setup, Extension
from Cython.Build import cythonize

include_dirs = ['.', \
              '/usr/include', \
              '/usr/local/cuda/include' ]
sources =['cudnn.pyx', \
          'prof_conv_bwd_filter.cc']
extra_objects = ['/usr/lib/x86_64-linux-gnu/libcudnn.so', \
                        '/usr/local/cuda/lib64/libcudart.so' ]

ext = Extension("cudnn", sources=["cudnn.pyx", "prof_conv_bwd_filter.cc"], language='c++', 
                include_dirs=include_dirs,
                extra_compile_args=['-std=c++11','-O3'], extra_link_args=["-std=c++11"],
                extra_objects=extra_objects)
setup(name="cudnn", ext_modules=cythonize([ext]))

"""
OBJS = main.o prof_conv_bwd_filter.o
NVOBJS = /usr/lib/x86_64-linux-gnu/libcudnn.so \
         /usr/local/cuda/lib64/libcudart.so
NAME = main.out
"""
