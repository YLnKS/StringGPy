# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:50:17 2015

@author: ylkomsamo
"""

import sys
sys.path.append('.')

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
import os
import numpy as np

inc_dirs = ['.', '/usr/include', os.path.join('.', 'src'), np.get_include()]

def scan_pyx(dir, files = [], cfiles = []):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scan_pyx(path, files)
    return files

def scan_c(dir, files = []):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".c"):
            files.append(path)
        elif os.path.isdir(path):
            scan_c(path, files)
    return files

# get the list of extensions
c_sources = scan_c("src")

def make_extension(ext_name):
    ext_path = ext_name.replace(".", os.path.sep) + ".pyx"
    return Extension(ext_name,
                     [ext_path] + c_sources,
                     include_dirs=inc_dirs,
                     extra_compile_args=['-fopenmp', '-O0'],
                     extra_link_args=['-fopenmp'])


# get the list of extensions
ext_names = scan_pyx("StringGPy")

# and build up the set of Extension objects
extensions = [make_extension(name) for name in ext_names]

setup(name='StringGPy',
      version='0.2',
      cmdclass={'build_ext': build_ext},
      ext_modules=extensions,
      packages=['StringGPy',
                'StringGPy.samplers',
                'StringGPy.examples',
                'StringGPy.utilities'],
      install_requires=['numpy',
                        'scipy',
                        'statsmodels',
                        'pandas',
                        'matplotlib',
                        'GPy'])
