import sys
sys.path.append('.')

from distutils.core import setup, Extension
import os
import site

inc_dirs =  [os.path.join(sp, 'numpy', 'core', 'include') for sp in site.getsitepackages()]
inc_dirs += ['/usr/include']
c_source = [os.path.join(rt, fl) for (rt,dr,fls) in\
	os.walk(os.path.join('.', 'c_extensions')) for fl in fls if fl.endswith('.c')]

setup(name='cStringGPy',\
		version='1.0',\
		description='This package implements string Gaussian processes methods.',\
		author='Yves-Laurent KOM SAMO',\
		author_email='ylks@robots.ox.ac.uk',\
		ext_modules=[Extension('cStringGPy', c_source, library_dirs=['/usr/lib'],\
			extra_compile_args=['-llapacke', '-lblas'], extra_link_args=['-llapacke', '-lblas']),],\
		include_dirs=inc_dirs)