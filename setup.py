
from __future__ import absolute_import, print_function
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs
from os import path
#------------------------------------------------------------------------------------
_dir = path.dirname(__file__)

with open(path.join(_dir,'spotipy','version.py'), encoding="utf-8") as f:
    exec(f.read())


class build_ext_openmp(build_ext):
    # https://www.openmp.org/resources/openmp-compilers-tools/
    # python setup.py build_ext --help-compiler
    openmp_compile_args = {
        'msvc':  [['/openmp']],
        'intel': [['-qopenmp']],
        '*':     [['-fopenmp'], ['-O3', '-Xpreprocessor','-fopenmp']],
    }
    openmp_link_args = openmp_compile_args # ?

    def build_extension(self, ext):
        compiler = self.compiler.compiler_type.lower()
        if compiler.startswith('intel'):
            compiler = 'intel'
        if compiler not in self.openmp_compile_args:
            compiler = '*'

        # thanks to @jaimergp (https://github.com/conda-forge/staged-recipes/pull/17766)
        # issue: qhull has a mix of c and c++ source files
        #        gcc warns about passing -std=c++11 for c files, but clang errors out
        compile_original = self.compiler._compile
        def compile_patched(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # remove c++ specific (extra) options for c files
            if src.lower().endswith('.c'):
                extra_postargs = [arg for arg in extra_postargs if not arg.lower().startswith('-std')]
            return compile_original(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # monkey patch the _compile method
        self.compiler._compile = compile_patched

        # store original args
        _extra_compile_args = list(ext.extra_compile_args)
        _extra_link_args    = list(ext.extra_link_args)

        
        # try compiler-specific flag(s) to enable openmp
        for compile_args, link_args in zip(self.openmp_compile_args[compiler], self.openmp_link_args[compiler]):
            
            try:
                ext.extra_compile_args = _extra_compile_args + compile_args
                ext.extra_link_args    = _extra_link_args    + link_args
                print('>>> try building with OpenMP support: ', compile_args, link_args)
                return super(build_ext_openmp, self).build_extension(ext)
            except:
                print(f">>> compiling with '{' '.join(compile_args)}' failed")

        print('>>> compiling with OpenMP support failed, re-trying without')

        ext.extra_compile_args = _extra_compile_args
        ext.extra_link_args    = _extra_link_args
        return super(build_ext_openmp, self).build_extension(ext)

external_root = path.join(_dir, 'spotipy', 'lib', 'external')
nanoflann_root = path.join(external_root, 'nanoflann')

setup(
    name='spotipy',
    version=__version__,
    description='spotipy',
    long_description_content_type='text/markdown',
    author='Martin Weigert',
    author_email='martin.weigert@epfl.ch',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    cmdclass={'build_ext': build_ext_openmp},
    ext_modules=[
        Extension(
            'spotipy.lib.spotflow2d',
            sources = ['spotipy/lib/spotflow2d.cpp'],
            extra_compile_args = ['-std=c++11'],
            include_dirs = get_numpy_include_dirs() + [nanoflann_root]
            ),

        Extension(
            'spotipy.lib.point_nms',
            sources = ['spotipy/lib/point_nms.cpp'],
            extra_compile_args = ['-std=c++11'],
            include_dirs = get_numpy_include_dirs() + [nanoflann_root],
        ),
        Extension(
            'spotipy.lib.filters',
            sources = ['spotipy/lib/filters.cpp'],
            extra_compile_args = ['-std=c++11'],
            include_dirs = get_numpy_include_dirs() ,
        )
    ],

    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'tifffile',
        'imageio',
        'scikit-image',
        'csbdeep',
        'stardist',
        'pandas'
        'opencv-python'
    ],

)