from __future__ import absolute_import, print_function
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs
from os import path
#------------------------------------------------------------------------------------
_dir = path.dirname(__file__)

with open(path.join(_dir,'spotipy','version.py'), encoding="utf-8") as f:
    exec(f.read())

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
        # 'opencv-python'
    ],

)
