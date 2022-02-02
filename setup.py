import os
import pathlib

from setuptools import setup

setup(name = 'pycu',
	  version = '0.1',
	  description = 'CUDA python API bindings',
	  author = 'Chris Uchytil',
	  author_email = 'uchytilc@uw.edu',
	  # platforms=["win-amd64", 'win32'],
	  license = "LICENSE.txt",
	  packages = ['pycu'],
	)
