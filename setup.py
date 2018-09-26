#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path
import sys

here = path.abspath(path.dirname(__file__))

long_description = """"
Mesoscale Activity Map EPhys Pipeline
see README.md for further information.
"""

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split()

setup(
    name='mapephys',
    version='0.0.1',
    description="MAP Ephys Pipeline",
    long_description=long_description,
    author='TODO: Correct Attribution',
    author_email='TODO: Correct Maintainer Email',
    license='TODO: Resolve',
    url='https://github.com/mesoscale-activity-map/map-ephys',
    keywords='neuroscience electrophysiology science datajoint',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
<<<<<<< HEAD
    scripts=['scripts/mapshell.py'],
=======
    scripts=['scripts/mapshell.py', 'scripts/map-mock-data.py'],
>>>>>>> 501054a9959db869b540f76be926cd1f02eec04f
    install_requires=requirements,
)
