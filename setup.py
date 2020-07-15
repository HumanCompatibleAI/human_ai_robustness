#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='human_ai_robustness',
      version='0.0.1',
      description='This package has shared components.',
      author='Paul Knott',
      author_email='paul.knott@nottingham.ac.uk',
      packages=find_packages(),
      install_requires=[
        'numpy==1.15.1',
      ],
    )