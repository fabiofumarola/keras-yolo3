#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='keras-yolo3',
      version='0.0.1',
      description='yolo keras implementation',
      author='Fabio Fumarola',
      author_email='fabiofumarola@gmail.com',
      packages=find_packages(),
      setup_requires=['lxml', 'numpy', 'Cython'],
      install_requires=[
          'numpy',
          'pillow',
          'keras',
          'opencv-python',
          'pytest',
          'tensorflow<=1.14',
          'tqdm',
          'scikit-learn'
      ],
      package_data={
          # If any package contains *.txt include them:
          '': ['font/*.oft'],
      }
      )
