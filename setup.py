#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: de Moura, K.
"""
from setuptools import setup, find_packages
import os
import codecs

setup_path = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(setup_path, 'README.md'), encoding='utf-8-sig') as f:
    README = f.read()

setup(
    name="shsv",
    version="0.1.0",
    description="Offline Handwritten Signature Verification Using a Stream-Based Approach",
    long_description=README,
    long_description_content_type='text/markdown',
    author="kdmoura",
    url="https://github.com/kdMoura/stream_hsv",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.5.0',
        'tqdm==4.66.5'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
   #  entry_points={
   #     "console_scripts": [
   #         "shsv-data=shsv.data:main",
   #         "shsv-model=shsv.model:main",
   #         "shsv-evaluation=shsv.evaluation:main",
   #     ],
   # },
    python_requires='>=3.12',
)

