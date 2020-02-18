# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="pytorch_helpers",
    version="0.0.1",
    packages=find_packages(),
    author="Yannik Rath",
    description="Helpers for creation/training of pytorch models.",
    install_requires=["torch", "numpy"],
)
