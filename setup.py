"""
Backward compatibility shim for setup.py.
Modern packaging uses pyproject.toml (PEP 517/518).
"""
from setuptools import setup

setup()
