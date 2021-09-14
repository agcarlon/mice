# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read()


setup(
    name='mice',
    version='1.0.0',
    description='Multi-iteration Stochastic Estimator',
    long_description=readme,
    author='Andre Gustavo Carlon',
    author_email='agcarlon@gmail.com',
    url='https://bitbucket.org/agcarlon',
    license=license,
    install_requires=requirements,
    packages=find_packages(exclude=('tests', 'docs'))
)
