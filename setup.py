# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


with open('REQUIREMENTS') as f:
    requirements = f.read()


setup(
    name='mice',
    version='0.1.1',
    description='Multi-iteration Stochastic Estimator',
    long_description_content_type="text/markdown",
    long_description=readme,
    author='Andre Gustavo Carlon',
    author_email='agcarlon@gmail.com',
    url='https://bitbucket.org/agcarlon/mice',
    license='GPL v3',
    install_requires=requirements,
    packages=find_packages(exclude=('tests', 'docs'))
)
