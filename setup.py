from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='safe-screeening',
    version='0.1',
    description='Safe Screening',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
)   
