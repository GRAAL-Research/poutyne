from setuptools import setup, find_packages

packages = find_packages()
setup(
    name='Pytoune',
    version='0.0.1',
    author='GRAAL',
    author_email='info@graal.com',
    packages=packages,
    description='Pytorch related utilities.'
)
