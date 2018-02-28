from setuptools import setup, find_packages

packages = find_packages()
setup(
    name='PyToune',
    version='0.2',
    author='Frédérik Paradis',
    author_email='fredy_14@live.fr',
    url = 'http://pytoune.org',
    download_url='https://github.com/GRAAL-Research/pytoune/archive/v0.2.zip',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=packages,
    install_requires=['numpy'],
    python_requires='>=3',
    description='A Keras-like framework and utilities for PyTorch.'
)
