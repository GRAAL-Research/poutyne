import os
from setuptools import setup, find_packages


current_file_path = os.path.abspath(os.path.dirname(__file__))
version_file_path = os.path.join(current_file_path, 'pytoune/version.py')
exec(compile(open(version_file_path, "rb").read(), version_file_path, 'exec'), globals(), locals())
version = __version__

packages = find_packages()
setup(
    name='PyToune',
    version=version,
    author='Frédérik Paradis',
    author_email='fredy_14@live.fr',
    url = 'http://pytoune.org',
    download_url='https://github.com/GRAAL-Research/pytoune/archive/v' + version + '.zip',
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
    install_requires=['numpy', 'pandas', 'torch'],
    python_requires='>=3',
    description='A Keras-like framework and utilities for PyTorch.',
    test_suite='nose.collector',
    tests_require=['nose'],
)
