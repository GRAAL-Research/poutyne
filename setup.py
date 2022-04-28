import os
import subprocess

from setuptools import setup, find_packages

current_file_path = os.path.abspath(os.path.dirname(__file__))


def get_readme():
    readme_file_path = os.path.join(current_file_path, 'README.md')
    with open(readme_file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_version():
    version_file_path = os.path.join(current_file_path, 'version.txt')
    with open(version_file_path, 'r', encoding='utf-8') as f:
        version = f.read().strip()

    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:  # pylint: disable=broad-except
        sha = 'Unknown'

    if os.getenv('POUTYNE_RELEASE_BUILD') != '1':
        version += '.dev1'
        if sha != 'Unknown':
            version += '+' + sha[:7]
    return version


def write_version_python_file(version):
    version_python_file = os.path.join(current_file_path, 'poutyne/version.py')
    with open(version_python_file, 'w', encoding='utf-8') as f:
        f.write(f"__version__ = {repr(version)}\n")


def main():
    readme = get_readme()

    version = get_version()
    print("Building version", version)
    write_version_python_file(version)

    packages = find_packages()
    setup(
        name='Poutyne',
        version=version,
        author='Frédérik Paradis',
        author_email='fredy_14@live.fr',
        url='https://poutyne.org',
        download_url='https://github.com/GRAAL-Research/poutyne/archive/v' + version + '.zip',
        license='LGPLv3',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        packages=packages,
        install_requires=['numpy', 'torch', 'torchmetrics'],
        python_requires='>=3.7',
        description='A simplified framework and utilities for PyTorch.',
        long_description=readme,
        long_description_content_type='text/markdown',
        extras_require={
            "colorama": "colorama>=0.4.3",
            "scikit-learn": "scikit-learn>=0.23.2",
            "tensorboard": "tensorboard>=2.4.0",
            "tensorboardX": "tensorboardX>=2.1",
            "torchvision": "torchvision>=0.8.1",
            "pandas": "pandas>=2.0.0.0",
            "mlflow": "mlflow>=1.12.1",
        },
    )


if __name__ == '__main__':
    main()
