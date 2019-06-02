import os
import re
import sys
# To use a consistent encoding
from codecs import open as copen
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with copen(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("gaussian_process", "__version__.py")

test_deps = ['pytest', 'pytest-cov', 'coveralls', 'validate_version_code', 'codacy-coverage',  'keras', 'silence_tensorflow', 'extra_keras_utils', 'numpy', 'holdouts_generator']

extras = {
    'test': test_deps,
}

setup(
    name='gaussian_process',
    version=__version__,
    description="Wrapper for sklearn.gp_minimize for a simpler parameter specification using nested dictionaries.",
    long_description=long_description,
    url="https://github.com/LucaCappelletti94/gaussian_process",
    author="Luca Cappelletti",
    author_email="cappelletti.luca94@gmail.com",
    # Choose your license
    license='MIT',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    tests_require=test_deps,
    install_requires=[
        "sklearn", "auto_tqdm", "pandas", "scikit-optimize", "deflate_dict", "dict_hash"
    ],
    extras_require=extras,
)