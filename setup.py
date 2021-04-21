import codecs
import os
import re

import setuptools

with open("README.md", "r", encoding='utf-8') as f:
    long_description = f.read()

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r', encoding='utf-8') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Dependencies
requirements = [
    "scipy",
    "cvxopt",
    "numpy",
    "pandas",
    "joblib",
    "tqdm>=4.43",
    "ishneholterlib",
    "numba",
    "astropy"
]

extras_require = {
    'docs': [
        'numpydoc',
        'sphinx_rtd_theme',
        'sphinx',
        'm2r2',
    ]
}
extras_require['all'] = list(set(i for val in extras_require.values() for i in val))

setuptools.setup(
    name="flirt",
    version=find_version("flirt", "__init__.py"),
    author="ETH Zurich – Chair of Information Management",
    description="Wearable Data Processing Toolkit",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/im-ethz/flirt",
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require=extras_require,

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
    ]
)
