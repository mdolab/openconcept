from setuptools import setup, find_packages
import re
import os

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("openconcept/__init__.py").read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="openconcept",
    version=__version__,
    description="Open aircraft conceptual design tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords="aircraft design optimization multidisciplinary multi-disciplinary analysis",
    author="Benjamin J. Brelje and Eytan J. Adler",
    author_email="",
    url="https://github.com/mdolab/openconcept",
    download_url="https://github.com/mdolab/openconcept",
    license="MIT License",
    packages=find_packages(include=["openconcept*"]),
    package_data={
        # engine deck data
        "openconcept.propulsion.empirical_data": ["**/*.npy"],
    },
    include_package_data=True,
    install_requires=[
        # Update the oldest package versions in the GitHub Actions build file, the readme,
        # and the index.rst file in the docs when you change these
        "numpy >=1.20, <2",
        "scipy>=1.7.0",
        "openmdao >=3.21",
    ],
    extras_require={
        "testing": ["pytest", "pytest-cov", "coverage", "openaerostruct", "parameterized", "om-pycycle>=4.4.0"],
        "docs": ["sphinx_mdolab_theme", "openaerostruct<=2.7.1"],
        "plot": ["matplotlib"],
    },
)
