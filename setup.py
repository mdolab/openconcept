from distutils.core import setup
import re
import os

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('openconcept/__init__.py').read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "readme.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='openconcept',
    version=__version__,
    description="Open Aircraft Conceptual Design Tools",
    long_description=long_description,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    keywords='aircraft design optimization multidisciplinary multi-disciplinary analysis',
    author='Benjamin Brelje',
    author_email='bbrelje@umich.edu',
    url='http://www.brelje.net',
    download_url='https://github.com/bbrelje/openconcept',
    license='MIT License',
    packages=[
        'openconcept',
        'openconcept.analysis',
        'openconcept.analysis.atmospherics',
        'openconcept.components',
        'openconcept.components.empirical_data',
        'openconcept.utilities',
        'openconcept.utilities.math'
    ],
    install_requires=[
        'six',
        'scipy>=1.0.0',
        'numpy>=1.14.0',
        'openmdao>=3.0.0, <=3.2.1',
    ],
    extras_require = {
        'testing':  ["pytest", "openmdao[docs]"]
      },
)
