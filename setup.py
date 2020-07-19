from distutils.core import setup

setup(
    name='openconcept',
    version='0.3.0',
    description="Open Aircraft Conceptual Design Tools",
    long_description="""OpenConcept is a set of analysis routines and components to aid
    in the conceptual design of aircraft with unconventional energy and propulsion architectures.
    It natively supports tracking of electrical power and heat as well as conventional fuel flows,
    and can handle cases ranging from 100 percent electric to 100 percent fuel-burning propulsion.
    """,
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
        'openmdao>=3.0.0',
        'pytest',
    ]
)
