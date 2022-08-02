.. _Install:

************
Installation
************

---------------
Getting Started
---------------
OpenConcept can be pip installed directly from PyPI with ``pip install openconcept``.

To run the examples or edit the source code:

#. Clone the repo to disk (``git clone https://github.com/mdolab/openconcept``)
#. Navigate to the root ``openconcept`` folder
#. Run ``pip install -e .`` to install the package (the ``-e`` can be omitted if not editing the source)

Get started by running the ``TBM850`` example:

#. Navigate to the ``examples`` folder
#. Run ``python TBM850.py`` to test OpenConcept on a single-engine turboprop aircraft (the TBM 850)
#. Look at the ``examples/aircraft data/TBM850.py`` file to play with the assumptions / config / geometry and see the effects on the output result

``examples/HybridTwin.py`` is set up to do MDO in a grid of specific energies and design ranges and save the results to disk. Visualization utilities will be added soon (to produce contour plots as shown in this Readme).

------------
Dependencies
------------
This toolkit requires the use of `OpenMDAO <https://openmdao.org>`__ 3.10.0 or later due to backward-incompatible changes. OpenMDAO requires a recent NumPy and SciPy.
Python 3.8 is recommended since it is the version with which the code is tested, but newer Python versions will likely work as well.
