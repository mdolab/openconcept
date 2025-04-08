# OpenConcept - A conceptual design toolkit with efficient gradients implemented in the OpenMDAO framework

### Authors: Benjamin J. Brelje and Eytan J. Adler

[![Build](https://github.com/mdolab/openconcept/workflows/Build/badge.svg?branch=main)](https://github.com/mdolab/openconcept/actions?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mdolab/openconcept/branch/main/graph/badge.svg?token=RR8CN3IOSL)](https://codecov.io/gh/mdolab/openconcept)
[![Documentation](https://readthedocs.com/projects/mdolab-openconcept/badge/?version=latest)](https://mdolab-openconcept.readthedocs-hosted.com/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/openconcept)](https://pypi.org/project/openconcept/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/openconcept)](https://pypi.org/project/openconcept/)

OpenConcept is a new toolkit for the conceptual design of aircraft. OpenConcept was developed in order to model and optimize aircraft with electric propulsion at low computational cost. The tools are built on top of NASA Glenn's [OpenMDAO](http://openmdao.org/) framework, which in turn is written in Python.

OpenConcept is capable of modeling a wide range of propulsion systems, including detailed thermal management systems.
The following figure (from [this paper](https://doi.org/10.3390/aerospace9050243)) shows one such system that is modeled in the `N3_HybridSingleAisle_Refrig.py` example.

<h2 align="center">
    <img src="https://raw.githubusercontent.com/mdolab/openconcept/main/doc/_static/images/full_parallel_system_chiller.png" width="500" />
</h2>

The following charts show more than 250 individually optimized hybrid-electric light twin aircraft (similar to a King Air C90GT). Optimizing hundreds of configurations can be done in a couple of hours on a standard laptop computer.

![Example charts](https://raw.githubusercontent.com/mdolab/openconcept/main/doc/_static/images/readme_charts.png)

The reason for OpenConcept's efficiency is the analytic derivatives built into each analysis routine and component. Accurate, efficient derivatives enable the use of Newton nonlinear equation solutions and gradient-based optimization at low computational cost.

## Documentation

Automatically-generated documentation is available at (https://mdolab-openconcept.readthedocs-hosted.com/en/latest/).

To build the docs locally, install the `sphinx_mdolab_theme` via `pip`. Then enter the `doc` folder in the root directory and run `make html`. The built documentation can be viewed by opening `_build/html/index.html`. OpenAeroStruct is required (also installable via `pip`) to build the OpenAeroStruct portion of the source docs.

## Getting Started

OpenConcept can be pip installed directly from PyPI

```shell
pip install openconcept
```

To run the examples or edit the source code:
1. Clone the repo to disk (`git clone https://github.com/mdolab/openconcept`)
2. Navigate to the root `openconcept` folder
3. Run `pip install -e .` to install the package (the `-e` can be omitted if not editing the source)

Get started by following the tutorials in the documentation to learn the most important parts of OpenConcept.
The features section of the documentation describes most of the components and system models available in OpenConcept.

### Requirements

<!-- Remember to change doc/index.rst too! -->

OpenConcept is tested regularly on builds with the oldest and latest supported package versions. The package versions in the oldest and latest builds are the following:

| Package | Oldest | Latest |
| ------- | ------- | ------ |
| Python | 3.8 | 3.11 |
| OpenMDAO | 3.21 | latest |
| NumPy | 1.20 | 1.26 |
| SciPy | 1.7.0 | latest |
| OpenAeroStruct | 2.7.1 | 2.7.1 |

## Citation

Please cite this software by reference to the [conference paper](https://www.researchgate.net/publication/326263660_Development_of_a_Conceptual_Design_Model_for_Aircraft_Electric_Propulsion_with_Efficient_Gradients):

Benjamin J. Brelje and Joaquim R. R. A. Martins, "Development of a Conceptual Design Model for Aircraft Electric Propulsion with Efficient Gradients", 2018 AIAA/IEEE Electric Aircraft Technologies Symposium, AIAA Propulsion and Energy Forum, (AIAA 2018-4979) DOI: 10.2514/6.2018-4979

```
@inproceedings{Brelje2018a,
	address = {{C}incinnati,~{OH}},
	author = {Benjamin J. Brelje and Joaquim R. R. A. Martins},
	booktitle = {Proceedings of the AIAA/IEEE Electric Aircraft Technologies Symposium},
	doi = {10.2514/6.2018-4979},
	month = {July},
	title = {Development of a Conceptual Design Model for Aircraft Electric Propulsion with Efficient Gradients},
	year = {2018}
}
```

If using the integrated OpenAeroStruct VLM or aerostructural aerodynamic models, please cite the following [conference paper](https://www.researchgate.net/publication/357559489_Aerostructural_wing_design_optimization_considering_full_mission_analysis):

Eytan J. Adler and Joaquim R. R. A. Martins, "Efficient Aerostructural Wing Optimization Considering Mission Analysis", Journal of Aircraft, 2022. DOI: 10.2514/1.c037096

```
@article{Adler2022d,
	author = {Adler, Eytan J. and Martins, Joaquim R. R. A.},
	doi = {10.2514/1.c037096},
	issn = {1533-3868},
	journal = {Journal of Aircraft},
	month = {December},
	publisher = {American Institute of Aeronautics and Astronautics},
	title = {Efficient Aerostructural Wing Optimization Considering Mission Analysis},
	year = {2022}
}
```
