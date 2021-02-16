# OpenConcept - A Conceptual Design toolkit with efficient gradients implemented in the OpenMDAO framework

### Author: Benjamin J. Brelje

[![Build Status](https://github.com/mdolab/openconcept/workflows/Build/badge.svg?branch=master)](https://github.com/mdolab/openconcept/actions?query=branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/mdolab/openconcept/badge.svg)](https://coveralls.io/github/mdolab/openconcept)
[![Documentation Status](https://readthedocs.org/projects/openconcept/badge/?version=latest)](https://openconcept.readthedocs.io/en/latest/?badge=latest)

OpenConcept is a new toolkit for the conceptual design of aircraft. OpenConcept was developed in order to model and optimize aircraft with electric propulsion at low computational cost. The tools are built on top of NASA Glenn's [OpenMDAO](http://openmdao.org/) framework, which in turn is written in Python.

The following charts show more than 250 individually optimized hybrid-electric light twin aircraft (similar to a King Air C90GT). Optimizing hundreds of configurations can be done in a couple of hours on a standard laptop computer.

![Example charts](docs/_static/images/readme_charts.png)

The reason for OpenConcept's efficiency is the analytic derivatives built into each analysis routine and component. Accurate, efficient derivatives enable the use of Newton nonlinear equation solutions and gradient-based optimization at low computational cost.

## Documentation

Automatically-generated documentation is available at (https://openconcept.readthedocs.io)

## Getting Started

1. Clone the repo to disk
2. Navigate to the root openconcept folder
3. Run `python setup.py install` to install the package
4. Navigate to the `examples` folder
5. Run `python TBM850.py` to test OpenConcept on a single-engine turboprop aircraft (the TBM 850)
6. Look at the `examples/aircraft data/TBM850.py` folder to play with the assumptions / config / geometry and see the effects on the output result

`examples/HybridTwin.py` is set up to do MDO in a grid of specific energies and design ranges and save the results to disk. Visualization utilities will be added soon (to produce contour plots as shown in this Readme)

### Requirements

This toolkit requires the use of OpenMDAO 3.0.0 or later (2.8+ will work with minor arg changes) and will evolve rapidly as general utilities are moved from OpenConcept into the main OpenMDAO repository. OpenMDAO requires a late numpy and scipy.

## Citation

Please cite this software by reference to the conference paper:

### Plaintext

Benjamin J. Brelje and Joaquim R.R.A. Martins. "Development of a Conceptual Design Model for Aircraft Electric Propulsion with Efficient Gradients", 2018 AIAA/IEEE Electric Aircraft Technologies Symposium, AIAA Propulsion and Energy Forum, (AIAA 2018-4979) DOI: TBD

### Bibtex

```
@inproceedings{Brelje2018,
	Address = {{C}incinnati,~{OH}},
	Author = {Benjamin J. Brelje and Joaquim R. R. A. Martins},
	Booktitle = {2018 AIAA/IEEE Electric Aircraft Technologies Symposium},
	Month = jul,
	Title = {Development of a Conceptual Design Model for Aircraft Electric Propulsion with Efficient Gradients},
	Year = 2018,
    Number = {AIAA-2018-4979},
	}
```

## Contributing
A contributor's guide is coming third (after completing documentation and automatic testing). I'm open to pull requests and issues in the meantime. Stay tuned.
