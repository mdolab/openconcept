# OpenConcept - A conceptual design toolkit with efficient gradients implemented in the OpenMDAO framework

### Author: Benjamin J. Brelje and Eytan J. Adler

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
2. Navigate to the root `openconcept` folder
3. Run `pip install -e .` to install the package (the `-e` can be omitted if not editing the source)
4. Navigate to the `examples` folder
5. Run `python TBM850.py` to test OpenConcept on a single-engine turboprop aircraft (the TBM 850)
6. Look at the `examples/aircraft data/TBM850.py` folder to play with the assumptions / config / geometry and see the effects on the output result

`examples/HybridTwin.py` is set up to do MDO in a grid of specific energies and design ranges and save the results to disk. Visualization utilities will be added soon (to produce contour plots as shown in this Readme)

### Requirements

This toolkit requires the use of [OpenMDAO](https://openmdao.org/) 3.10.0 or later. OpenMDAO requires a late NumPy and SciPy.

## Citation

Please cite this software by reference to the [conference paper](https://www.researchgate.net/publication/326263660_Development_of_a_Conceptual_Design_Model_for_Aircraft_Electric_Propulsion_with_Efficient_Gradients):

Benjamin J. Brelje and Joaquim R.R.A. Martins, "Development of a Conceptual Design Model for Aircraft Electric Propulsion with Efficient Gradients", 2018 AIAA/IEEE Electric Aircraft Technologies Symposium, AIAA Propulsion and Energy Forum, (AIAA 2018-4979) DOI: 10.2514/6.2018-4979

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

Eytan J. Adler and Joaquim R.R.A. Martins, "Aerostructural wing design optimization considering full mission analysis", 2022 AIAA SciTech Forum, San Diego, CA, January 2022. DOI: 10.2514/6.2022-0382

```
@inproceedings{Adler2022a,
	author      = {Eytan J. Adler and Joaquim R. R. A. Martins},
	title       = {Aerostructural wing design optimization considering full mission analysis},
	booktitle   = {AIAA SciTech Forum},
	doi         = {10.2514/6.2022-0382},
	month       = {January},
	year        = {2022}
	}
```

## Contributing
A contributor's guide is coming third (after completing documentation and automatic testing). I'm open to pull requests and issues in the meantime. Stay tuned.
