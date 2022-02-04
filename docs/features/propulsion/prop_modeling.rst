.. propmodeling:

********************
Propulsion Modeling
********************

OpenConcept is designed to facilitate bottoms-up modeling of aircraft propulsion architectures with conceptual-level fidelity.
Electric and fuel-burning components are supported.

Single Turboprop Example
------------------------

This example illustrates the simples possible case (turboprop engine connected to a propeller).
The propulsion system is instantiated as an OpenMDAO ``Group``.

*Source: ``examples/propulsion_layouts/simple_turboprop.py``*

.. literalinclude:: /../examples/propulsion_layouts/simple_turboprop.py
    :pyobject: TurbopropPropulsionSystem
    :language: python

Series Hybrid Example
---------------------

This example illustrates the complexities which arise when electrical components are included.

*Source: ``examples/propulsion_layouts/simple_series_hybrid.py``*

.. literalinclude:: /../examples/propulsion_layouts/simple_series_hybrid.py
    :pyobject: SingleSeriesHybridElectricPropulsionSystem
    :language: python

Components
----------

.. toctree::
    :maxdepth: 2

    motor.rst