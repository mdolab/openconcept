.. propmodeling:

********************
Propulsion Modeling
********************

OpenConcept is designed to facilitate bottoms-up modeling of aircraft propulsion architectures with conceptual-level fidelity.
Electric and fuel-burning components are supported.

Single Turboprop Example
------------------------

This example illustrates the simples possible case (turboprop engine connected to a propeller).
The propulsion system is instantiated as an OpenMDAO `Group`.

*Source: `examples/propulsion_layouts/simple_turboprop.py`*

.. embed-code::
    examples.propulsion_layouts.simple_turboprop.TurbopropPropulsionSystem

Series Hybrid Example
---------------------

This example illustrates the complexities which arise when electrical components are included.

*Source: `examples.propulsion_layouts.simple_series_hybrid.py`*

.. embed-code::
    examples.propulsion_layouts.simple_series_hybrid.SingleSeriesHybridElectricPropulsionSystem

Components
----------

.. toctree::
    :maxdepth: 2

    motor.rst