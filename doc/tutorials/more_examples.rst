.. _More-examples:

*************
More Examples
*************

While we work on more extensive tutorials, you can learn more about OpenConcept by diving into the source code and playing around with OpenConcept's models.
This page suggests logical next places to look to gain an understanding of what is possible.

Propulsion modeling
===================

The :ref:`turboprop example <Turboprop-tutorial>` uses OpenConcept's ``TurbopropPropulsionSystem``.
If you have not already looked at the underlying code for that model, that's a good place to start.
The component's source code is in ``openconcept/propulsion/systems/simple_turboprop.py`` and can be found on GitHub `here <https://github.com/mdolab/openconcept/blob/main/openconcept/propulsion/systems/simple_turboprop.py>`__.

After that, we recommend looking at the ``TwinTurbopropPropulsionSystem`` in the same file to understand how to build models with more than one propulsor that properly handle the failed engine case on takeoff.
This propulsion system is used in the ``KingAirC90GT.py`` example.

For an introduction to hybrid electric propulsion systems, have a look at the ``TwinSeriesHybridElectricPropulsionSystem``.
This model is in ``openconcept/propulsion/systems/simple_series_hybrid.py`` or on GitHub `here <https://github.com/mdolab/openconcept/blob/main/openconcept/propulsion/systems/simple_series_hybrid.py>`__.
The ``HybridTwin.py`` example aircraft shows how to use this propulsion system and sets up an optimization problem for it.

Finally, the ``B738.py`` example shows the use of the CFM56 surrogate model from pyCycle.
There are similar propulsion models for the N+3 geared turbofan and a parallel hybrid version of the same N+3 engine.


Thermal management modeling
===========================

One of OpenConcept's key contributions is the ability to model an aircraft thermal management system and the associated unsteady heat flows.
``AllElectricSinglePropulsionSystemWithThermal_Incompressible`` models an all-electric propulsion architecture with a thermal management system.
The code is located in ``openconcept/propulsion/systems/simple_all_electric.py`` and on GitHub `here <https://github.com/mdolab/openconcept/blob/main/openconcept/propulsion/systems/simple_all_electric.py>`__.
The ``ElectricSinglewithThermal.py`` example models the TBM aircraft from the turboprop example, but retrofit with this all-electric propulsion system.

The ``TwinSeriesHybridElectricThermalPropulsionSystem`` model adds a thermal management system to the twin-engine series hybrid propulsion system.
That model can be found in ``openconcept/propulsion/systems/thermal_series_hybrid.py`` or on GitHub `here <https://github.com/mdolab/openconcept/blob/main/openconcept/propulsion/systems/thermal_series_hybrid.py>`__.
The ``HybridTwin_thermal.py`` example shows how to use this propulsion system.

The most detailed thermal management system model is in the ``N3_HybridSingleAisle_Refrig.py`` example aircraft.
It models a parallel hybrid single aisle commercial transport aircraft.
The model includes an electric motor with a cooling jacket, a battery with a bandolier cooling system, a refrigerator, compressible ducts, pumps, and even coolant hose models.

Other useful places to look
===========================

The ``B738_VLM_drag.py`` and ``B738_aerostructural.py`` examples show how to use the OpenAeroStruct VLM and aerostructural models.
The ``B738_sizing.py`` example demonstrates the use of the weight buildup, drag buildup, and :math:`C_{L, \text{max}}` estimate.
