.. _Turboprop-tutorial:

*********
Turboprop
*********

This tutorial builds on the :ref:`previous tutorial <Integrator-tutorial>` by vastly improving the aircraft model.
We'll use components from OpenConcept to model the turboshaft engine, constant speed propeller, aerodynamics, and weight.
We'll also use a new mission profile that models takeoff by performing a balanced field length computation.
The model here could be considered the first "useful" aircraft model since it more accurately models the relationship between throttle, thrust, and fuel flow and also the aerodynamics.
This aircraft model is based on the Socata TBM 850 aircraft.

Imports
=======

.. literalinclude:: ../../examples/TBM850.py
    :start-after: # rst Imports (beg)
    :end-before: # rst Imports (end)

Compared to the previous examples, this adds a handful of imports from OpenConcept.
We import the propulsion system, aerodynamic model, weight estimate, and a few math utilities.
We also import a new type of mission analysis we haven't seen in previous tutorials: ``FullMissionAnalysis``.
This includes a balanced field length takeoff calculation.
Finally, we import ``acdata`` from the TBM's data file.
``acdata`` is a dictionary that organizes aircraft parameters (this is an alternative to what we've done so far of defining these values in the mission group).

Aircraft model
==============

This aircraft model builds on the aircraft in the :ref:`integrator tutorial <Integrator-tutorial>` by replacing the simple thrust and drag model we developed with much more detailed OpenConcept models.
The propulsion system uses OpenConcept's ``TurbopropPropulsionSystem``, which couples a turboshaft engine to a constant speed propeller.
We also use OpenConcept's ``PolarDrag`` component to compute drag using a simple drag polar.
The final addition is an operating empty weight (OEW) computation.
The OEW output is not used in the weight calculation, but it is a useful output to know (perhaps for optimization) and shows off another OpenConcept component.

Let's take a look at the aircraft model as a whole and then we'll dive into each part.

.. literalinclude:: ../../examples/TBM850.py
    :start-after: # rst Aircraft (beg)
    :end-at: # rst Weight (end)

Options
-------

The options are the same as the previous tutorials.

.. literalinclude:: ../../examples/TBM850.py
    :start-after: # rst Options
    :end-at: # rst Setup
    :dedent: 4

Setup
-----

Now we'll break down the components of the setup method for the aircraft model.

Propulsion
~~~~~~~~~~

We use OpenConcept's ``TurbopropPropulsionSystem`` to estimate the thrust as a function of throttle.
It uses a turboprop, which assumes constant TSFC, connected to a constant speed propeller, which uses a propeller map.

.. literalinclude:: ../../examples/TBM850.py
    :start-after: # rst Propulsion (beg)
    :end-at: # rst Propulsion (end)
    :dedent: 8

The propulsion system requires some flight conditions, engine rating, propeller diameter, and throttle.
We also set the propeller speed to 2000 rpm.
The propulsion system computes thrust, which is promoted, and fuel flow, which will be connected to the fuel burn integrator.

Aerodynamics
~~~~~~~~~~~~

Weights
~~~~~~~

Mission
=======

Show analysis group, DV comp, etc.

Run script
==========

Functions to set up the mission and solvers.

.. image:: assets/turboprop_takeoff_results.svg

.. image:: assets/turboprop_mission_results.svg

The N2 diagram for the model is the following:

.. embed-n2::
  ../examples/TBM850.py

Summary
=======

.. literalinclude:: ../../examples/TBM850.py
