.. _Turboprop-tutorial:

*********************
OpenConcept Turboprop
*********************

This tutorial builds on the :ref:`previous tutorial <Integrator-tutorial>` by vastly improving the aircraft model.
We'll use components from OpenConcept to model the turboshaft engine, constant speed propeller, aerodynamics, and weight.
We'll also use a new mission profile that models takeoff by performing a balanced field length computation.
The model here could be considered the first "useful" aircraft model since it more accurately models the relationship between throttle, thrust, and fuel flow and also the aerodynamics.
This aircraft model is based on the Socata TBM 850 aircraft.

Imports
=======

Aircraft model
==============

How variables make it down into the model (variables defined in the data file).

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
