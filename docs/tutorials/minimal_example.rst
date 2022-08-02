.. _Minimal-example-tutorial:

***************
Minimal Example
***************

This example shows how to set up an OpenConcept aircraft and mission analysis model.
The goal here is to use only what is absolutely necessary with the idea of introducing the starting point for building more complicated and detailed models.

This example uses a simplified aircraft model and basic mission profile with climb, cruise, and descent phases.

Aircraft model
==============

At it's most basic, an OpenConcept aircraft model takes in a lift coefficient and throttle position (from 0 to 1) and returns thrust, weight, and drag.
In the code, these variables are named ``"fltcond|CL"``, ``"throttle"``, ``"thrust"``, ``"weight"``, and ``"drag"``, respectively.
In practice, thrust, weight, and drag are very complicated functions of the inputs and have consequences for fuel burn, component temperatures, and battery state of charge.
This is where the OpenConcept models are used.

The complexity can grow rapidly, so for now we will not use any of these OpenConcept models; instead we develop a minimal aircraft model.
We assume constant weight across the whole mission.
Thrust is modeled simply as maximum thrust times the throttle.
Drag is computed as lift divided by lift-to-drag ratio.

.. literalinclude:: ../../examples/minimal.py
    :start-after: # rst Aircraft (beg)
    :end-before: # rst Aircraft (end)

Mission
=======

Basic three-phase mission.

Run script
==========

Setup, define necessary segment variables.
