.. _Energy-storage:

**************
Energy Storage
**************

This module contains components that can store energy.

Battery models
==============

``SimpleBattery``
-----------------

This component simple uses a simple equation to relate the electrical power draw to the heat generated: :math:`\text{heat} = \text{electricity load} (1 - \eta)`.
Cost is assumed to be a linear function of weight.
Component sizing margin is computed which describes the electrical load to the max power of the battery (defined by battery weight and specific power).
This is not automatically forced to be less than one, so the user is responsible for checking/enforcing this in an analysis or optimization.

.. warning::
    This component does not track its state of charge, so without an additional integrator there is no way to know when the battery has been depleted. For this reason, it is recommended to use the ``SOCBattery``.

``SOCBattery``
--------------

This component uses the same model as the ``SimpleBattery``, but adds an integrator to compute the state of charge (from 0.0 to 1.0).
By default, it starts at a state of charge of 1.0 (100% charge).

Hydrogen tank model
===================

``LH2TankNoBoilOff``
--------------------

This provides a physics-based structural weight model of a liquid hydrogen tank.
It includes an integrator for computing the current mass of LH2 inside the tank.
For details, see the source code.
