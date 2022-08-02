.. _Overview:

********
Overview
********

.. note::
    This is a blurb from a recent paper of mine but it's probably not what should stay here.

OpenConcept is a software toolkit originally developed for mission analysis of electric and hybrid-electric fixed-wing aircraft.
It is built using the OpenMDAO framework developed at NASA, which facilitates automated coupling of multidisciplinary analysis blocks and system-level analytic gradient computation.
OpenConcept uses these analytic derivatives to enable rapid problem convergence with gradient-based solvers and efficient gradient-based optimization.

An OpenConcept aircraft model takes in aircraft design variables (e.g. wing area and propulsion system sizing variables), throttle position, lift coefficient, and flight conditions, and outputs thrust, weight and drag.
The aircraft design variables are assumed to start at default values that match the reference aircraft model, including the operating empty weight.
The architecture builder assembles the propulsion system that is used in the aircraft model.
A parabolic drag polar, based on parameters from the reference aircraft model, is used to compute the drag.

The climb, cruise, and descent segments assume steady flight, which is achieved by solving for the throttle setting and lift coefficient such that the horizontal and vertical accelerations are zero at each integration point in the mission.
The climb and descent segments use a predefined profile, determined by a set airspeed and vertical speed.
The length of the cruise segment is then set such that the total mission range meets the value input by the user.
A Newton solver converges this system.

OpenConcept's numerical integration scheme uses Simpson's Rule to integrate state variables with an all-at-once approach.
For performance, OpenConcept uses vectorized computations in each mission segment.
This means that time-marching ordinary differential equation integration approaches cannot be used because vectorized quantities must be computed all at once.
The integrator integrates necessary variables, such as fuel flow and airspeed, to compute quantities needed for the mission analysis, such as fuel weight and distance flown, respectively.
The integrator can also be used in combination with OpenConcept components to enable novel analyses, including time-accurate battery temperature modeling and unsteady thermal management system analysis.

.. note::
    Learn OpenMDAO first!
