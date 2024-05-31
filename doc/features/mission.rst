.. _MissionAnalysis:

*******
Mission
*******

The mission analysis computes the fuel and/or battery energy consumption for a specified flight mission.
You can also keep track of the temperature of components if you have thermal models.

In OpenConcept, a **mission** consists of multiple **phases**:
*phases* are the building blocks of the mission.
For example, a basic three-phase mission is composed of a climb phase, cruise phase, and descent phase.

Mission profiles
================

OpenConcept implements several mission profiles that users can use for an analysis.
The missions are implemented in ``openconcept/mission/profiles.py``.
You can also make your own mission profile following the format from these examples.

Basic three-phase mission: ``BasicMission``
-------------------------------------------
This is a basic climb-cruise-descent mission for a fixed-wing aircraft.

For this mission, users should specify the following variables in the run script:

- takeoff and landing altitude ``takeoff|h``. If a ground roll is included, that altitude needs to be set separately via the ground roll's ``fltcond|h`` variable. This parameter should not be used with the ``FullMissionAnalysis`` or ``FullMissionWithReserve`` because it does not properly set takeoff altitudes as you may expect.
- cruise altitude ``cruise|h0``.
- mission range ``mission_range``.
- payload weight ``payload``.
- vertical speed ``<climb, cruise, descent>.fltcond|vs`` for each phase.
- airspeed ``<climb, cruise, descent>.fltcond|Ueas`` for each phase.
- (optional) ``takeoff|v2`` if you include a ground roll phase before climb. The ground roll phase is not included by default.
  
The duration of each phase is automatically set given the cruise altitude and mission range.  

Full mission including takeoff: ``FullMissionAnalysis``
-------------------------------------------------------
This adds a balanced-field takeoff analysis to the three-phase mission.
The additional takeoff phases are:

- ``v0v1``: from a standstill to decision speed (v1). An instance of ``GroundRollPhase``.
- ``v1vr``: from v1 to rotation. An instance of ``GroundRollPhase``.
- ``rotate``: rotation in the air before steady climb. An instance of ``RotationPhase`` or ``RobustRotationPhase``.
- ``v1v0``: energency stopping from v1 to a stop. An instance of ``GroundRollPhase``.

We use ``BLIImplicitSolve`` to solve for the decision speed ``v1`` where the one-engine-out takeoff distance is equal to the braking distance for rejected takeoff.

The optional variables you may set in the run scripts are

- throttle for takeoff phases ``<v0v1, v1vr, rotate>.throttle``, default is 1.0.
- ground rolling friction coefficient ``<v0v1, v1vr, v1v0>.braking``, default is 0.03 for accelerating phases and 0.4 for braking.
- altitude ``<v0v1, v1vr, rotate, v1v0>.fltcond|h``.
- obstacle clearance height ``rotate.h_obs``, default is 35 ft.
- CL/CLmax ration in rotation ``rotate.CL_rotate_mult``, default is 0.83.

It may be necessary to set initial values for the takeoff airspeeds (``<v0v1, v1vr, v1v0>.fltcond|Utrue``) before the solver is called to improve convergence.

Mission with reserve: ``MissionWithReserve``
--------------------------------------------
This adds a reserve mission and loiter phase to the three-phase mission.
Additional variables you need to set in the run script are

- vertical speed and airspeed for additional phases: ``<reserve_climb, reserve_cruise, reserve_descent, loiter>.<fltcond|Ueas, fltcond|vs>``
- reserve range ``reserve_range`` and altitude ``reserve|h0``.
- loiter duration ``loiter_duration`` and loiter altitude ``loiter|h0``.

Full mission with reserve: ``FullMissionWithReserve``
--------------------------------------------
This mission combines ``FullMissionAnalysis`` and ``MissionWithReserve``, so it includes takeoff phases, climb, cruise, descent, and a reserve mission.
Refer to the documentation for ``FullMissionAnalysis`` and ``MissionWithReserve`` to determine which parameters must be set.

Phase types
===========
A phase is a building block of a mission profile.
The phases and relevant classes are implemented in ``openconcept/mission/phases.py``.
Users usually don't need to modify these code when creating their own mission profile.

Steady flight: ``SteadyFlightPhase``
------------------------------------
The ``SteadyFlightPhase`` class can be instantiated for steady climb, cruise, descent, and loiter phases.
For this phase, you need to specify the airspeed (``<phase_name>.fltcond|Ueas``) and vertical speed (``<phase_name>.fltcond|Ueas``) in your run script.
You may optionally set the duration of the phase (``<phase_name>.duration``), or alternatively, the duration can be set automatically in the mission profile group.

To ensure steady flight, both vertical and horizontal accelerations will be set to 0.
It first computes the lift coefficient required for zero vertical accelration; CL is then passed to the aircraft model, which returns the lift and drag.
Then, it solves for the throttle values such that horizontal acceleration is zero.
This is done by solving a system of nonlinear equations (``horizontal acceleration = 0``) w.r.t. throttle using a `BalanceComp <https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/balance_comp.html>`_ for each phase.

Balanced-field takeoff
----------------------
Balanced-field takeoff analysis is implemented in the following classes: ``BFLImplicitSolve``, ``GroundRollPhase``, ``RotationPhase``, ``RobustRotationPhase``, ``ClimbAnglePhase``.
Unlike the steady flight phases, the takeoff phases are not steady and acceleration is non-zero.
Therefore, the engine throttle needs to be specified to compute the acceleration (100% by defalut for accelerating phases and 0 for braking).
Users can also set the throttle manually in the run script.
The acceleration is then integrated to compute the velocity.

.. VTOL transition
.. ---------------
.. This is only relevant to VTOL configurations. Maybe move to a different page (like eVTOL mission and phases) to avoid confusion?

Mission groups
==============
OpenConcept provides some groups that make mission analysis and phase definition easier.

``PhaseGroup``
--------------
This is the base class for an OpenConcept mission phase.
It automatically identifies ``Integrator`` instances within the model and links the time duration variable to them.
It also collects the names of all the integrand states so that the ``TrajectoryGroup`` can find them to link across phases.

``IntegratorGroup``
-------------------
The ``IntegratorGroup`` is an alternative way of setting up and integrator (the ``Integrator`` component is used more frequently).
This group adds an ODE integration component (called ``"ode_integ"``), locates output variables tagged with the "integrate" tag, and automatically connects the tagged rate source to the integrator.

``TrajectoryGroup``
-------------------
This is the base class for a mission profile.
It provides the ``link_phases`` method which is used to connect integration variables across mission phases.
