.. _MissionAnalysis:

****************
Mission Analysis
****************

The mission analysis computes the fuel and/or battery energy consumption for a specified flight mission.
You can also keep track of the temperature of components if you have thermal models.

In OpenConcept, a **mission** consists of multiple **phases**:
*phases* are the building blocks of the mission.
For example, a basic three-phase mission is composed of a climb phase, cruise phase, and descent phase.

Available missions
==================

OpenConcept implements several mission profiles that users can use for an analysis.
The missions are implemented in ``openconcept/PATH/mission_profiles.py``.
You can also make your own mission profile following the format from these examples.

Basic three-phase mission: ``BasicMission``
-------------------------------------------
This is a basic climb-cruise-descent mission for a fixed-wing aircraft.

For this mission, users should specify the following variables in the runscript:

- takeoff altitude ``takeoff|h0``, default is 0 ft.
- cruise altitude ``cruise|h0``.
- mission range ``mission_range``.
- payload weight ``payload``.
- vertical speed ``<climb, cruise, descent>.fltcond|vs`` for each phase.
- air speed ``<climb, cruise, descent>.fltcond|Ueas`` for each phase.
- (Optional) ``takeoff|v2`` is you include a ground roll phase before climb. The ground roll phase is not included by default.
  
The duration of each phase is automatically set given the cruise altitude and mission range.  

Full mission including takeoff: ``FullMissionAnalysis``
-------------------------------------------------------
This adds a balanced-field takeoff analysis to the three-phase mission.
The additional takeoff phases are:

- ``v0v1``: from a standstill to decision speed (v1). An instance of ``GroundRollPhase``.
- ``v1vr``: from v1 to rotation. An instance of ``GroundRollPhase``.
- ``rotate``: rotation in the air before steady climb. An instance of ``RotationPhase`` or ``RobustRotationPhase``.
- ``v1vr``: energency stopping from v1 to a stop. An instance of ``GroundRollPhase``.

We use ``BLIImplicitSolve`` to solve for the decision speed ``v1`` where the one-engine-out takeoff distance is equal to the braking distance for rejected takeoff.

The optional variables you may set in the runscripts are

- throttle for takeoff phases ``<v0v1, v1vr, rotate>.throttle``, default is 1.0.
- ground rolling friction coefficient ``<v0v1, v1vr, v1v0>.braking``, default is 0.03 for accelerating phases and 0.4 for braking.
- altitude ``<v0v1, v1vr, rotate, v1v0>.fltcond|h``.
- obstacle clearance height ``rotate.h_obs``, default is 35 ft.
- CL/CLmax ration in rotation ``rotate.CL_rotate_mult``, default is 0.83.

Mission with reserve: ``MissionWithReserve``
--------------------------------------------
This adds a reserve mission and loiter phase to the three-phase mission.
Additional variables you need to set in the runscript are

- vertical speed and air speed for additional phases: ``<reserve_climb, reserve_cruise, reserve_descent, loiter>.<fltcond|Ueas, fltcond|vs>``
- reserve range ``reserve_range`` and altitude ``reserve|h0``.
- loiter duration ``loiter_duration`` and loiter altitude ``loiter|h0``.
  

Phase types
===========
A phase is a building block of a mission profiles.
The phases and relevant classes are implemented in ``openconcept/PATH/mission_profiles.py``.
Users usually don't need to modify these code when creating their own mission profile.

Steady flight
-------------
Class ``SteadyFlightPhase`` can be instantiated for steady climb, cruise, descent, and loiter phases.
For this phase, you need to specify the airspeed (``<phase_name>.fltcond|Ueas``) and vertical speed (``<phase_name>.fltcond|Ueas``) in your runscript.
You may optionally set the duration of the phase (``<phase_name>.duration``), or alternatively, the duration can be set automatically in the mission profile group.

To ensure the steady flight, both vertical and horizontal accelerations will be set to 0.
It first computes the lift coefficient required for zero vertical accelration; CL is then passes to the aircraft model, which returns the lift and drag.
Then, it automatically finds the time history of throttle such that horizontal acceleration is zero.
This is done by solving a system of nonlinear equations (``horizontal acceleration = 0``) w.r.t. throttle using `BalanceComp <https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/balance_comp.html>`_ for each phase.

Balanced-field takeoff
----------------------
Balanced-field takeoff analysis is implemented in the following classes: ``BFLImplicitSolve, GroundRollPhase, RotationPhase, RobustRotationPhase, ClimbAnglePhase``.
Unlike the steady flight phases, the takeoff phases are not steady and acceleration is non-zero.
Therefore, the engine throttle needs to be specified to compute the acceleration, which is 100% by defalut for accelerating phases and 0 for braking.
Users can also set the throttle manually in the runscript.
The acceleration is then integrated to compute the velocity.

VTOL transition
---------------
This is only relevant to VTOL configurations. Maybe move to a different page (like eVTOL mission and phases) to avoid confusion?


