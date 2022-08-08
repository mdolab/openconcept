.. _Propulsion:

**********
Propulsion
**********

Available propulsion systems
============================
User can build their own propulsion systems following the format of the example systems listed here.
For details of each propulsion systems, see the source docs.
TODO: link to source docs here

All-electric propulsion
-----------------------
This is an electric propulsion system consisting of a constant-speed propeller, motor, and battery.
In addition, this models a thermal management system using a compressible or incompressible 1D duct with heat exchanger.

This model takes the motor throttle as a control input and computes the thrust and battery state of charge (SOC).
TODO: what are the outputs related to thermal management??

Turboprop
---------
This is a simple turboprop system consisting of constant-speed propeller(s) and turboshaft(s).
``TurbopropPropulsionSystem`` implements a system of one propeller and one turboshaft.
Also, ``TwinTurbopropPropulsionSystem`` implements a twin system that has two propellers and turboshafts.
Users can create their own turboprop model by changing the parameters, e.g. engine rating and propeller diameter.

This model takes the engine throttle as a control input and computes the thrust and fuel flow.

Series-hybrid electric propulsion
---------------------------------
In series-hybrid propulsion systems, motor(s) draws electrical load from both battery and turboshaft generator.
The control inputs are the motor throttle, turboshaft throttle, and the power split fraction between the battery and generator;
The turboshaft throttle must be driven by an implicit solver or optimizer to equate the generator's electric power output and the engine power required by the splitter.
Given these inputs, the model the computes the thrust, fuel flow, and electric load.

OpenConcept implements both single and twin series-hybrid electric propulsion systems in ``simple_series_hybrid.py``.
The systems with thermal management components are also implemented in ``thermal_series_hybrid.py``.

Turbofan
--------
OpenConcept implements two turbofan engine models, CFM56 and a geared turbofan (GTF) for NASA's N+3 aircraft.
Both are the surrogate models of `pyCycle <https://github.com/OpenMDAO/pyCycle>`_, a thermodynamic cycle modeling tool builed on the OpenMDAO framework.
The inputs to the turbofan models are the engine throttle and flight conditions (Mach number and altitude), and outputs are the thrust and fuel flow.
In addition, the CFM56 model outputs the turbine inlet temperature, and the N+3 model outputs the surge margin.

We also implement a N+3 engine with hybridization, which requires hybrid shaft power as an additional input.