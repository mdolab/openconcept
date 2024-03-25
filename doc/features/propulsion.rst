.. _Propulsion:

**********
Propulsion
**********

Propulsion systems
==================
User can build their own propulsion systems following the format of the example systems listed here.
For details of each propulsion systems, see the :ref:`source docs <source_documentation>`.

All-electric propulsion
-----------------------
This is an electric propulsion system consisting of a constant-speed propeller, motor, and battery.
In addition, this models a thermal management system using a compressible (``AllElectricSinglePropulsionSystemWithThermal_Compressible``) or incompressible (``AllElectricSinglePropulsionSystemWithThermal_Incompressible``) 1D duct with heat exchanger.

This model takes the motor throttle as a control input and computes the thrust and battery state of charge (SOC).

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
``TwinSeriesHybridElectricPropulsionSystem`` is the recommended one to use.
Others require the user to explicitly set up additional components to generate feasible analysis results (see the comments in the code).

The systems with thermal management components are also implemented in ``thermal_series_hybrid.py``.

Turbofan
--------
OpenConcept implements two turbofan engine models, CFM56 and a geared turbofan (GTF) for NASA's N+3 aircraft.
Both are surrogate models derived from `pyCycle <https://github.com/OpenMDAO/pyCycle>`__, a thermodynamic cycle modeling tool built on the OpenMDAO framework.
The inputs to the turbofan models are the engine throttle and flight conditions (Mach number and altitude), and outputs are the thrust and fuel flow.
In addition, the CFM56 model outputs the turbine inlet temperature, and the N+3 model outputs the surge margin.

We also implement a N+3 engine with hybridization, which requires hybrid shaft power as an additional input.

Models
======

The propulsion systems are made up of individual propulsion components.
Available individual models are listed here.

Electric motor: ``SimpleMotor``
-------------------------------

An electric motor model that computes shaft power by multiplying throttle by the motor's electrical power rating and efficiency.
The electrical power that does not go toward shaft power is modeled as heat (see the ``LiquidCooledMotor`` to add thermal management to this motor model).
Weight and cost are linear functions of the electric power rating.

Turboshaft: ``SimpleTurboshaft``
--------------------------------

Computes shaft power by multiplying throttle by the engine's rated shaft power.
The fuel flow is computed by multiplying the generated shaft power by a provided power-specific fuel consumption.
As with the electric motor, cost and weight are modeled as linear functions of the power rating.

Propeller: ``SimplePropeller``
------------------------------

This model uses an empirical propeller efficiency map for a constant speed turboprop propeller under the hood.
For low speed, it uses a static thrust coefficient map from :footcite:t:`raymer2006aircraft`.
Propeller maps for three and four bladed propellers are included.

Generator: ``SimpleGenerator``
------------------------------

This model uses essentially the same model as ``SimpleMotor`` but in reverse.
It takes in a shaft power and computes electrical power and heat generated.

Power splitter: ``PowerSplit``
------------------------------

This component enables electrical or mechanical shaft power to be split to two components.
It uses either a fractional or fixed split method where fractional splits the input power by a fraction (set by an input) and fixed sends a specified amount of power (set by an input) to one of the outputs.
The efficiency can be changed from the default of 100%, which results in some heat being generated.
Cost and weight are modeled as linear functions of the power rating.

Rubber turbofan: ``RubberizedTurbofan``
---------------------------------------

This model enables the pyCycle-based CFM56 and N+3 turbofan models to be scaled to different thrust ratings.
The scaling is done by multiplying thrust and fuel flow by the same value.
The model also has an option to use hydrogen, which scales the fuel flow to maintain the same thrust-specific energy consumption (see :footcite:t:`Adler2023` for a definition) as the kerosene-powered CFM56 and N+3 engine models.

.. footbibliography::
