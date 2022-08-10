.. _Thermal:

*******
Thermal
*******

OpenConcept includes a range of thermal components to model thermal management systems, primarily for electrified propulsion architectures.
Most of these components are used in example aircraft.
The ``N3_HybridSingleAisle_Refrig.py`` example uses most of these components, so it's a good place to look for example usage.
The propulsion system in ``HybridTwin_thermal.py`` uses other ones.

Liquid-cooled components
========================

These group together basic thermal models to create components that model the effect of heat being added to some thermal mass (or it's massless) and then dumped into a liquid coolant loop.

Generic: ``LiquidCooledComp``
-----------------------------

This is a generic liquid cooled component.
Given some heat generated, it models the accumulation of the heat within some thermal mass (in mass mode), which is then dumped into a coolant stream using a ``ConstantSurfaceTemperatureColdPlate_NTU``.
In massless mode, the temperature of the component is set such that the amount of heat it generates is the same as the amount of heat entering the coolant.

Battery: ``LiquidCooledBattery``
--------------------------------

This component performs a similar function to the ``LiquidCooledComp`` but it introduces a model of heat accumulation and heat transfer to coolant that is specific to a bandolier-style battery cooling system.
It also enables tracking of battery core and surface temperatures.
The thermal model is from the ``BandolierCoolingSystem``.

Motor: ``LiquidCooledMotor``
----------------------------

This component performs a similar function to the ``LiquidCooledComp`` but it introduces a model of heat accumulation and heat transfer to coolant that is specific to an electric motor with a liquid cooling jacket around it.
The thermal model is from the ``MotorCoolingJacket``.

Refrigerator: ``HeatPumpWithIntegratedCoolantLoop``
===================================================

This models a refrigerator (a.k.a. chiller) that is connected to two coolant loops: one on the hot side and one on the cold side.
Coolant in the loop on the cold side is chilled.
Heat extracted from cold side coolant (and additional heat due to inefficiency) is added to coolant on the hot side loop.
The model also enables the use of a bypass, which connects the cold and hot side loops around the refrigerator.

Ducts
=====

The heat generated onboard needs to be dumped to the atmosphere somehow.
In many cases, this is done using a heat exchanger in a duct.
Ambient air flows through the duct and extracts heat from the heat exchanger.
These ducts are designed to model the effect on drag of adding a ducted heat exchanger to an aircraft.

``ImplicitCompressibleDuct``
----------------------------

This component models a ducted heat exchanger with compressible flow assumptions, which means the effects of heat addition are captured.
The ``ImplicitCompressibleDuct`` includes a heat exchanger (``HXGroup``) within the model.
If you'd like to define your own heat exchanger outside the duct, use the (``ImplicitCompressibleDuct_ExternalHX``).

``ImplicitCompressibleDuct_ExternalHX``
---------------------------------------

The same as the ``ImplicitCompressibleDuct`` but without a heat exchanger within the model.
Details on how to incorporate the heat exchanger can be found in comments in the code.

``ExplicitIncompressibleDuct``
------------------------------

This duct models a ducted heat exchanger at incompressible flow conditions.
Because it cannot model flow with heat addition, it is generally a conservative estimate of the cooling drag.
It assumes that the static pressure at the duct's exist is the ambient pressure, which may or may not be a reasonable assumption.

.. note::
    Given the limitations of the duct, it is recommended to use one of the compressible duct models instead.

Heat exchanger: ``HXGroup``
===========================

A detailed liquid-air compact heat exchanger model.
For more details on the model, see the source code in ``openconcept/thermal/heat_exchanger.py``.

Heat pipe: ``HeatPipe``
=======================

Models an ammonia heat pipe.
A heat pipe is a device that takes advantage of evaporation and condensation of a working fluid to move heat passively and rapidly.
The model may be inaccurate for temperatures outside the -75 to 100 deg C range because the ammonia properties interpolate data.
To change the working fluid, a new surrogate model of fluid properties would be required.

Coolant pump: ``SimplePump``
============================

Coolant pump model that computes the required electrical load given flow properties and a required pressure rise.
The pressure drop required is accumulated from hoses and the heat exchanger.
Weight is a linear function of the rated power.
The component sizing margin is computed as the electrical load required divided by the power rating.

Coolant hose: ``SimpleHose``
============================

A hose to transport coolant.
The purpose of modeling the hoses on the aircraft level is that it computes a weight added to the aircraft and the pressure drop to size the pumps.

Manifolds
=========

These components are used to split and combine coolant flows.

``FlowSplit``
-------------

This component splits coolant mass flow based on a fractional parameter.

``FlowCombine``
---------------

This component combines to coolant flow streams.
It includes equations to compute the output coolant temperature of the combined input flows.

Heat transfer to coolant
========================

At some point in the liquid-cooled thermal management system, heat must be transferred to the coolant.
These components provide general ways of doing this.
Other models provide-component specific methods for this, such as the ``LiquidCooledBattery`` and ``LiquidCooledMotor``.

``PerfectHeatTransferComp``
---------------------------

This component assumes that all heat enters the coolant with no thermal resistance.

``ConstantSurfaceTemperatureColdPlate_NTU``
-------------------------------------------

This component models a microchannel cold plate with a uniform temperature.
Unlike the ``PerfectHeatTransferComp``, it computes heat entering the coolant using some thermal resistance computed based on the plate geometry.

Coolant reservoir: ``CoolantReservoir``
=======================================

This component models a reservoir of coolant that can be used to buffer transient temperature changes by adding thermal mass to the system.

Thermal component residuals
===========================

For modelling temperatures of components, OpenConcept methods can be separated in two categories.
The first assumes that the component has mass, which means it can accumulate heat so the heat added to it may not be the same as the heat removed from it by the cooling system.
The second assumes the component is massless, which means that the heat added to it equals the heat removed.

The first category can more accurately model the temperature, particularly when the component has significant mass and the conditions/heat flows change substantially in time.
Deciding when it is important to model the mass requires engineering judgement, but if you can afford the added complexity it is usually a good choice to model mass.

At the most basic level, we need some sort of base equation to solve for each of these two cases.
This is the function these components provide.

Both of these are used in ``LiquidCooledComp``, so look there for example usage.

``ThermalComponentWithMass``
----------------------------

When thermal mass is considered, the base equation we need is one that defines the rate of change of temperature of the component for a given amount of heat added and removed.
An integrator should then be attached to integrate the temperature rate of change, computing component temperature.

``ThermalComponentMassless``
----------------------------

When mass is ignored, we use a different structure.
We figure out the temperature of the component that results in the heat added being equal to the heat removed.
This becomes an implicit relationship, so ``ThermalComponentMassless`` computes net heat flow that will be driven to zero by the solver.
The temperature it outputs should be used in the heat addition/removal models (e.g., the heat flow to the coolant depends on the temperature of the component and the coolant).
