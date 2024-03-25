.. _Weights:

*******
Weights
*******

This module provides empty weight approximations using mostly empirical textbook methods.
Component positions within the aircraft are not considered; all masses are accumulated into a single number.

Conventional jet transport aircraft OEW: ``JetTransportEmptyWeight``
====================================================================

This model combines estimates from :footcite:t:`raymer2006aircraft`, :footcite:t:`roskam2019airplane`, and others to estimate the operating empty weight of a jet transport aircraft.
The model includes two correction factor options: ``structural_fudge`` that multiplies structural weights and another ``total_fudge`` which multiplies the final total weight.
A complete list of the required inputs and outputs can be found in OpenConcept's API documentation, and more details are available in the source code.

This model uses the following components from the `openconcept.weights` module to estimate the total empty weight:

- ``WingWeight_JetTransport``
- ``HstabConst_JetTransport``
- ``HstabWeight_JetTransport``
- ``VstabWeight_JetTransport``
- ``FuselageKws_JetTransport``
- ``FuselageWeight_JetTransport``
- ``MainLandingGearWeight_JetTransport``
- ``NoseLandingGearWeight_JetTransport``
- ``EngineWeight_JetTransport``
- ``EngineSystemsWeight_JetTransport``
- ``NacelleWeight_JetTransport``
- ``FurnishingWeight_JetTransport``
- ``EquipmentWeight_JetTransport``

Blended wing body jet OEW: ``BWBEmptyWeight``
=============================================

This blended wing body empty weight model is a modified version of the ``JetTransportEmptyWeight`` buildup.
It contains the following changes from the conventional configuration jet transport empty weight buildup:

- Separate model for the weight of the pressurized portion of the centerbody for passengers or cargo (``CabinWeight_BWB`` component)
- Separate model for the weight of the unpressurized portion of the centerbody behind the passengers or cargo (``AftbodyWeight_BWB`` component)
- Removed fuselage and tail weights


Single-engine turboprop OEW: ``SingleTurboPropEmptyWeight``
===========================================================

This model combines estimates from :footcite:t:`raymer2006aircraft` and :footcite:t:`roskam2019airplane` to compute the total operating empty weight of a small single-engine turboprop aircraft.
The engine and propeller weight are not computed since OpenConcept's turboshaft and propeller models compute those separately.
Thus, those weights must be provided to this component by the user.

This model uses the following components from the `openconcept.weights` module to estimate the total empty weight:

- ``WingWeight_SmallTurboprop``
- ``EmpennageWeight_SmallTurboprop``
- ``FuselageWeight_SmallTurboprop``
- ``NacelleWeight_SmallSingleTurboprop``
- ``LandingGearWeight_SmallTurboprop``
- ``FuelSystemWeight_SmallTurboprop``
- ``EquipmentWeight_SmallTurboprop``

For turboprops with multiple engines, ``NacelleWeight_MultiTurboprop`` may be used instead of ``NacelleWeight_SmallSingleTurboprop``.

Twin-engine series hybrid OEW: ``TwinSeriesHybridEmptyWeight``
==============================================================

This model uses all the same components as ``SingleTurboPropEmptyWeight``, except it adds weight inputs required by the user to account for the hybrid propulsion system.
The additional weights, which are computed by other OpenConcept components, are electric motor weight and generator weight.

.. footbibliography::
