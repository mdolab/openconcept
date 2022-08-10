.. _Weights:

*******
Weights
*******

This module provides empty weight approximations using mostly empirical textbook methods.
For now there are only models for small turboprop-sized aircraft, but more may be added in the future.
Component positions within the aircraft are not considered; all masses are accumulated into a single number.

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
