.. _Atmospherics:

************
Atmospherics
************

Most of OpenConcept's atmospheric models use the 1976 Standard Atmosphere.
For more details, see the source documentation for the ``ComputeAtmosphericProperties`` component.

Models for specific atmospheric properties are also available on their own:

    - ``TrueAirspeedComp``
    - ``EquivalentAirpseedComp``
    - ``TemperatureComp``
    - ``SpeedOfSoundComp``
    - ``PressureComp``
    - ``MachNumberComp``
    - ``DynamicPressureComp``
    - ``DensityComp``

The code is adapted from this paper, with permission:

.. code:: bibtex

    @conference{Jasa2018b,
        Address = {Orlando, FL},
        Author = {John P. Jasa and John T. Hwang and Joaquim R. R. A. Martins},
        Booktitle = {2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum},
        Month = {January},
        Title = {Design and Trajectory Optimization of a Morphing Wing Aircraft},
        Year = {2018}
    }
