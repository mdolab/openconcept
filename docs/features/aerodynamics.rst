.. _Aerodynamics:

************
Aerodynamics
************

This document has some stuff in it.
Can be used 

Simple drag polar: ``PolarDrag``
================================
The class ``PolarDrag`` is the most basic aerodynamic model that uses the drag polar model.
This component computes the drag force given the flight condition (to be specific, dynamic pressure and lift coefficient) at each node along the mission profile.
In run script, users should set the values for the following aircraft design parameters, or declare them as design variables.

.. list-table:: Aircraft design variables for drag polar model
    :header-rows: 1

    * - Variable name
      - Property
    * - ac|geom|wing|S_ref
      - Reference wing area
    * - ac|geom|wing|AR
      - Wing aspect ratio
    * - e
      - Wing Oswald efficiency
    * - CD0
      - Zero-lift drag coefficient


Using OpenAeroStruct
====================
Instead of the simple drag polar model, you can use `OpenAeroStruct <https://github.com/mdolab/OpenAeroStruct>`_ to compute the drag.
This allows you to parametrize the aircraft wing design in more details and explore the effects of wing geometry, including taper, sweep, and twist.
OpenAeroStruct implements the vortex-lattice method (VLM) for aerodynamics and beam-based finite element method (FEM) for structure.
For more detail, please check the `documentation <https://mdolab-openaerostruct.readthedocs-hosted.com/en/latest/>`_.

OpenConcept also uses a surrogate model trained by OpenAeroStruct analyses to reduce the computational cost.
The data generation and surrogate training is automated, thus the users don't need to do anything about it by themselves.

VLM-based aerodynamic model: ``VLMDragPolar``
------------------------------------------------
This model uses the vortex-lattice method (VLM) and a surrogate model to compute the drag.
The inputs to this model are the flight conditions (Mach number, altitude, dynamic pressure, lift coefficient) and aircraft design parameters.

Users should set the following design parameters and options in the run script.

.. list-table:: Aircraft design variables for VLM-based model
    :header-rows: 1

    * - Variable name
      - Property
      - Type
    * - ac|geom|wing|S_ref
      - Reference wing area
      - float
    * - ac|geom|wing|AR
      - Wing aspect ratio
      - float
    * - ac|geom|wing|taper
      - Taper ratio
      - float
    * - ac|geom|wing|c4sweep
      - Sweep angle at quarter chord
      - float
    * - ac|geom|wing|twist
      - Spanwise distribution of twist, from wint tip to root.
      - 1d ndarray, lendth ``num_twist``
    * - ac|aero|CD_nonwing
      - Drag coefficient of components other than the wing; e.g. fuselage,
        tail, interference drag, etc.
      - float
    * - fltcond|TempIncrement
      - Temperature increment for non-standard day
      - float

.. list-table:: Options for VLM-based model
    :widths: 30 50 20
    :header-rows: 1

    * - Variable name
      - Property
      - Type
    * - ``num_x``
      - VLM mesh size (number of vertices) in chordwise direction.
      - int
    * - ``num_y``
      - VLM mesh size (number of vertices) in spanwise direction.
      - int
    * - ``num_twist``
      - Number of spanwise control points for twist distribution.
      - int

There are other advanced options, e.g., the surrogate training points in Mach-alpha-altitude space.
The default settings should work fine for these advanced options, but if you want to make changes, please refer to the source docs.

Aerostructural model: ``AeroStructDragPolar``
-----------------------------------------------------
This model is similar to the VLM-based aerodynamic model, but it performs aerostructural analysis (that coupled VLM and structural FEM) instead of aerodynamic analysis (just FEM).
This means that we now consider the wing deformation due to aerodynamic loads, which is important for high aspect ratio wings.

The additional input variables users need to set in the run script are listed below.
Like the ``num_twist`` option, You may need to set the options ``num_toverc, num_skin, num_spar``.

.. list-table:: Additional design variables for aerostructural model
    :widths: 30 50 20
    :header-rows: 1

    * - Variable name
      - Property
      - Type
    * - ac|geom|wing|toverc
      - Spanwise distribution of thickness to chord ratio.
      - 1d ndarray, lendth ``num_toverc``
    * - ac|geom|wing|skin_thickness
      - Spanwise distribution of skin thickness.
      - 1d ndarray, lendth ``num_skin``
    * - ac|geom|wing|spar_thickness
      - Spanwise distribution of spar thickness.
      - 1d ndarray, lendth ``num_spar``

In addition to the drag, the aerostructural model also outputs the structural failure indicator (``failure``) and wing weight (``ac|weights|W_wing``).
The `failure` variable must be negative, i.e., ``failure <= 0`` to constrain wingboxes stresses to be less than yield stress.

Understanding the surrogate modeling
------------------------------------

OpenConcept uses surrogate models based on the OpenAeroStruct to reduce the computational cost for mission analysis.
The surrogate models are trained in the 3D input space of Mach number, angle of attack, and altitude.
The outputs of the surrogate models are CL, CD, and failure (only for ``AeroStructDragPolar``).

For more details about the surrogate models, see our `paper <https://www.researchgate.net/publication/357559489_Aerostructural_wing_design_optimization_considering_full_mission_analysis>`_.

Other models
============

The aerodynamics module also includes a couple components that may be useful to be aware of:

  - ``StallSpeed``, which uses :math:`C_{L, \text{max}}`, aircraft weight, and wing area to compute the stall speed
  - ``Lift``, which computes lift force using lift coefficient, wing area, and dynamic pressure
