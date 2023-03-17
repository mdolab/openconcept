.. _Aerodynamics:

************
Aerodynamics
************

OpenConcept's aerodynamics components provide an estimate of drag as a function of lift coefficient, flight conditions, and other parameters.

Simple drag polar: ``PolarDrag``
================================

``PolarDrag`` is the most basic aerodynamic model and uses a parabolic drag polar.
This component computes the drag force given the flight condition (dynamic pressure and lift coefficient) at each node along the mission profile.
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
This enables a more detailed parameterization of the aircraft wing.
OpenAeroStruct implements the vortex-lattice method (VLM) for aerodynamics and beam-based finite element method (FEM) for structures (in the case of the aerostructural drag polar).
For more details, please check the `documentation <https://mdolab-openaerostruct.readthedocs-hosted.com/en/latest/>`_.

The aerodynamic-only model supports three types of mesh generation for planform and geometry flexibility.
The aerostructural model is currently limited to simple planform geometries.
Additionally, the wing does not include a horizontal tail to trim it.

OpenConcept uses a surrogate model trained by OpenAeroStruct analyses to reduce the computational cost.
The data generation and surrogate training is automated; specifying the training grid manually may improve accuracy and decrease computational cost.

VLM-based aerodynamic model: ``VLMDragPolar``
------------------------------------------------
This model uses the vortex-lattice method (VLM) to compute the drag.
The inputs to this model are the flight conditions (Mach number, altitude, dynamic pressure, lift coefficient) and aircraft design parameters.

The aerodynamic mesh can be defined in one of three ways:

1. Use simple planform variables to define a trapezoidal planform. These planform variables are wing area, aspect ratio, taper ratio, and quarter chord sweep angle.

2. Define multiple spanwise wing sections between which a mesh is linearly interpolated. Each section is defined by its streamwise offset, chord length, and spanwise position. The whole planform is then scaled uniformly to match the desired wing area.

3. Directly provide an OpenAeroStruct-compatible mesh.

More details on the inputs, outputs, and options are available in the source code documentation.

Aerostructural model: ``AeroStructDragPolar``
-----------------------------------------------------
This model is similar to the VLM-based aerodynamic model, but it performs aerostructural analysis (that couples VLM and structural FEM) instead of aerodynamic analysis (just FEM).
This means that we now consider the wing deformation due to aerodynamic loads, which is important for high aspect ratio wings.
The structural model does not include point loads (e.g., for the engine) or distributed fuel loads.

The additional input variables users need to set in the run script are listed below.
Like the ``num_twist`` option, you may need to set ``num_toverc``, ``num_skin``, and ``num_spar``.

.. list-table:: Additional design variables for aerostructural model
    :widths: 30 50 20
    :header-rows: 1

    * - Variable name
      - Property
      - Type
    * - ac|geom|wing|toverc
      - Spanwise distribution of thickness to chord ratio.
      - 1D ndarray, lendth ``num_toverc``
    * - ac|geom|wing|skin_thickness
      - Spanwise distribution of skin thickness.
      - 1D ndarray, lendth ``num_skin``
    * - ac|geom|wing|spar_thickness
      - Spanwise distribution of spar thickness.
      - 1D ndarray, lendth ``num_spar``

In addition to the drag, the aerostructural model also outputs the structural failure indicator (``failure``) and wing weight (``ac|weights|W_wing``).
The `failure` variable must be negative (``failure <= 0``) to constrain wingbox stresses to be less than the yield stress.

Understanding the surrogate modeling
------------------------------------

OpenConcept uses surrogate models based on OpenAeroStruct analyses to reduce the computational cost for mission analysis.
The surrogate models are trained in the 3D input space of Mach number, angle of attack, and altitude.
The outputs of the surrogate models are CL and CD (and failure for ``AeroStructDragPolar``).

For more details about the surrogate models, see our `paper <https://mdolab.engin.umich.edu/bibliography/Adler2022d>`_.

Other models
============

The aerodynamics module also includes a couple components that may be useful:

  - ``StallSpeed``, which uses :math:`C_{L, \text{max}}`, aircraft weight, and wing area to compute the stall speed
  - ``Lift``, which computes lift force using lift coefficient, wing area, and dynamic pressure
