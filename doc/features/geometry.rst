.. _Geometry:

********
Geometry
********

This module includes tools for computing useful geometry quantities.

Wing geometry
=============

``WingMACTrapezoidal``
----------------------
This component computes the mean aerodynamic chord of a trapezoidal wing planform (defined by area, aspect ratio, and taper).

``WingSpan``
------------
This component computes the span of an arbitrary wing as :math:`\sqrt{S_{ref} AR}`, where :math:`S_{ref}` is the wing planform area and :math:`AR` is the wing aspect ratio.

``WingAspectRatio``
-------------------
This component computes the aspect ratio of an arbitrary wing as :math:`b^2 / S_{ref}` where :math:`b` is the wing span and :math:`S_{ref}` is the wing planform area.

``WingSweepFromSections``
-------------------------
This component computes the average quarter chord sweep angle of a wing defined in linearly-varying piecewise sections in the spanwise direction.
The average sweep angle is weighted by section areas.

``WingAreaFromSections``
-------------------------
This component computes the planform area of a wing defined in linearly-varying piecewise sections in the spanwise direction.

.. warning::
    If you are using this component in conjunction with ``SectionPlanformMesh`` and the inputs you are passing to this component are the same as those passed to ``SectionPlanformMesh``, ensure that you have set the ``scale_area`` option in ``SectionPlanformMesh`` to ``False``.
    Otherwise, the resulting wing area will be off by a factor.

``WingMACFromSections``
------------------------
This component computes the mean aerodynamic chord of a wing defined in linearly-varying piecewise sections in the spanwise direction.
It returns both the mean aerodynamic chord and the longitudinal position of the mean aerodynamic chord's quarter chord.

Other quantities
================

``CylinderSurfaceArea``
-----------------------
Computes the surface area of a cylinder, which can be used to estimate, for example, the wetted area of a fuselage or engine nacelle.
