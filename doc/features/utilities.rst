.. _Utilities:

*********
Utilities
*********

OpenConcept provides utilities for math and other generally-useful operations.

Math
====

Integration: ``Integrator``
---------------------------

This integrator is perhaps the most important part of the OpenConcept interface because it is critical for mission analysis, energy usage modeling, and unsteady thermal models.
It can use the BDF3 integration scheme or Simpson's rule to perform integration.
By default it uses BDF3, but Simpson's rule is the most common one used by OpenConcept.

For a description of how to use it, see the :ref:`integrator tutorial <Integrator-tutorial>`.

Addition and subtraction: ``AddSubtractComp``
---------------------------------------------

This component can add/subtract a combination of vectors and scalars (any vector in the equation must be the same length).
Scaling factors on each input can be defined to switch between addition and subtraction (and other scaling factors if desired).

Multiplication and division: ``ElementMultiplyDivideComp``
----------------------------------------------------------

Similar to the ``AddSubtractComp``, but instead of scaling factors you specify whether each input is divided (by default multiplied).

Vector manipulation
-------------------

``VectorConcatenateComp``
~~~~~~~~~~~~~~~~~~~~~~~~~

Concatenates one or more sets of more than one vector into one or more output vectors.

``VectorSplitComp``
~~~~~~~~~~~~~~~~~~~

Splits one or more vectors into one or more sets of 2+ vectors.

Differentiation:  ``FirstDerivative``
-------------------------------------

Differentiates a vector using a second or fourth order finite difference approximation.

Maximum: ``MaxComp``
--------------------

Returns the maximum value of a vector input.

Minimum: ``MinComp``
--------------------

Returns the minimum value of a vector input.

General
=======

Outputs from dictionary: ``DictIndepVarComp``
---------------------------------------------

Creates a component with outputs defined by keys in a nested dictionary.
The values of each output are taken from the dictionary.
Each variable from the dictionary must be added by the user with ``add_output_from_dict``.
This component is based on OpenMDAO's ``IndepVarComp``.

Linear interpolation: ``LinearInterpolator``
--------------------------------------------

Creates a vector that linearly interpolates between an initial and final value.

Rename variables: ``DVLabel``
-----------------------------

Helper component that is needed when variables must be passed directly from input to output of an element with no other component in between.

This component is adapted from Justin Gray's pyCycle software.

Select elements from vector: ``SelectorComp``
---------------------------------------------

Given a set of vector inputs, this component allows the user to specify which input each spot in the vector output pulls from.

Dymos parameters from dictionary: ``DymosDesignParamsFromDict``
---------------------------------------------------------------

Creates Dymos parameters from an external file with a Python dictionary.

Visulization
============

``plot_trajectory``
-------------------

Plot data from a mission.

``plot_trajectory_grid``
------------------------

Plot data from multiple missions against each other.
