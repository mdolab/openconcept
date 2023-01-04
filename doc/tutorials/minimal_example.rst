.. _Minimal-example-tutorial:

***************
Minimal Example
***************

This example shows how to set up an OpenConcept aircraft and mission analysis model.
The goal here is to use only what is absolutely necessary with the idea of introducing the starting point for building more complicated and detailed models.
It uses a simplified aircraft model and basic mission profile with climb, cruise, and descent phases.

OpenConcept leans heavily on OpenMDAO for the numerical framework, in particular ``ExplicitComponent``, ``ImplicitComponent``, ``Group``, ``Problem``, the OpenMDAO solvers, and the optimizer interfaces.
If you are not already familiar with OpenMDAO, we strongly recommend going through their `basic user guide <https://openmdao.org/newdocs/versions/latest/basic_user_guide/basic_user_guide.html>`_.

.. note::
  The script described in this tutorial is called minimal.py and can be found in the examples folder.

Imports
=======

.. literalinclude:: ../../openconcept/examples/minimal.py
    :start-after: # rst Imports (beg)
    :end-before: # rst Imports (end)

We start by importing the necessary modules.
In this example, we need OpenMDAO to use its classes and solvers.
From OpenConcept, we use the BasicMission mission analysis component.
Lastly, we use NumPy to initialize vectors and matplotlib to make figures at the end.

Aircraft model
==============

At it's most basic, an OpenConcept aircraft model takes in a lift coefficient and throttle position (from 0 to 1) and returns thrust, weight, and drag.
In the code, these variables are named ``"fltcond|CL"``, ``"throttle"``, ``"thrust"``, ``"weight"``, and ``"drag"``, respectively.
You can add as much or as little detail in this computation of the outputs as you'd like, but all the model cares is that the outputs can be controlled by the inputs.
OpenConcept provides component models to build these aircraft systems.

The complexity can grow rapidly, so for now we will not use any of these OpenConcept models; instead we develop a minimal aircraft model.
We assume constant weight across the whole mission.
Thrust is modeled as maximum thrust times the throttle.
Drag is computed as lift divided by lift-to-drag ratio.

Let's first take a look at the code for the whole aircraft model, and then we'll explain each part.
The whole model looks like this:

.. literalinclude:: ../../openconcept/examples/minimal.py
    :start-after: # rst Aircraft (beg)
    :end-before: # rst Aircraft (end)

Options
-------

.. literalinclude:: ../../openconcept/examples/minimal.py
    :start-after: # rst Options
    :end-before: # rst Setup
    :dedent: 4

We start by defining the options for the model.
These two options are **required** for all OpenConcept aircraft models:

- ``"num_nodes"``: OpenConcept numerically integrates states w.r.t. time in each mission phase. This option will tell the aircraft model how many numerical integration points are used in the phase. All the required inputs and outputs listed above are vectors of length ``"num_nodes"``.
- ``"flight_phase"``: The mission analysis group sets this option to a string that is the name of the current flight phase. For example in the basic three-phase mission, this will be set either to ``"climb"``, ``"cruise"``, or ``"descent"``. This option can be used by the aircraft model to set phase-specific values. For example, in takeoff segments the user may want to set different aerodynamic parameters that correspond to a flaps-extended configuration.

Setup
-----

.. literalinclude:: ../../openconcept/examples/minimal.py
    :start-after: # rst Setup
    :end-before: # rst Compute
    :dedent: 4

Next, we add the inputs and outputs to the aircraft model.
We start with the lift coefficient and throttle---the inputs required by OpenConcept.
Note that the shape of these inputs is defined as the number of nodes.
In other words, these inputs are vectors of length ``"num_nodes"``.

There are other parameters that the mission analysis will automatically connect to the aircraft model if they're defined as inputs.
A set of flight condition variables and other aircraft parameters defined by the user at the top level analysis group.

The available flight condition variables are the following:

.. list-table:: Flight condition variables for steady flight phases
    :header-rows: 1

    * - Variable name
      - Property
      - Vector length
    * - fltcond|CL
      - Lift coefficient
      - ``"num_nodes"``
    * - fltcond|q
      - Dynamic pressure
      - ``"num_nodes"``
    * - fltcond|rho
      - Density
      - ``"num_nodes"``
    * - fltcond|p
      - Pressure
      - ``"num_nodes"``
    * - fltcond|T
      - Temperature (includes increment)
      - ``"num_nodes"``
    * - fltcond|a
      - Speed of sound
      - ``"num_nodes"``
    * - fltcond|TempIncrement
      - Increment on the 1976 Standard Atmosphere temperature
      - ``"num_nodes"``
    * - fltcond|M
      - Mach number
      - ``"num_nodes"``
    * - fltcond|Utrue
      - True airspeed
      - ``"num_nodes"``
    * - fltcond|Ueas
      - Equivalent airspeed
      - ``"num_nodes"``
    * - fltcond|groundspeed
      - Ground speed
      - ``"num_nodes"``
    * - fltcond|vs
      - Vertical speed
      - ``"num_nodes"``
    * - fltcond|h
      - Altitude
      - ``"num_nodes"``
    * - fltcond|h_initial
      - Initial altitude in phase
      - 1
    * - fltcond|h_final
      - Final altitude in phase
      - 1
    * - fltcond|cosgamma
      - Cosine of the flight path angle
      - ``"num_nodes"``
    * - fltcond|singamma
      - Sine of the flight path angle
      - ``"num_nodes"``

The aircraft parameters that the mission analysis passes through to the aircraft model are those set by the user in the top-level group (called ``"MissionAnalysis"`` in our case).
It will pass any variable whose name starts with ``"ac|"``.
Here, we define four aircraft parameters that are used by the aircraft model: ``"ac|geom|wing|S_ref"`` (wing area), ``"ac||weights|TOW"`` (aircraft weight), ``"ac|propulsion|max_thrust"``, and ``"ac|aero|L_over_D"`` (lift-to-drag ratio).
We will see in the mission tutorial section how they are defined at the top level.

Next, we add the outputs needed by OpenConcept to converge the mission: ``"weight"``, ``"drag"``, and ``"thrust"``.

Finally, we declare the derivatives of the outputs w.r.t. the inputs.
In this case, we have OpenMDAO compute all the partial derivatives using complex step.
In practice, analytically defining the partial derivatives offers more accurate and faster derivative computations.

Compute
-------

.. literalinclude:: ../../openconcept/examples/minimal.py
    :start-after: # rst Compute
    :end-before: # rst Aircraft (end)
    :dedent: 4

In this section, we actually compute the output values using the inputs.
The weight is simply equal to the defined weight.
The thrust is equal to the throttle input times the max thrust input.
The drag is equal to the lift divided by L/D, where lift is computed using the dynamic pressure, lift coefficient, and wing area.

Mission
=======

.. literalinclude:: ../../openconcept/examples/minimal.py
    :start-after: # rst Mission (beg)
    :end-before: # rst Mission (end)

Now that we have an aircraft model, we need to define the mission it will fly.
In OpenConcept, we do this by defining a top-level OpenMDAO group.
This group usually contains two components:

- An ``IndepVarComp`` or ``DictIndepVarComp`` with the aircraft parameters (the values are automatically passed to the aircraft model)
- The OpenConcept mission analysis block

In this case we have only four aircraft parameters, so we define them in the script using an OpenMDAO ``IndepVarComp``.
Following tutorials will use a slightly different method to keep the parameters more organized.
These outputs are promoted from the ``IndepVarComp`` to the level of the ``MissionAnalysis`` group (``promote_outputs=["*"]`` promotes all outputs).

Next we add the mission analysis.
In this tutorial, we use OpenConcept's ``BasicMission``, which consists of a climb, a cruise, and a descent flight phase.
The aircraft model class is passed into the ``BasicMission``, along with the number of nodes per phase.
``BasicMission`` will instantiate copies of the aircraft model class in each flight phase and promote all inputs from the aircraft model that begin with ``"ac|"`` to the ``BasicMission`` level.
We promote all inputs that begin with ``"ac|*"`` from the ``BasicMission`` to the ``MissionAnalysis`` level.
This way, OpenMDAO will automatically connect the outputs from the ``IndepVarComp`` to the aircraft parameter inputs of the aircraft models in each phase.

In this group, we declare an option to set the number of nodes per phase so that the user can initialize the value when the problem is set up.

.. note::
    ``"ac|geom|wing|S_ref"`` is a **required** top-level aircraft parameter that must be defined.
    OpenConcept uses it to compute lift coefficient from lift force.
    Which other aircraft parameters are required is dependent on which OpenConcept models the user chooses to include in the aircraft model.

Run script
==========

Setup problem
-------------

Now that we have the necessary models.
The last step before running the model is setting up the OpenMDAO problem and providing the necessary values to define the mission profile.

.. literalinclude:: ../../openconcept/examples/minimal.py
    :start-after: # rst Setup problem (beg)
    :end-before: # rst Setup problem (end)

We initialize an OpenMDAO Problem and add the ``MissionAnalysis`` class we defined as the problem's model.
Here is where we specify the number of nodes per flight phase, using 11.
Next, we add a solver that is used to determine the throttle and lift coefficient values that satisfy the steady flight force balance across the mission.
We use OpenMDAO's Newton solver and assign a direct linear solver to solve each subiteration of the nonlinear solver.

Once the problem is setup, we set the necessary values to specify the mission profile.
``BasicMission`` has climb, cruise, and descent phases, but we still need to tell it the speed each is flown at, the cruise altitude, etc.
This mission requires a vertical speed and airspeed in each phase.
It also requires an initial cruise altitude and total mission length.
For more details, see the :ref:`mission analysis documentation <MissionAnalysis>`.

Run it!
-------

Finally, we actually run the analysis.

.. literalinclude:: ../../openconcept/examples/minimal.py
    :start-after: # rst Run (beg)
    :end-before: # rst Run (end)

After running the model, we do a bit of postprocessing to visualize the results.
The first thing we do is create an N2 diagram.
This allows you to explore the structure of the model and the values of each variable.
Lastly, we get some values from the model and create plot of some values, using matplotlib.

The model should converge in a few iterations.
The plot it generates should look like this:

.. image:: assets/minimal_example_results.svg

The N2 diagram for the model is the following:

.. embed-n2::
  ../openconcept/examples/minimal.py

Summary
=======

In this tutorial, we developed a simple run script to explain how to set up an OpenConcept mission analysis.
The aircraft model is very simplified and does not use any of OpenConcept's models.
Nonetheless, it obeys all the input/output requirements of an OpenConcept aircraft and thus can be used in the mission analysis.

You may notice that the results of this analysis are not particularly useful.
It does not offer any information about fuel burn, energy usage, component temperatures, or sizing requirements.
In the next tutorial, we'll develop a more comprehensive aircraft model that is more useful for conceptual design.

The final script looks like this:

.. literalinclude:: ../../openconcept/examples/minimal.py
