.. _Integrator-tutorial:

********************
Using the Integrator
********************

In this tutorial, we will build off the previous minimal example tutorial by adding a numerical integrator to compute fuel burn throughout the mission.
This will require the following additions to the previous aircraft model:

- A component to compute fuel flow rate using thrust and thrust-specific fuel consumption (TSFC)
- An integrator to integrate the fuel flow rate w.r.t. time to compute the weight of fuel burned at each numerical integration point
- A component to compute the weight at each integration point by subtracting the fuel burned from the takeoff weight

Other than these changes to the aircraft model the code will look very similar to the minimal example, so we will gloss over some details that are described more in the :ref:`minimal example <Minimal-example-tutorial>`.
If you have not done so already, it is recommended to go through that tutorial first.

Imports
=======

.. literalinclude:: ../../examples/minimal_integrator.py
    :start-after: # rst Imports (beg)
    :end-before: # rst Imports (end)

The notable addition to the imports is OpenConcept's `Integrator` class.
We also import the `Aircraft` class and `setup_problem` function from the previous tutorial.

Aircraft model
==============

The aircraft model is no longer a set of explicit equations; it now requires a combination of OpenMDAO components to compute the weight.
For this reason, the aircraft model is now an OpenMDAO ``Group`` instead of an ``ExplicitComponent``.

Let's first take a look at the code for the entire aircraft model and then we will break down what is happening in each section.

.. literalinclude:: ../../examples/minimal_integrator.py
    :start-after: # rst Aircraft (beg)
    :end-at: # rst Weight (end)

Options
-------

The options are the same as the minimal example tutorial.

.. literalinclude:: ../../examples/minimal_integrator.py
    :start-after: # rst Options
    :end-before: # rst Setup
    :dedent: 4

Setup
-----

.. note::
    The order you add components to OpenMDAO groups (using ``add_subsystem``) matters!
    Generally, it is best to try to add components in the order that achieves as much feed-forward variable passing as possible.
    For example, we have a component that computes thrust and another that takes thrust as an input.
    To make this feed-forward, we add the component that takes thrust as an input *after* the component that computes it.

Thrust and drag from minimal aircraft
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step in setting up the new aircraft model is to add the simplified aircraft to the group.
We still use this model to compute thrust and drag, but the weight calculation will be modified.
For this reason, we promote only the thrust and drag outputs to the new aircraft model group level.
All the inputs are still required, so we promote them all to the group level (this way OpenConcept will automatically connect them as we discussed last time).
If you are confused about the promotion, check out the OpenMDAO documentation.

.. literalinclude:: ../../examples/minimal_integrator.py
    :start-after: # rst Simple aircraft (beg)
    :end-before: # rst Simple aircraft (end)
    :dedent: 8

Fuel flow rate
~~~~~~~~~~~~~~

Next, we need to compute the fuel flow rate to pass to the integrator.
Since this is a simple function of thrust and TSFC, we use an OpenMDAO ``ExecComp`` (the OpenMDAO docs are very thorough if you are confused about the syntax).
We give it the fuel flow equation we want it to evaluate and define the units and shape of each parameter.
Notice that fuel flow and thrust are both vectors because they are evaluated at each numerical integration point and will change throughout each flight phase.
The TSFC is a scalar because it is a single constant parameter defined for the aircraft.
Finally, we promote the inputs.
Thrust is automatically connected to the thrust output from the minimal aircraft model.
TSFC is promoted to a name beginning with ``"ac|"`` so that the mission analysis promotes the variable to the top level so we can set it the same way as the other aircraft parameters.

.. literalinclude:: ../../examples/minimal_integrator.py
    :start-after: # rst Fuel flow (beg)
    :end-before: # rst Fuel flow (end)
    :dedent: 8

Integrator
~~~~~~~~~~

Now we are ready to add the integration.
This is done by adding an OpenConcept ``Integrator`` component to the model.
After adding the integrator, we add an integrated variable and associated variable to integrate using the integrator's ``add_integrand`` method.
Let's step through all the details of these calls---there's a lot to unpack.

.. literalinclude:: ../../examples/minimal_integrator.py
    :start-after: # rst Integrator (beg)
    :end-before: # rst Integrator (end)
    :dedent: 8

When ``Integrator`` is initialized, there are a few important options that must be set.
As we've seen before, we set ``num_nodes`` to tell it how many integration points to use.

``diff_units`` are the units of the differential.
For example, in our equation we are computing

.. math::
    \text{fuel burn} = \int_{t_\text{initial}}^{t_\text{final}} \dot{m}_\text{fuel} \: dt

The differential is :math:`dt` and has units of time (we'll use seconds here).

The ``time_setup`` option sets what information the integrator uses to figure out the time at each integration point.
The options are ``"dt"duration"``, or ``"bounds"``.

- ``"dt"`` creates an input called ``"dt"`` that specifies the time spacing between each numerical integration point
- ``"duration"`` creates an input called ``"duration"`` that specifies the total time of the phase. The time between each integration point is computed by dividing the duration by the number of time steps (number of nodes minus one). This is the most common choice for the time setup and has the advantage that OpenConcept automatically connects the ``"duration"`` input to the mission-level duration, so there is no manual time connection needed.
- ``"bounds"`` creates inputs called ``"t_initial"`` and ``"t_final"`` that specify the initial and final time of the segment. This internally computes duration and then time is computed the same was as for the duration approach.

The final option is the integration scheme.
The two options are ``"bdf3"`` and ``"simpson"``.
``"bdf3"`` uses the third-order-accurate BDF3 integration scheme.
``"simpson"`` uses Simpson's rule.
Simpson's rule is the most common choice for use in OpenConcept.

In the next line we add information about the quantity we want to integrate.
We first define the name of the integrated quantity: ``"fuel_burned"``.
This will become an output of the integrator (accessed in this case as ``"fuel_integrator.fuel_burned"``).
We then define the rate we want integrated: ``"fuel_flow"``.

Mission
=======

Run script
==========

Run it!
-------

The model should converge in a few iterations.
You can see the N2 diagram for the model :download:`here <assets/minimal_integrator_n2.html>`.
The plot it generates should look like this:

.. image:: assets/minimal_integrator_results.svg

You can see that the weight is no longer constant.
This results in a varying throttle in the cruise segment, unlike the constant throttle from the :ref:`minimal example <Minimal-example-tutorial>`.
Also notice that the fuel flow and throttle have the exact same shape, which makes sense because they are directly related by a factor of TSFC.

Summary
=======
