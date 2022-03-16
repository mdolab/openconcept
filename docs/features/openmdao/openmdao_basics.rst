.. _OpenMDAOBasics:

***************
OpenMDAO Basics
***************

NASA Glenn's `OpenMDAO <http://openmdao.org/>`_ is the framework on which OpenConcept is built.
It is a Python-based environment for modeling, simulation, and optimization which handles derivatives efficiently.
This greatly reduces the computational cost of solving nonlinear equations and performing gradient-based optimization.

Users unfamiliar with OpenMDAO are encouraged to visit the `getting started <http://openmdao.org/twodocs/versions/latest/getting_started/index.html>`_ page on their doc site.
I will cover several portions of the OpenMDAO package that are used extensively in OpenConcept.

Component, Group, and Connections
---------------------------------
OpenMDAO models are generally made up of numerous ``Component`` instances.
Components can be combined into a ``Group`` of components.
Each component (with some exceptions) has *inputs* and *outputs*.
When outputs of one component are connected to inputs of another, components can be chained together and more complex models are formed.

The following simple model calculates the drag on an airplane using a simple drag polar formulation.

Pay attention to these four class methods which are very common in OpenMDAO models:
    - The ``initialize`` method allows the user to define options which are instantiated with the component in a larger model.
    - The ``setup`` method declares inputs, outputs, and the nature of the derivatives of outputs with respect to inputs (this is important for reasons that will be addressed later).
    - The ``compute`` method establishes how the outputs should be computed.
    - The ``compute_partials`` method tells the component how to compute partial derivatives.

.. literalinclude:: /../openconcept/analysis/aerodynamics.py
    :pyobject: PolarDrag
    :language: python

Connections in OpenMDAO can be `defined explicitly <http://openmdao.org/twodocs/versions/latest/features/core_features/grouping_components/connect.html>`_, or `semi-implicitly through variable promotion <http://openmdao.org/twodocs/versions/latest/features/core_features/grouping_components/add_subsystem.html>`_.
Refer to the OpenMDAO docs for more details.

In general, I use variable promotion if I have some high-level design parameters which need to be propagated down to many different components (e.g. wing area).
I use structured, unambiguous variable names so I can find and replace them if I need to refactor the code.
Explicit connections are useful down at lower levels in the model where only one or two connections need to be made and it's unlikely that end users will edit the model.

The Problem Class
-----------------
To perform analysis and/or optimization, OpenMDAO requires a ``Problem`` object to be instantiated.
Examples of problems representative of aircraft design can be found in the OpenConcept ``examples/`` folder.

The ``Problem`` object needs some attributes to be set in order to work properly.
    - ``problem.model`` is an OpenMDAO ``Group`` containing all the necessary ``Component`` or ``Group`` objects to model the problem. It is the problem physics.
    - ``problem.driver`` is the optimization algorithm (required in order to do MDO). A common choice of driver is the ``ScipyOptimizeDriver``.
    - ``problem.nonlinear_solver`` is required for problems which have implicit state variables. OpenMDAO's ``NewtonSolver`` is amazingly efficient at solving problems with accurate derivatives.
    - ``problem.linear_solver`` is required when Newton methods are used for nonlinear systems, since a linear solve is a necssary component in computing the Newton step.

If optimization is being done, the following methods must be employed:
    - ``problem.model.add_design_var()`` method tells the optimizer which variables can be altered during the optimization.
    - ``problem.model.add_objective()`` method sets the objective function for the optimization (the variable to be minimized/maximized).
    - ``problem.model.add_constraint()`` method adds constraints (since most MDO problems are constrained in some way)

Finally, the ``problem.setup()`` method is run once the model, settings, and optimization are all defined. This is required for OpenMDAO to work properly.

Setting and Accessing Values
----------------------------
If variable values need to be set (for example, initial design variable values), they can be accessed (read and written) like:

.. code-block:: python

    problem['mycomponent.myvarname'] = 30

Running the Model/Optimization
------------------------------
To run an analysis-only problem, the ``problem.run_model()`` method will execute all the components.
If optimization is being conducted, the ``problem.run_driver()`` method must be used instead.

Partial Derivatives
-------------------
OpenMDAO uses the *Modular Analysis and Unified Derivatives (MAUD)* formulation to compute efficient derivatives.
Both gradient-based optimization and Newton nonlinear solutions require *total derivatives* of the first input variables with respect to the last output variables.
In complex models, there may be a series of complicated mathematical expressions in between (intermediate states).
OpenMDAO uses the *partial* derivatives of each component and assembles them together efficiently to obtain total derivatives (see OpenMDAO `doc page <http://openmdao.org/twodocs/versions/latest/theory_manual/total_derivs/total_derivs_theory.html>`_).

If you use the pre-built OpenConcept components, you don't need to worry about this at all. Enjoy the efficient and accurate solutions.

If you build your own, custom components, you need to make sure that you're supplying accurate partial derivatives.
This usually means either ensuring your model can handle complex input and output variables (to use the complex step method), or supplying analytic derivatives to each component using the ``compute_partials`` method.
If you do this, you should make sure to use OpenMDAO's ``check_partials`` method regularly.

**Errors in partial derivatives can cause MAJOR Newton and optimizer convergence issues which can be difficult to debug.**

Recording and Retrieving Results
--------------------------------
OpenMDAO includes a SQLlite database interface for recording the results of model/optimization runs.
First, instantiate an ``openmdao.api.SqliteRecorder`` object. Then attach the object to the problem.model object, like this:

.. code-block:: python

    recorder = SqliteRecorder(filename_to_save)
    problem.model.add_recorder(recorder)

Command Line Tools
------------------
OpenMDAO provides `command line utilities <http://openmdao.org/twodocs/versions/latest/features/debugging/om_command.html>`_ to make sure your models are configured correctly.
While the utility has many uses (see the OpenMDAO docs), every user should make a habit of running `openmdao check myscript.py` and ensuring that no inputs are left unconnected.

Putting it all Together
-----------------------
The ``examples/TBM850.py`` `script <https://github.com/mdolab/openconcept/blob/main/examples/TBM850.py>`_ models a single-engine turboprop aircraft and uses all of the elements mentioned on this page in an OpenConcept context.





