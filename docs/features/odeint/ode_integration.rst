.. _ODEIntegration:

***************
ODE Integration
***************

OpenConcept includes built-in utilities for integrating ODEs, a frequent task in airplane performance analysis.

`Component`, `Group`, and Connections
-------------------------------------
If users are running examples or instantiating OpenConcept native components (e.g. the battery), there's no need to know what's going on under the hood in the ODE integration.
However, for writing custom analysis routines, developing new components, or troubleshooting new airplane models, some background can be helpful.
The `openconcept.Trajectory` class acts just like an OpenMDAO `Group` except that it adds the ability to automatically link integrated states from one phase of a trajectory to the next using the `link_phases` method.
The `openconcept.Phase` class acts just like an OpenMDAO `Group` except it finds all the integrated states and automatically links the time duration variable to them. 
It also collects the names of all the integrated states so that the `Trajectory` can find them and link them.
The `openconcept.IntegratorGroup` class again acts just like OpenMDAO's `Group` except it adds an ODE Integration component (called ode_integ), locates output variables tagged with the "integrate" tag, and automatically connects the tagged rate source to the integrator.
Any state you wish to be automatically integrated needs to be held in an `IntegratorGroup`.
However, `IntegratorGroup` instances can be buried deep in a model made up of mainly plain `Group`. 

The following example illustrates the usage of this feature.
The tags following the "integrate" tag define the name, units, and default values of the integrated output.

.. embed-code::
    openconcept.analysis.tests.test_trajectories.TestForDocs.trajectory_example
