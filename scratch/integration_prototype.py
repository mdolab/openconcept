import openconcept.api as oc
import openmdao.api as om
import numpy as np

# --------------------- This is just an example
class NewtonSecondLaw(om.ExplicitComponent):
    "A regular OpenMDAO component"
    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        num_nodes = self.options['num_nodes']
        self.add_input('mass', val=2.0*np.ones((num_nodes,)), units='kg')
        self.add_input('force', val=1.0*np.ones((num_nodes,)), units='N')
        # mark the output variable for integration using openmdao tags
        self.add_output('accel', val=0.5*np.ones((num_nodes,)), units='m/s**2', tags=['integrate',
                                                                                      'state_name:velocity',
                                                                                      'state_units:m/s',
                                                                                      'state_val:5.0',
                                                                                      'state_promotes:True'])
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        outputs['accel'] = inputs['force'] / inputs['mass']

class DragEquation(om.ExplicitComponent):
    "Another regular OpenMDAO component that happens to take a state variable as input"
    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        num_nodes = self.options['num_nodes']
        self.add_input('velocity', val=0.0*np.ones((num_nodes,)), units='m/s')
        self.add_output('force', val=0.0*np.ones((num_nodes,)), units='N')
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        outputs['force'] = -0.10 * inputs['velocity'] ** 2

class VehicleModel(oc.IntegratorGroup):
    """
    A user wishing to integrate an ODE rate will need to subclass 
    this IntegratorGroup instead of the default OpenMDAO Group
    but it behaves in all other ways exactly the same.

    You can now incorporate this Vehicle Model in your model tree
    using the regular Group. Only the direct parent of the rate
    to be integrated has to use this special class.
    """
    def initialize(self):
        self.options.declare('num_nodes', default=11)

    def setup(self):
        num_nodes = self.options['num_nodes']
        self.add_subsystem('nsl', NewtonSecondLaw(num_nodes=num_nodes))
        self.add_subsystem('drag', DragEquation(num_nodes=num_nodes))
        # velocity output is created automatically by the integrator
        # if you want you can promote it, or you can connect it directly as here
        self.connect('velocity', 'drag.velocity')
        self.connect('drag.force','nsl.force')

class MyPhase(oc.PhaseGroup):
    "An OpenConcept Phase comprises part of a time-based TrajectoryGroup and always needs to have a 'duration' defined"
    def setup(self):
        self.add_subsystem('ivc', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['duration'])
        self.add_subsystem('vm', VehicleModel(time_units='min'))

class MyTraj(oc.TrajectoryGroup):
    "An OpenConcept TrajectoryGroup consists of one or more phases that may be linked together. This will often be a top-level model"
    def setup(self):
        self.add_subsystem('phase1', MyPhase()) 
        self.add_subsystem('phase2', MyPhase())
        # the link_phases directive ensures continuity of state variables across phase boundaries
        self.link_phases(self.phase1, self.phase2)

if __name__ == "__main__":
    prob = om.Problem(MyTraj())
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6    
    prob.setup()
    # set the initial value of the state at the beginning of the TrajectoryGroup
    prob['phase1.vm.ode_integ.velocity_initial'] = 10.0
    prob.run_model()
    prob.model.list_outputs(print_arrays=True, units=True)
    prob.model.list_inputs(print_arrays=True, units=True)