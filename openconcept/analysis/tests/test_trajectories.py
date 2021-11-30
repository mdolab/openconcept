from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openconcept.api as oc
import openmdao.api as om

"""
What are all the pieces I'm trying to test here?

Integrator
X. Integrator automatically added
X. Integrator automatically connected to rate source
X. Integrator options
X. Multiple integrated quantities in a group
X. No integrated quantities in a group
X. Comprehensible error message when created outside of a phase?
X. Single node integator works

Phase
X. Time automatically connected
X. Comprehensible error message when time doesn't exist
X. Num nodes correctly passed down the chain
X. Find many integrators in a model and connect all of them to correct time var
X. Find integrators at multiple levels and connect all of them to correct time var

Trajectory
x. Connections automatically created between many phases
x. Skip works correctly
"""
# ============== Feature Doc ==================== #
class TestForDocs(unittest.TestCase):
    def test_for_docs(self):
        prob = self.trajectory_example()
        assert_near_equal(prob['phase2.vm.ode_integ.velocity_final'], 1.66689857, 1e-8)
        
    def trajectory_example(self):
        """
        A simple example illustrating the auto-integration feature in OpenConcept

        It simulates the deceleration of a vehicle under aerodynamic drag.
        """
        import openconcept.api as oc
        import openmdao.api as om
        import numpy as np

        class NewtonSecondLaw(om.ExplicitComponent):
            "A regular OpenMDAO component computing acceleration from mass and force"
            def initialize(self):
                self.options.declare('num_nodes', default=1)

            def setup(self):
                num_nodes = self.options['num_nodes']
                self.add_input('mass', val=2.0*np.ones((num_nodes,)), units='kg')
                self.add_input('force', val=1.0*np.ones((num_nodes,)), units='N')
                # mark the output variable for integration using openmdao tags
                self.add_output('accel', val=0.5*np.ones((num_nodes,)), 
                units='m/s**2', tags=['integrate',
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
                self.add_subsystem('vm', VehicleModel(time_units='min', num_nodes=self.options['num_nodes']))

        class MyTraj(oc.TrajectoryGroup):
            "An OpenConcept TrajectoryGroup consists of one or more phases that may be linked together. This will often be a top-level model"
            def setup(self):
                self.add_subsystem('phase1', MyPhase(num_nodes=11)) 
                self.add_subsystem('phase2', MyPhase(num_nodes=11))
                # the link_phases directive ensures continuity of state variables across phase boundaries
                self.link_phases(self.phase1, self.phase2)

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
        
        return prob

# ============== IntegratorGroup Tests ========== #

class IntegratorGroupTestBase(oc.IntegratorGroup):
    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn = self.options['num_nodes']
        iv = self.add_subsystem('iv', om.IndepVarComp('x', val=np.linspace(0, 5, nn), units='s'))
        ec = self.add_subsystem('ec', om.ExecComp(['df = -10.2*x**2 + 4.2*x -10.5'], 
                  df={'val': 1.0*np.ones((nn,)),
                     'units': 'kg/s',
                     'tags': ['integrate', 'state_name:f', 'state_units:kg']},
                  x={'val': 1.0*np.ones((nn,)),
                       'units': 's'}))
        self.connect('iv.x', 'ec.x')
        self.set_order(['iv', 'ec', 'ode_integ'])

class TestIntegratorSingleState(unittest.TestCase):
    class TestPhase(oc.PhaseGroup):
        def initialize(self):
            self.options.declare('num_nodes', default=1)

        def setup(self):
            nn = self.options['num_nodes']
            self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
            self.add_subsystem('ic', IntegratorGroupTestBase(num_nodes=nn))

    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact)
        self.p['ic.ode_integ.f_initial'] = -2.0
        self.p.run_model()
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact-2.0)

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class IntegratorTestMultipleOutputs(IntegratorGroupTestBase):
    def setup(self):
        super(IntegratorTestMultipleOutputs, self).setup()
        nn = self.options['num_nodes']
        ec2 = self.add_subsystem('ec2', om.ExecComp(['df2 = 5.1*x**2 +0.5*x-7.2'], 
            df2={'val': 1.0*np.ones((nn,)),
                'units': 'W',
                'tags': ['integrate', 'state_name:f2', 'state_units:J']},
            x={'val': 1.0*np.ones((nn,)),
                'units': 's'}))
        self.connect('iv.x', 'ec2.x')
        self.set_order(['iv', 'ec', 'ec2', 'ode_integ'])

class TestIntegratorMultipleState(unittest.TestCase):
    class TestPhase(oc.PhaseGroup):
        def initialize(self):
            self.options.declare('num_nodes', default=1)

        def setup(self):
            nn = self.options['num_nodes']
            self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
            self.add_subsystem('ic', IntegratorTestMultipleOutputs(num_nodes=nn))

    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact)
        assert_near_equal(self.p['ic.ode_integ.f2'], f2_exact)
        self.p['ic.ode_integ.f_initial'] = -4.3
        self.p['ic.ode_integ.f2_initial'] = 85.1
        self.p.run_model()
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact - 4.3)
        assert_near_equal(self.p['ic.ode_integ.f2'], f2_exact + 85.1)  

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class IntegratorTestPromotes(IntegratorGroupTestBase):
    def setup(self):
        super(IntegratorTestPromotes, self).setup()
        nn = self.options['num_nodes']
        ec2 = self.add_subsystem('ec2', om.ExecComp(['df2 = 5.1*x**2 +0.5*x-7.2'], 
            df2={'val': 1.0*np.ones((nn,)),
                'units': 'W',
                'tags': ['integrate', 'state_name:f2', 'state_units:J', 'state_promotes:True']},
            x={'val': 1.0*np.ones((nn,)),
                'units': 's'}))
        self.connect('iv.x', 'ec2.x')
        self.set_order(['iv', 'ec', 'ec2', 'ode_integ'])

class TestIntegratorPromotes(unittest.TestCase):
    class TestPhase(oc.PhaseGroup):
        def initialize(self):
            self.options.declare('num_nodes', default=1)

        def setup(self):
            nn = self.options['num_nodes']
            self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
            self.add_subsystem('ic', IntegratorTestPromotes(num_nodes=nn))

    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact)
        assert_near_equal(self.p['ic.f2'], f2_exact)
        self.p['ic.ode_integ.f_initial'] = -4.3
        self.p['ic.ode_integ.f2_initial'] = 85.1
        self.p.run_model()
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact - 4.3)
        assert_near_equal(self.p['ic.f2'], f2_exact + 85.1)  
        
    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class IntegratorTestValLimits(IntegratorGroupTestBase):
    def setup(self):
        super(IntegratorTestValLimits, self).setup()
        nn = self.options['num_nodes']
        ec2 = self.add_subsystem('ec2', om.ExecComp(['df2 = 5.1*x**2 +0.5*x-7.2'], 
            df2={'val': 1.0*np.ones((nn,)),
                'units': 'W',
                'tags': ['integrate', 'state_name:f2', 'state_units:J', 
                         'state_upper:1e20', 'state_lower:-1e20', 'state_val:np.linspace(0,5,'+str(nn)+')']},
            x={'val': 1.0*np.ones((nn,)),
                'units': 's'}))
        self.connect('iv.x', 'ec2.x')
        self.set_order(['iv', 'ec', 'ec2', 'ode_integ'])

class TestIntegratorValLimits(unittest.TestCase):
    class TestPhase(oc.PhaseGroup):
        def initialize(self):
            self.options.declare('num_nodes', default=1)

        def setup(self):
            nn = self.options['num_nodes']
            self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
            self.add_subsystem('ic', IntegratorTestValLimits(num_nodes=nn))

    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        assert_near_equal(self.p['ic.ode_integ.f'], 0.0*np.ones((self.nn,)))
        assert_near_equal(self.p['ic.ode_integ.f2'], np.linspace(0, 5, self.nn))

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestIntegratorMultipleStateSingleNode(TestIntegratorMultipleState):

    def setUp(self):
        self.nn = 1
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        self.p.setup(force_alloc_complex=True)

class IntegratorTestDuplicateRateNames(IntegratorGroupTestBase):
    def setup(self):
        super(IntegratorTestDuplicateRateNames, self).setup()
        nn = self.options['num_nodes']
        ec2 = self.add_subsystem('ec2', om.ExecComp(['df = 5.1*x**3 +0.5*x-7.2'], 
            df={'val': 1.0*np.ones((nn,)),
                'units': 'W',
                'tags': ['integrate', 'state_name:f2', 'state_units:J']},
            x={'val': 1.0*np.ones((nn,)),
                'units': 's'}))
        self.connect('iv.x', 'ec2.x')
        self.set_order(['iv', 'ec', 'ec2', 'ode_integ'])

class TestIntegratorDuplicateRateName(unittest.TestCase):
    class TestPhase(oc.PhaseGroup):
        def initialize(self):
            self.options.declare('num_nodes', default=1)

        def setup(self):
            nn = self.options['num_nodes']
            self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
            self.add_subsystem('ic', IntegratorTestDuplicateRateNames(num_nodes=nn))

    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        
    def test_asserts(self):
        with self.assertRaises(ValueError) as cm:
            self.p.setup(force_alloc_complex=True)

class IntegratorTestDuplicateStateNames(IntegratorGroupTestBase):
    def setup(self):
        super(IntegratorTestDuplicateStateNames, self).setup()
        nn = self.options['num_nodes']
        ec2 = self.add_subsystem('ec2', om.ExecComp(['df2 = 5.1*x**3 +0.5*x-7.2'], 
            df2={'val': 1.0*np.ones((nn,)),
                'units': 'W',
                'tags': ['integrate', 'state_name:f', 'state_units:J']},
            x={'val': 1.0*np.ones((nn,)),
                'units': 's'}))
        self.connect('iv.x', 'ec2.x')
        self.set_order(['iv', 'ec', 'ec2', 'ode_integ'])

class TestIntegratorDuplicateStateName(unittest.TestCase):
    class TestPhase(oc.PhaseGroup):
        def initialize(self):
            self.options.declare('num_nodes', default=1)

        def setup(self):
            nn = self.options['num_nodes']
            self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
            self.add_subsystem('ic', IntegratorTestDuplicateStateNames(num_nodes=nn))

    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        
    def test_asserts(self):
        with self.assertRaises(ValueError) as cm:
            self.p.setup(force_alloc_complex=True)
        self.assertIn("Variable name 'f_final' already exists.", '{}'.format(cm.exception))

class TestIntegratorOutsideofPhase(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=IntegratorGroupTestBase(num_nodes=self.nn))
        
    def test_asserts(self):
        with self.assertRaises(NameError) as cm:
            self.p.setup(force_alloc_complex=True)
        self.assertEqual('{}'.format(cm.exception),
                         'Integrator group must be created within an OpenConcept phase or Dymos trajectory')

class TestIntegratorNoIntegratedState(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        grp = oc.IntegratorGroup()
        grp.add_subsystem('iv', om.IndepVarComp('a', val=1.0))
        phase = oc.PhaseGroup()
        phase.add_subsystem('iv', om.IndepVarComp('duration', val=3.0, units='s'), promotes_outputs=['*'])
        phase.add_subsystem('grp', grp)
        self.p = om.Problem(model=phase)
        self.p.setup()
        
    def test_runs(self):
        self.p.run_model()

class IntegratorGroupWithGroup(IntegratorGroupTestBase):
    def setup(self):
        super(IntegratorGroupWithGroup, self).setup()
        self.add_subsystem('group', om.Group())

class TestIntegratorWithGroup(unittest.TestCase):

    class TestPhase(oc.PhaseGroup):
        def initialize(self):
            self.options.declare('num_nodes', default=1)

        def setup(self):
            nn = self.options['num_nodes']
            self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
            self.add_subsystem('ic', IntegratorGroupWithGroup(num_nodes=nn))

    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact)
        self.p['ic.ode_integ.f_initial'] = -2.0
        self.p.run_model()
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact-2.0)

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)


class IntegratorGroupTestPromotedRate(oc.IntegratorGroup):
    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn = self.options['num_nodes']
        iv = self.add_subsystem('iv', om.IndepVarComp('x', val=np.linspace(0, 5, nn), units='s'))
        ec = self.add_subsystem('ec', om.ExecComp(['df = -10.2*x**2 + 4.2*x -10.5'], 
                  df={'val': 1.0*np.ones((nn,)),
                     'units': 'kg/s',
                     'tags': ['integrate', 'state_name:f', 'state_units:kg']},
                  x={'val': 1.0*np.ones((nn,)),
                       'units': 's'}), promotes_outputs=['df'])
        self.connect('iv.x', 'ec.x')
        self.set_order(['iv', 'ec', 'ode_integ'])

class TestIntegratorSingleStatePromotedRate(unittest.TestCase):
    class TestPhase(oc.PhaseGroup):
        def initialize(self):
            self.options.declare('num_nodes', default=1)

        def setup(self):
            nn = self.options['num_nodes']
            self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
            self.add_subsystem('ic', IntegratorGroupTestPromotedRate(num_nodes=nn))

    def setUp(self):
        self.nn = 5
        self.p = om.Problem(model=self.TestPhase(num_nodes=self.nn))
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact)
        self.p['ic.ode_integ.f_initial'] = -2.0
        self.p.run_model()
        assert_near_equal(self.p['ic.ode_integ.f'], f_exact-2.0)

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

# ============== PhaseGroup Tests ========== #

class TestPhaseNoTime(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        grp = oc.IntegratorGroup()
        grp.add_subsystem('iv', om.IndepVarComp('a', val=1.0))
        phase = oc.PhaseGroup()
        phase.add_subsystem('grp', grp)
        self.p = om.Problem(model=phase)
        
    def test_raises_error(self):
        with self.assertRaises(NameError) as x:
            self.p.setup()

class TestPhaseMultipleIntegrators(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        grp1 = IntegratorGroupTestBase(num_nodes=self.nn)
        grp2 = om.Group()
        grp2a = grp2.add_subsystem('a', IntegratorGroupTestBase(num_nodes=self.nn))
        grp2b = grp2.add_subsystem('b', IntegratorGroupTestBase(num_nodes=self.nn))
        phase = oc.PhaseGroup(num_nodes=self.nn)
        phase.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
        phase.add_subsystem('grp1', grp1)
        phase.add_subsystem('grp2', grp2)

        self.p = om.Problem(model=phase)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        assert_near_equal(self.p['grp1.ode_integ.f'], f_exact)
        assert_near_equal(self.p['grp2.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['grp2.b.ode_integ.f'], f_exact)
        self.p['grp2.a.ode_integ.f_initial'] = -2.0
        self.p.run_model()
        assert_near_equal(self.p['grp2.a.ode_integ.f'], f_exact-2.0)

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestPhasePromotedDurationVariable(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        grp1 = IntegratorGroupTestBase(num_nodes=self.nn)
        grp2 = om.Group()
        grp2a = grp2.add_subsystem('a', IntegratorGroupTestBase(num_nodes=self.nn))
        grp2b = grp2.add_subsystem('b', IntegratorGroupTestBase(num_nodes=self.nn))
        phase = oc.PhaseGroup(num_nodes=self.nn)
        phase.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
        phase.add_subsystem('grp1', grp1)
        phase.add_subsystem('grp2', grp2)
        phase.add_subsystem('c', om.ExecComp('result = 1.0*duration', duration={'units':'s'}), promotes_inputs=['duration'])

        self.p = om.Problem(model=phase)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        assert_near_equal(self.p['grp1.ode_integ.f'], f_exact)
        assert_near_equal(self.p['grp2.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['grp2.b.ode_integ.f'], f_exact)
        self.p['grp2.a.ode_integ.f_initial'] = -2.0
        self.p.run_model()
        assert_near_equal(self.p['grp2.a.ode_integ.f'], f_exact-2.0)

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)


# ============ Trajectory Tests ============ #

class PhaseForTrajTest(oc.PhaseGroup):
    def initialize(self):
        self.options.declare('num_nodes', default=1)
    
    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
        a = self.add_subsystem('a', IntegratorGroupTestBase(num_nodes=nn))
        b = self.add_subsystem('b', IntegratorTestMultipleOutputs(num_nodes=nn))

class PhaseForTrajTestWithPromotion(oc.PhaseGroup):
    def initialize(self):
        self.options.declare('num_nodes', default=1)
    
    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
        a = self.add_subsystem('a', IntegratorGroupTestBase(num_nodes=nn))
        # promote the outputs of b
        b = self.add_subsystem('b', IntegratorTestMultipleOutputs(num_nodes=nn), promotes_outputs=['*f2*'], promotes_inputs=['*df2'])

class PhaseForTrajTestWithPromotionNamesCollide(oc.PhaseGroup):
    def initialize(self):
        self.options.declare('num_nodes', default=1)
    
    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('iv', om.IndepVarComp('duration', val=5.0, units='s'), promotes_outputs=['*'])
        a = self.add_subsystem('a', IntegratorGroupTestBase(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        # promote the outputs of b
        b = self.add_subsystem('b', IntegratorTestMultipleOutputs(num_nodes=nn), promotes_outputs=['*f2*'], promotes_inputs=['*df2'])

class TestTrajectoryAllPhaseConnect(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        traj = oc.TrajectoryGroup()
       
        phase1 = traj.add_subsystem('phase1', PhaseForTrajTest(num_nodes=5))
        phase2 = traj.add_subsystem('phase2', PhaseForTrajTest(num_nodes=5))
        phase3 = traj.add_subsystem('phase3', PhaseForTrajTest(num_nodes=5))

        traj.link_phases(phase1, phase2)
        traj.link_phases(phase2, phase3)

        self.p = om.Problem(model=traj)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x

        # check first phase result
        assert_near_equal(self.p['phase1.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f2'], f2_exact)

        # check second phase result
        assert_near_equal(self.p['phase2.a.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.b.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.b.ode_integ.f2'], f2_exact+f2_exact[-1])

        # check third phase result
        assert_near_equal(self.p['phase3.a.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.b.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.b.ode_integ.f2'], f2_exact+2.0*f2_exact[-1])

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestTrajectoryAllPhaseConnectWithVarPromotion(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        traj = oc.TrajectoryGroup()
       
        phase1 = traj.add_subsystem('phase1', PhaseForTrajTestWithPromotion(num_nodes=5))
        phase2 = traj.add_subsystem('phase2', PhaseForTrajTestWithPromotion(num_nodes=5))
        phase3 = traj.add_subsystem('phase3', PhaseForTrajTestWithPromotion(num_nodes=5))

        traj.link_phases(phase1, phase2)
        traj.link_phases(phase2, phase3)

        self.p = om.Problem(model=traj)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x

        # check first phase result
        assert_near_equal(self.p['phase1.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.ode_integ.f2'], f2_exact)

        # check second phase result
        assert_near_equal(self.p['phase2.a.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.b.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.ode_integ.f2'], f2_exact+f2_exact[-1])

        # check third phase result
        assert_near_equal(self.p['phase3.a.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.b.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.ode_integ.f2'], f2_exact+2.0*f2_exact[-1])

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestTrajectorySkipPromotedVar(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        traj = oc.TrajectoryGroup()
       
        phase1 = traj.add_subsystem('phase1', PhaseForTrajTestWithPromotion(num_nodes=5))
        phase2 = traj.add_subsystem('phase2', PhaseForTrajTestWithPromotion(num_nodes=5))
        phase3 = traj.add_subsystem('phase3', PhaseForTrajTestWithPromotion(num_nodes=5))

        traj.link_phases(phase1, phase2, states_to_skip=['ode_integ.f2'])
        traj.link_phases(phase2, phase3)

        self.p = om.Problem(model=traj)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x

        # check first phase result
        assert_near_equal(self.p['phase1.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.ode_integ.f2'], f2_exact)

        # check second phase result
        assert_near_equal(self.p['phase2.a.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.b.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.ode_integ.f2'], f2_exact)

        # check third phase result
        assert_near_equal(self.p['phase3.a.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.b.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.ode_integ.f2'], f2_exact+f2_exact[-1])

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestTrajectoryAllPhaseConnectWithVarPromotionODEIntegCollide(unittest.TestCase):
    # This checks for the situation when multiple integrator comps are promoting up ode_integ to the phase level
    # When this happens duplicate connections can occur
    def setUp(self):
        self.nn = 5
        traj = oc.TrajectoryGroup()
       
        phase1 = traj.add_subsystem('phase1', PhaseForTrajTestWithPromotionNamesCollide(num_nodes=5))
        phase2 = traj.add_subsystem('phase2', PhaseForTrajTestWithPromotionNamesCollide(num_nodes=5))
        phase3 = traj.add_subsystem('phase3', PhaseForTrajTestWithPromotionNamesCollide(num_nodes=5))

        traj.link_phases(phase1, phase2)
        traj.link_phases(phase2, phase3)

        self.p = om.Problem(model=traj)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x

        # check first phase result
        assert_near_equal(self.p['phase1.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.ode_integ.f2'], f2_exact)

        # check second phase result
        assert_near_equal(self.p['phase2.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.b.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.ode_integ.f2'], f2_exact+f2_exact[-1])

        # check third phase result
        assert_near_equal(self.p['phase3.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.b.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.ode_integ.f2'], f2_exact+2.0*f2_exact[-1])

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestTrajectoryTwoPhaseConnect(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        traj = oc.TrajectoryGroup()
       
        phase1 = traj.add_subsystem('phase1', PhaseForTrajTest(num_nodes=5))
        phase2 = traj.add_subsystem('phase2', PhaseForTrajTest(num_nodes=5))
        phase3 = traj.add_subsystem('phase3', PhaseForTrajTest(num_nodes=5))

        traj.link_phases(phase1, phase2)

        self.p = om.Problem(model=traj)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x

        # check first phase result
        assert_near_equal(self.p['phase1.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f2'], f2_exact)

        # check second phase result
        assert_near_equal(self.p['phase2.a.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.b.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.b.ode_integ.f2'], f2_exact+f2_exact[-1])

        # check third phase result
        assert_near_equal(self.p['phase3.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase3.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase3.b.ode_integ.f2'], f2_exact)

class TestTrajectorySkipState(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        traj = oc.TrajectoryGroup()
       
        phase1 = traj.add_subsystem('phase1', PhaseForTrajTest(num_nodes=5))
        phase2 = traj.add_subsystem('phase2', PhaseForTrajTest(num_nodes=5))
        phase3 = traj.add_subsystem('phase3', PhaseForTrajTest(num_nodes=5))

        traj.link_phases(phase1, phase2, states_to_skip=['b.ode_integ.f'])
        traj.link_phases(phase2, phase3, states_to_skip=['b.ode_integ.f2'])

        self.p = om.Problem(model=traj)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x

        # check first phase result
        assert_near_equal(self.p['phase1.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase1.b.ode_integ.f2'], f2_exact)

        # check second phase result
        assert_near_equal(self.p['phase2.a.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['phase2.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['phase2.b.ode_integ.f2'], f2_exact+f2_exact[-1])

        # check third phase result
        assert_near_equal(self.p['phase3.a.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['phase3.b.ode_integ.f'], f_exact+1.0*f_exact[-1])
        assert_near_equal(self.p['phase3.b.ode_integ.f2'], f2_exact)

class TestTrajectoryLinkPhaseStrings(unittest.TestCase):
    def test_raises(self):
        self.nn = 5
        traj = oc.TrajectoryGroup()
       
        phase1 = traj.add_subsystem('phase1', PhaseForTrajTest(num_nodes=5))

        with self.assertRaises(ValueError) as context:
            traj.link_phases('phase1', 'phase2', states_to_skip=['b.ode_integ.f'])

class TestBuryTrajectoryOneLevelDown(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        traj = oc.TrajectoryGroup()
       
        phase1 = traj.add_subsystem('phase1', PhaseForTrajTest(num_nodes=5))
        phase2 = traj.add_subsystem('phase2', PhaseForTrajTest(num_nodes=5))
        phase3 = traj.add_subsystem('phase3', PhaseForTrajTest(num_nodes=5))

        traj.link_phases(phase1, phase2)
        traj.link_phases(phase2, phase3)
        topgroup = om.Group()

        topgroup.add_subsystem('traj', traj)
        self.p = om.Problem(model=topgroup)
        self.p.setup(force_alloc_complex=True)

    def test_results(self):
        self.p.run_model()
        x = np.linspace(0, 5, self.nn)
        f_exact = -10.2*x**3/3 + 4.2*x**2/2 -10.5*x
        f2_exact = 5.1*x**3/3 +0.5*x**2/2-7.2*x

        # check first phase result
        assert_near_equal(self.p['traj.phase1.a.ode_integ.f'], f_exact)
        assert_near_equal(self.p['traj.phase1.b.ode_integ.f'], f_exact)
        assert_near_equal(self.p['traj.phase1.b.ode_integ.f2'], f2_exact)

        # check second phase result
        assert_near_equal(self.p['traj.phase2.a.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['traj.phase2.b.ode_integ.f'], f_exact+f_exact[-1])
        assert_near_equal(self.p['traj.phase2.b.ode_integ.f2'], f2_exact+f2_exact[-1])

        # check third phase result
        assert_near_equal(self.p['traj.phase3.a.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['traj.phase3.b.ode_integ.f'], f_exact+2.0*f_exact[-1])
        assert_near_equal(self.p['traj.phase3.b.ode_integ.f2'], f2_exact+2.0*f2_exact[-1])

    def test_partials(self):
        self.p.run_model()
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

# TODO test promoted skipped states
if __name__ == "__main__":
    unittest.main()