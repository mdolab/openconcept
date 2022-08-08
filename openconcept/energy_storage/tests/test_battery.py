from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem, DirectSolver, NewtonSolver, IndepVarComp
from openconcept.energy_storage import SimpleBattery, LiquidCooledBattery


class BatteryTestGroup(Group):
    """
    Test the battery component
    """
    def initialize(self):
        self.options.declare('vec_size',default=1,desc="Number of mission analysis points to run")
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('p', default=5000., desc='Battery specific power (W/kg)' )
        self.options.declare('e', default=300., desc='Battery spec energy CAREFUL: (Wh/kg)')
        self.options.declare('cost_inc', default=50., desc='$ cost per kg')
        self.options.declare('cost_base', default=1., desc= '$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        if not use_defaults:
            eta_b = self.options['efficiency']
            p = self.options['p']
            e = self.options['e']
            ci = self.options['cost_inc']
            cb = self.options['cost_base']
            self.add_subsystem('battery', SimpleBattery(num_nodes=nn,
                                                        efficiency=eta_b,
                                                        specific_power=p,
                                                        specific_energy=e,
                                                        cost_inc=ci,
                                                        cost_base=cb))
        else:
            self.add_subsystem('battery', SimpleBattery(num_nodes=nn))

        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('battery_weight', val=100, units='kg')
        iv.add_output('elec_load', val=np.ones(nn) * 100, units='kW')
        self.connect('iv.battery_weight','battery.battery_weight')
        self.connect('iv.elec_load','battery.elec_load')

class SimpleBatteryTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(BatteryTestGroup(vec_size=10, use_defaults=True))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob['battery.heat_out'], np.ones(10)*100*0.0, tolerance=1e-15)
        assert_near_equal(prob['battery.component_sizing_margin'], np.ones(10)*0.20, tolerance=1e-15)
        assert_near_equal(prob['battery.component_cost'], 5001, tolerance=1e-15)
        assert_near_equal(prob.get_val('battery.max_energy', units='W*h'), 300*100, tolerance=1e-15)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        prob = Problem(BatteryTestGroup(vec_size=10,
                                        use_defaults=False,
                                        efficiency=0.95,
                                        p=3000,
                                        e=500,
                                        cost_inc=100,
                                        cost_base=0))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob.get_val('battery.heat_out', units='kW'), np.ones(10)*100*0.05, tolerance=1e-15)
        assert_near_equal(prob['battery.component_sizing_margin'], np.ones(10)/3, tolerance=1e-15)
        assert_near_equal(prob['battery.component_cost'], 10000, tolerance=1e-15)
        assert_near_equal(prob.get_val('battery.max_energy', units='W*h'), 500*100, tolerance=1e-15)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class QuasiSteadyBatteryCoolingTestCase(unittest.TestCase):
    """
    Test the liquid cooled battery in quasi-steady (massless) mode
    """
    def generate_model(self, nn):
        prob = Problem()
        iv = prob.model.add_subsystem('iv', IndepVarComp(), promotes_outputs=['*'])
        iv.add_output('q_in', val=np.linspace(2000,5000,nn), units='W')
        iv.add_output('mdot_coolant', val=1*np.ones((nn,)), units='kg/s')
        iv.add_output('T_in', val=25*np.ones((nn,)), units='degC')
        iv.add_output('battery_weight', val=100., units='kg')
        prob.model.add_subsystem('test', LiquidCooledBattery(num_nodes=nn, quasi_steady=True), promotes=['*'])
        prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=True)
        prob.model.linear_solver = DirectSolver()
        prob.setup(check=True, force_alloc_complex=True)
        return prob

    def test_scalar(self):
        prob = self.generate_model(nn=1)
        prob.run_model()
        assert_near_equal(prob.get_val('dTdt'), 0.0, tolerance=1e-14)
        assert_near_equal(prob.get_val('T_surface', units='K'), 298.94004878, tolerance=1e-10)
        assert_near_equal(prob.get_val('T_core', units='K'), 307.10184074, tolerance=1e-10)
        assert_near_equal(prob.get_val('test.hex.q', units='W'), 2000.0, tolerance=1e-10)
        assert_near_equal(prob.get_val('T_out', units='K'), 298.6761773, tolerance=1e-10)
        assert_near_equal(prob.get_val('T', units='K'), 303.02094476, tolerance=1e-10)
        
        partials = prob.check_partials(method='cs',compact_print=True, step=1e-50)
        assert_check_partials(partials)

    def test_vector(self):
        prob = self.generate_model(nn=11)
        prob.run_model()
        assert_near_equal(prob.get_val('dTdt'), np.zeros((11,)), tolerance=1e-14)
        assert_near_equal(prob.get_val('T_surface', units='K'),
                          np.array([333.94004878, 334.0585561 , 334.17706342, 334.29557074,
                           334.41407805, 334.53258537, 334.65109269, 334.76960001,
                           334.88810732, 335.00661464, 335.12512196])-35., tolerance=1e-10)
        assert_near_equal(prob.get_val('T_core', units='K'), 
                          np.array([342.10184074, 343.44461685, 344.78739296, 346.13016907,
                           347.47294518, 348.81572129, 350.1584974 , 351.50127351,
                           352.84404962, 354.18682573, 355.52960184])-35., tolerance=1e-10)
        assert_near_equal(prob.get_val('test.hex.q', units='W'), np.linspace(2000,5000,11), tolerance=1e-10)
        assert_near_equal(prob.get_val('T_out', units='K'), 
                          np.array([333.67617732, 333.75510392, 333.83403052, 333.91295712,
                           333.99188371, 334.07081031, 334.14973691, 334.22866351,
                           334.30759011, 334.38651671, 334.4654433 ])-35., tolerance=1e-10)
        assert_near_equal(prob.get_val('T', units='K'),
                          np.array([338.02094476, 338.75158647, 339.48222819, 340.2128699 ,
                        340.94351162, 341.67415333, 342.40479505, 343.13543676,
                        343.86607847, 344.59672019, 345.3273619])-35., tolerance=1e-10)

        # prob.model.list_outputs(print_arrays=True, units='True')
        partials = prob.check_partials(method='cs',compact_print=True, step=1e-50)
        assert_check_partials(partials)

class UnsteadyBatteryCoolingTestCase(unittest.TestCase):
    """
    Test the liquid cooled battery in unsteady mode
    """
    def generate_model(self, nn):
        """
        An example demonstrating unsteady battery cooling
        """
        from openconcept.mission import PhaseGroup, TrajectoryGroup
        import openmdao.api as om
        import numpy as np

        class VehicleModel(Group):
            def initialize(self):
                self.options.declare('num_nodes', default=11)

            def setup(self):
                num_nodes = self.options['num_nodes']
                ivc = self.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
                ivc.add_output('battery_heat', val=np.ones((num_nodes,))*5000, units='W')
                ivc.add_output('coolant_temp', 25.*np.ones((num_nodes,)), units='degC')
                ivc.add_output('mdot_coolant', 1.0*np.ones((num_nodes,)), units='kg/s')
                ivc.add_output('battery_weight', 100, units='kg')
                ivc.add_output('n_cpb', 21)
                ivc.add_output('t_channel', 0.0005, units='m')


                self.add_subsystem('bcs', LiquidCooledBattery(num_nodes=num_nodes, quasi_steady=False))
                self.connect('battery_heat', 'bcs.q_in')
                self.connect('coolant_temp', 'bcs.T_in')
                self.connect('mdot_coolant', 'bcs.mdot_coolant')
                self.connect('battery_weight', 'bcs.battery_weight')
                self.connect('n_cpb', 'bcs.n_cpb')
                self.connect('t_channel', 'bcs.t_channel')

        class TrajectoryPhase(PhaseGroup):
            "An OpenConcept Phase comprises part of a time-based TrajectoryGroup and always needs to have a 'duration' defined"
            def setup(self):
                self.add_subsystem('ivc', IndepVarComp('duration', val=30, units='min'), promotes_outputs=['duration'])
                self.add_subsystem('vm', VehicleModel(num_nodes=self.options['num_nodes']))

        class Trajectory(TrajectoryGroup):
            "An OpenConcept TrajectoryGroup consists of one or more phases that may be linked together. This will often be a top-level model"
            def setup(self):
                self.add_subsystem('phase1', TrajectoryPhase(num_nodes=nn)) 
                # self.add_subsystem('phase2', TrajectoryPhase(num_nodes=nn))
                # the link_phases directive ensures continuity of state variables across phase boundaries
                # self.link_phases(self.phase1, self.phase2)

        prob = Problem(Trajectory())
        prob.model.nonlinear_solver = NewtonSolver(iprint=2)
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.model.nonlinear_solver.options['maxiter'] = 20
        prob.model.nonlinear_solver.options['atol'] = 1e-6
        prob.model.nonlinear_solver.options['rtol'] = 1e-6    
        prob.setup(force_alloc_complex=True)
        # set the initial value of the state at the beginning of the TrajectoryGroup
        prob['phase1.vm.bcs.T_initial'] = 300.
        prob.run_model()
        # prob.model.list_outputs(print_arrays=True, units=True)
        # prob.model.list_inputs(print_arrays=True, units=True)
        
        return prob

    def test_vector(self):
        prob = self.generate_model(nn=11)
        prob.run_model()
        assert_near_equal(prob.get_val('phase1.vm.bcs.T_surface', units='K'),
                          np.array([298.45006299, 299.70461767, 299.97097736, 300.08642573,
                               300.11093705, 300.121561  , 300.12381662, 300.12479427,
                               300.12500184, 300.1250918 , 300.12511091]), tolerance=1e-10)
        assert_near_equal(prob.get_val('phase1.vm.bcs.T_core', units='K'), 
                          np.array([301.54993701, 315.76497532, 318.78302876, 320.09114476,
                               320.36887627, 320.48925354, 320.51481133, 320.52588886,
                               320.52824077, 320.52926016, 320.52947659]), tolerance=1e-10)
        assert_near_equal(prob.get_val('phase1.vm.bcs.T_out', units='K'), 
                          np.array([298.34984379, 299.18538488, 299.36278206, 299.43967138,
                               299.45599607, 299.46307168, 299.46457394, 299.46522506,
                               299.4653633 , 299.46542322, 299.46543594]), tolerance=1e-10)
        assert_near_equal(prob.get_val('phase1.vm.bcs.T', units='K'),
                          np.array([300.        , 307.73479649, 309.37700306, 310.08878525,
                               310.23990666, 310.30540727, 310.31931397, 310.32534156,
                               310.3266213 , 310.32717598, 310.32729375]), tolerance=1e-10)

        # prob.model.list_outputs(print_arrays=True, units='True')
        partials = prob.check_partials(method='cs',compact_print=True, step=1e-50)
        assert_check_partials(partials)
