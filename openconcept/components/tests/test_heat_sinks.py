from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.components.heat_sinks import LiquidCooledBattery, LiquidCooledMotor, SimpleHose, SimplePump

class QuasiSteadyBatteryCoolingTestCase(unittest.TestCase):
    """
    Test the liquid cooled battery in quasi-steady (massless) mode
    """
    def generate_model(self, nn):
        prob = om.Problem()
        iv = prob.model.add_subsystem('iv', om.IndepVarComp(), promotes_outputs=['*'])
        iv.add_output('q_in', val=np.linspace(2000,5000,nn), units='W')
        iv.add_output('mdot_coolant', val=1*np.ones((nn,)), units='kg/s')
        iv.add_output('T_in', val=25*np.ones((nn,)), units='degC')
        iv.add_output('battery_weight', val=100., units='kg')
        prob.model.add_subsystem('test', LiquidCooledBattery(num_nodes=nn, quasi_steady=True), promotes=['*'])
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        prob.model.linear_solver = om.DirectSolver()
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

        class VehicleModel(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', default=11)

            def setup(self):
                num_nodes = self.options['num_nodes']
                ivc = self.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
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
                self.add_subsystem('ivc', om.IndepVarComp('duration', val=30, units='min'), promotes_outputs=['duration'])
                self.add_subsystem('vm', VehicleModel(num_nodes=self.options['num_nodes']))

        class Trajectory(TrajectoryGroup):
            "An OpenConcept TrajectoryGroup consists of one or more phases that may be linked together. This will often be a top-level model"
            def setup(self):
                self.add_subsystem('phase1', TrajectoryPhase(num_nodes=nn)) 
                # self.add_subsystem('phase2', TrajectoryPhase(num_nodes=nn))
                # the link_phases directive ensures continuity of state variables across phase boundaries
                # self.link_phases(self.phase1, self.phase2)

        prob = om.Problem(Trajectory())
        prob.model.nonlinear_solver = om.NewtonSolver(iprint=2)
        prob.model.linear_solver = om.DirectSolver()
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

class QuasiSteadyMotorCoolingTestCase(unittest.TestCase):
    """
    Test the liquid cooled motor in quasi-steady (massless) mode
    """
    def generate_model(self, nn):
        prob = om.Problem()
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('q_in', val=np.ones((nn,))*10000, units='W')
        ivc.add_output('T_in', 25.*np.ones((nn,)), units='degC')
        ivc.add_output('mdot_coolant', 3.0*np.ones((nn,)), units='kg/s')
        ivc.add_output('motor_weight', 40, units='kg')
        ivc.add_output('power_rating', 200, units='kW')
        prob.model.add_subsystem('lcm', LiquidCooledMotor(num_nodes=nn, quasi_steady=True), promotes_inputs=['*'])
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        prob.model.linear_solver = om.DirectSolver()
        prob.setup(check=True, force_alloc_complex=True)
        return prob

    def test_scalar(self):
        prob = self.generate_model(nn=1)
        prob.run_model()
        power_rating = 200000
        mdot_coolant = 3.0
        q_generated = power_rating * 0.05
        cp_coolant = 3801
        UA = 1100/650000*power_rating
        Cmin = cp_coolant * mdot_coolant # cp * mass flow rate
        NTU = UA/Cmin
        T_in = 298.15
        effectiveness = 1 - np.exp(-NTU)
        delta_T = q_generated / effectiveness / Cmin
        assert_near_equal(prob.get_val('lcm.dTdt'), 0.0, tolerance=1e-14)
        assert_near_equal(prob.get_val('lcm.T', units='K'), T_in + delta_T, tolerance=1e-10)
        assert_near_equal(prob.get_val('lcm.T_out', units='K'), T_in + q_generated / Cmin, tolerance=1e-10)
        partials = prob.check_partials(method='cs',compact_print=True)
        # prob.model.list_outputs(print_arrays=True, units=True)
        assert_check_partials(partials)

    def test_vector(self):
        prob = self.generate_model(nn=11)
        prob.run_model()
        power_rating = 200000
        mdot_coolant = 3.0
        q_generated = power_rating * 0.05
        cp_coolant = 3801
        UA = 1100/650000*power_rating
        Cmin = cp_coolant * mdot_coolant # cp * mass flow rate
        NTU = UA/Cmin
        T_in = 298.15
        effectiveness = 1 - np.exp(-NTU)
        delta_T = q_generated / effectiveness / Cmin
        assert_near_equal(prob.get_val('lcm.dTdt'), np.zeros((11,)), tolerance=1e-14)
        assert_near_equal(prob.get_val('lcm.T', units='K'), np.ones((11,))*(T_in + delta_T), tolerance=1e-10)
        assert_near_equal(prob.get_val('lcm.T_out', units='K'), np.ones((11,))*(T_in + q_generated / Cmin), tolerance=1e-10)
        # prob.model.list_outputs(print_arrays=True, units='True')
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class UnsteadyMotorCoolingTestCase(unittest.TestCase):
    """
    Test the liquid cooled motor in unsteady mode
    """
    def generate_model(self, nn):
        """
        An example demonstrating unsteady motor cooling
        """
        from openconcept.mission import PhaseGroup, TrajectoryGroup
        import openmdao.api as om
        import numpy as np

        class VehicleModel(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', default=11)                
                
            def setup(self):
                num_nodes = self.options['num_nodes']
                ivc = self.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
                ivc.add_output('q_in', val=np.ones((num_nodes,))*10000, units='W')
                ivc.add_output('T_in', 25.*np.ones((num_nodes,)), units='degC')
                ivc.add_output('mdot_coolant', 3.0*np.ones((num_nodes,)), units='kg/s')
                ivc.add_output('motor_weight', 40, units='kg')
                ivc.add_output('power_rating', 200, units='kW')
                self.add_subsystem('lcm', LiquidCooledMotor(num_nodes=num_nodes, quasi_steady=False), promotes_inputs=['*'])

        class TrajectoryPhase(PhaseGroup):
            "An OpenConcept Phase comprises part of a time-based TrajectoryGroup and always needs to have a 'duration' defined"
            def setup(self):
                self.add_subsystem('ivc', om.IndepVarComp('duration', val=20, units='min'), promotes_outputs=['duration'])
                self.add_subsystem('vm', VehicleModel(num_nodes=self.options['num_nodes']))

        class Trajectory(TrajectoryGroup):
            "An OpenConcept TrajectoryGroup consists of one or more phases that may be linked together. This will often be a top-level model"
            def setup(self):
                self.add_subsystem('phase1', TrajectoryPhase(num_nodes=nn)) 
                # self.add_subsystem('phase2', TrajectoryPhase(num_nodes=nn))
                # the link_phases directive ensures continuity of state variables across phase boundaries
                # self.link_phases(self.phase1, self.phase2)

        prob = om.Problem(Trajectory())
        prob.model.nonlinear_solver = om.NewtonSolver(iprint=2)
        prob.model.linear_solver = om.DirectSolver()
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.model.nonlinear_solver.options['maxiter'] = 20
        prob.model.nonlinear_solver.options['atol'] = 1e-6
        prob.model.nonlinear_solver.options['rtol'] = 1e-6    
        prob.setup(force_alloc_complex=True)
        # set the initial value of the state at the beginning of the TrajectoryGroup
        prob['phase1.vm.T_initial'] = 300.
        prob.run_model()
        # prob.model.list_outputs(print_arrays=True, units=True)
        # prob.model.list_inputs(print_arrays=True, units=True)
        
        return prob

    def test_vector(self):
        prob = self.generate_model(nn=11)
        prob.run_model()
        power_rating = 200000
        mdot_coolant = 3.0
        q_generated = power_rating * 0.05
        cp_coolant = 3801
        UA = 1100/650000*power_rating
        Cmin = cp_coolant * mdot_coolant # cp * mass flow rate
        NTU = UA/Cmin
        T_in = 298.15
        effectiveness = 1 - np.exp(-NTU)
        delta_T = q_generated / effectiveness / Cmin

        assert_near_equal(prob.get_val('phase1.vm.lcm.T', units='K'),
                          np.array([300.        , 319.02071102, 324.65196197, 327.0073297 ,
                           327.7046573 , 327.99632659, 328.08267788, 328.11879579,
                           328.12948882, 328.13396137, 328.1352855]), tolerance=1e-10)
        assert_near_equal(prob.get_val('phase1.vm.lcm.T_out', units='K'), 
                          np.array([298.2041044 , 298.76037687, 298.92506629, 298.99395048,
                           299.01434425, 299.0228743 , 299.0253997 , 299.02645599,
                           299.02676872, 299.02689952, 299.02693824]), tolerance=1e-10)
        assert_near_equal(prob.get_val('phase1.vm.lcm.T', units='K')[0],
                          np.array([300.]), tolerance=1e-10)
        # at the end of the period the unsteady value should be approx the quasi-steady value
        assert_near_equal(prob.get_val('phase1.vm.lcm.T', units='K')[-1],
                          np.array([T_in + delta_T]), tolerance=1e-5)
        assert_near_equal(prob.get_val('phase1.vm.lcm.T_out', units='K')[-1], 
                          np.array([T_in + q_generated / Cmin]), tolerance=1e-5)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class TestHose(unittest.TestCase):
    """
    Test the coolant hose component
    """

    def generate_model(self, nn):
        prob = om.Problem()

        hose_diam = 0.02
        hose_length = 16.
        hose_design_pressure = 1e6
        mdot_coolant = np.linspace(0.6, 1.2, nn)
        rho_coolant = 1020*np.ones((nn,))
        mu_coolant = 1.68e-3
        sigma = 2.07e6
        rho_hose = 1356.3

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('hose_diameter', val=hose_diam, units='m')
        ivc.add_output('hose_length', val=hose_length, units='m')
        ivc.add_output('hose_design_pressure', val=hose_design_pressure, units='Pa')
        ivc.add_output('mdot_coolant', val=mdot_coolant, units='kg/s')
        ivc.add_output('rho_coolant', val=rho_coolant, units='kg/m**3')
        ivc.add_output('mu_coolant', val=mu_coolant, units='kg/m/s')
        prob.model.add_subsystem('hose', SimpleHose(num_nodes=nn), promotes_inputs=['*'])
        prob.setup(check=True, force_alloc_complex=True)
        
        xs_area = np.pi * (hose_diam / 2 )**2
        U = mdot_coolant / rho_coolant / xs_area
        Redh = rho_coolant * U * hose_diam / mu_coolant
        f = 0.3164 * Redh ** (-1/4)
        dp = f * rho_coolant / 2 * hose_length * U ** 2 / hose_diam

        wall_thickness = hose_design_pressure * (hose_diam / 2) / sigma
        hose_weight = wall_thickness * np.pi * (hose_diam + wall_thickness) * rho_hose * hose_length
        fluid_weight = xs_area * rho_coolant[0] * hose_length
        return prob, dp, (hose_weight + fluid_weight)

    def test_scalar(self):
        prob, dp, weight = self.generate_model(nn=1)
        prob.run_model()
        assert_near_equal(prob.get_val('hose.delta_p', units='Pa'),
                          dp, tolerance=1e-10)
        assert_near_equal(prob.get_val('hose.component_weight', units='kg'),
                          weight, tolerance=1e-10)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_vector(self):
        prob, dp, weight = self.generate_model(nn=11)
        prob.run_model()
        assert_near_equal(prob.get_val('hose.delta_p', units='Pa'),
                          dp, tolerance=1e-10)
        assert_near_equal(prob.get_val('hose.component_weight', units='kg'),
                          weight, tolerance=1e-10)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)   

class TestPump(unittest.TestCase):
    """
    Test the coolant pump component
    """

    def generate_model(self, nn):
        prob = om.Problem()

        efficiency = 0.35
        spec_power = 1 / 450
        rho_coolant = 1020*np.ones(nn)
        mdot_coolant = np.linspace(0.6, 1.2, nn)
        delta_p = np.linspace(2e4, 4e4, nn)
        power_rating = 1000
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('power_rating', val=power_rating, units='W')
        ivc.add_output('delta_p', val=delta_p, units='Pa')
        ivc.add_output('mdot_coolant', val=mdot_coolant, units='kg/s')
        ivc.add_output('rho_coolant', val=rho_coolant, units='kg/m**3')
        prob.model.add_subsystem('pump', SimplePump(num_nodes=nn), promotes_inputs=['*'])
        prob.setup(check=True, force_alloc_complex=True)
        
        fluid_power = (mdot_coolant / rho_coolant) * delta_p
        weight = power_rating * spec_power
        elec_load = fluid_power / efficiency
        margin = elec_load / power_rating

        return prob, elec_load, weight, margin

    def test_scalar(self):
        prob, elec_load, weight, margin = self.generate_model(nn=1)
        prob.run_model()
        assert_near_equal(prob.get_val('pump.elec_load', units='W'),
                          elec_load, tolerance=1e-10)
        assert_near_equal(prob.get_val('pump.component_weight', units='kg'),
                          weight, tolerance=1e-10)
        assert_near_equal(prob.get_val('pump.component_sizing_margin', units=None),
                          margin, tolerance=1e-10)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_scalar(self):
        prob, elec_load, weight, margin = self.generate_model(nn=11)
        prob.run_model()
        assert_near_equal(prob.get_val('pump.elec_load', units='W'),
                          elec_load, tolerance=1e-10)
        assert_near_equal(prob.get_val('pump.component_weight', units='kg'),
                          weight, tolerance=1e-10)
        assert_near_equal(prob.get_val('pump.component_sizing_margin', units=None),
                          margin, tolerance=1e-10)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

if __name__ == "__main__":
    unittest.main()


