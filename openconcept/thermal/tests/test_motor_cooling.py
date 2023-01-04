import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import numpy as np
from openconcept.thermal import LiquidCooledMotor


class QuasiSteadyMotorCoolingTestCase(unittest.TestCase):
    """
    Test the liquid cooled motor in quasi-steady (massless) mode
    """

    def generate_model(self, nn):
        prob = om.Problem()
        ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes_outputs=["*"])
        ivc.add_output("q_in", val=np.ones((nn,)) * 10000, units="W")
        ivc.add_output("T_in", 25.0 * np.ones((nn,)), units="degC")
        ivc.add_output("mdot_coolant", 3.0 * np.ones((nn,)), units="kg/s")
        ivc.add_output("motor_weight", 40, units="kg")
        ivc.add_output("power_rating", 200, units="kW")
        prob.model.add_subsystem("lcm", LiquidCooledMotor(num_nodes=nn, quasi_steady=True), promotes_inputs=["*"])
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
        UA = 1100 / 650000 * power_rating
        Cmin = cp_coolant * mdot_coolant  # cp * mass flow rate
        NTU = UA / Cmin
        T_in = 298.15
        effectiveness = 1 - np.exp(-NTU)
        delta_T = q_generated / effectiveness / Cmin
        assert_near_equal(prob.get_val("lcm.dTdt"), 0.0, tolerance=1e-14)
        assert_near_equal(prob.get_val("lcm.T", units="K"), T_in + delta_T, tolerance=1e-10)
        assert_near_equal(prob.get_val("lcm.T_out", units="K"), T_in + q_generated / Cmin, tolerance=1e-10)
        partials = prob.check_partials(method="cs", compact_print=True)
        # prob.model.list_outputs(print_arrays=True, units=True)
        assert_check_partials(partials)

    def test_vector(self):
        prob = self.generate_model(nn=11)
        prob.run_model()
        power_rating = 200000
        mdot_coolant = 3.0
        q_generated = power_rating * 0.05
        cp_coolant = 3801
        UA = 1100 / 650000 * power_rating
        Cmin = cp_coolant * mdot_coolant  # cp * mass flow rate
        NTU = UA / Cmin
        T_in = 298.15
        effectiveness = 1 - np.exp(-NTU)
        delta_T = q_generated / effectiveness / Cmin
        assert_near_equal(prob.get_val("lcm.dTdt"), np.zeros((11,)), tolerance=1e-14)
        assert_near_equal(prob.get_val("lcm.T", units="K"), np.ones((11,)) * (T_in + delta_T), tolerance=1e-10)
        assert_near_equal(
            prob.get_val("lcm.T_out", units="K"), np.ones((11,)) * (T_in + q_generated / Cmin), tolerance=1e-10
        )
        # prob.model.list_outputs(print_arrays=True, units='True')
        partials = prob.check_partials(method="cs", compact_print=True)
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
                self.options.declare("num_nodes", default=11)

            def setup(self):
                num_nodes = self.options["num_nodes"]
                ivc = self.add_subsystem("ivc", om.IndepVarComp(), promotes_outputs=["*"])
                ivc.add_output("q_in", val=np.ones((num_nodes,)) * 10000, units="W")
                ivc.add_output("T_in", 25.0 * np.ones((num_nodes,)), units="degC")
                ivc.add_output("mdot_coolant", 3.0 * np.ones((num_nodes,)), units="kg/s")
                ivc.add_output("motor_weight", 40, units="kg")
                ivc.add_output("power_rating", 200, units="kW")
                self.add_subsystem(
                    "lcm", LiquidCooledMotor(num_nodes=num_nodes, quasi_steady=False), promotes_inputs=["*"]
                )

        class TrajectoryPhase(PhaseGroup):
            "An OpenConcept Phase comprises part of a time-based TrajectoryGroup and always needs to have a 'duration' defined"

            def setup(self):
                self.add_subsystem(
                    "ivc", om.IndepVarComp("duration", val=20, units="min"), promotes_outputs=["duration"]
                )
                self.add_subsystem("vm", VehicleModel(num_nodes=self.options["num_nodes"]))

        class Trajectory(TrajectoryGroup):
            "An OpenConcept TrajectoryGroup consists of one or more phases that may be linked together. This will often be a top-level model"

            def setup(self):
                self.add_subsystem("phase1", TrajectoryPhase(num_nodes=nn))
                # self.add_subsystem('phase2', TrajectoryPhase(num_nodes=nn))
                # the link_phases directive ensures continuity of state variables across phase boundaries
                # self.link_phases(self.phase1, self.phase2)

        prob = om.Problem(Trajectory())
        prob.model.nonlinear_solver = om.NewtonSolver(iprint=2)
        prob.model.linear_solver = om.DirectSolver()
        prob.model.nonlinear_solver.options["solve_subsystems"] = True
        prob.model.nonlinear_solver.options["maxiter"] = 20
        prob.model.nonlinear_solver.options["atol"] = 1e-6
        prob.model.nonlinear_solver.options["rtol"] = 1e-6
        prob.setup(force_alloc_complex=True)
        # set the initial value of the state at the beginning of the TrajectoryGroup
        prob["phase1.vm.T_initial"] = 300.0
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
        UA = 1100 / 650000 * power_rating
        Cmin = cp_coolant * mdot_coolant  # cp * mass flow rate
        NTU = UA / Cmin
        T_in = 298.15
        effectiveness = 1 - np.exp(-NTU)
        delta_T = q_generated / effectiveness / Cmin

        assert_near_equal(
            prob.get_val("phase1.vm.lcm.T", units="K"),
            np.array(
                [
                    300.0,
                    319.02071102,
                    324.65196197,
                    327.0073297,
                    327.7046573,
                    327.99632659,
                    328.08267788,
                    328.11879579,
                    328.12948882,
                    328.13396137,
                    328.1352855,
                ]
            ),
            tolerance=1e-10,
        )
        assert_near_equal(
            prob.get_val("phase1.vm.lcm.T_out", units="K"),
            np.array(
                [
                    298.2041044,
                    298.76037687,
                    298.92506629,
                    298.99395048,
                    299.01434425,
                    299.0228743,
                    299.0253997,
                    299.02645599,
                    299.02676872,
                    299.02689952,
                    299.02693824,
                ]
            ),
            tolerance=1e-10,
        )
        assert_near_equal(prob.get_val("phase1.vm.lcm.T", units="K")[0], np.array([300.0]), tolerance=1e-10)
        # at the end of the period the unsteady value should be approx the quasi-steady value
        assert_near_equal(prob.get_val("phase1.vm.lcm.T", units="K")[-1], np.array([T_in + delta_T]), tolerance=1e-5)
        assert_near_equal(
            prob.get_val("phase1.vm.lcm.T_out", units="K")[-1], np.array([T_in + q_generated / Cmin]), tolerance=1e-5
        )
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)
