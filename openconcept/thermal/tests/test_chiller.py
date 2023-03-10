import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.thermal.chiller import (
    LinearSelector,
    COPHeatPump,
    HeatPumpWeight,
    HeatPumpWithIntegratedCoolantLoop,
    COPExplicit,
)


class LinearSelectorTestCase(unittest.TestCase):
    def test_bypass(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model.add_subsystem("comp", LinearSelector(), promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p.get_val("T_out_cold", units="K"), p.get_val("T_in_hot", units="K"))
        assert_near_equal(p.get_val("T_out_hot", units="K"), p.get_val("T_in_cold", units="K"))
        assert_near_equal(p.get_val("elec_load", units="W"), np.zeros(1))
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_no_bypass(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model.add_subsystem("comp", LinearSelector(), promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("bypass", np.zeros(1))
        p.run_model()

        assert_near_equal(p.get_val("T_out_cold", units="K"), p.get_val("T_out_refrig_cold", units="K"))
        assert_near_equal(p.get_val("T_out_hot", units="K"), p.get_val("T_out_refrig_hot", units="K"))
        assert_near_equal(p.get_val("elec_load", units="W"), p.get_val("elec_load", units="W"))
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_vectorized(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model.add_subsystem("comp", LinearSelector(num_nodes=3), promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("bypass", np.array([0.0, 0.5, 1.0]))
        p.set_val("T_in_cold", np.array([295.0, 300.0, 305.0]), units="K")
        p.set_val("T_in_hot", np.array([295.0, 290.0, 285.0]), units="K")
        p.set_val("T_out_refrig_cold", np.array([[250.0, 260.0, 270.0]]), units="K")
        p.set_val("T_out_refrig_hot", np.array([[350.0, 360.0, 370.0]]), units="K")
        p.set_val("power_rating", 100.0, units="W")
        p.run_model()

        assert_near_equal(p.get_val("T_out_cold", units="K"), np.array([250.0, 275.0, 285.0]))
        assert_near_equal(p.get_val("T_out_hot", units="K"), np.array([350.0, 330.0, 305.0]))
        assert_near_equal(p.get_val("elec_load", units="W"), np.array([100.0, 50.0, 0.0]) / 0.95)
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class COPHeatPumpTestCase(unittest.TestCase):
    def test_single(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model.add_subsystem("comp", COPHeatPump(), promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("COP", np.ones(1))
        p.set_val("power_rating", 1.0, units="kW")
        p.run_model()

        assert_near_equal(p.get_val("q_in_1", units="W"), np.array([-1000.0]))
        assert_near_equal(p.get_val("q_in_2", units="W"), np.array([2000.0]))
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_vectorized(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model.add_subsystem("comp", COPHeatPump(num_nodes=3), promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("COP", np.array([1.0, 1.5, 2.0]))
        p.set_val("power_rating", 1.0, units="kW")
        p.run_model()

        assert_near_equal(p.get_val("q_in_1", units="W"), np.array([-1000.0, -1500.0, -2000.0]))
        assert_near_equal(p.get_val("q_in_2", units="W"), np.array([2000.0, 2500.0, 3000.0]))
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class HeatPumpWeightTestCase(unittest.TestCase):
    def test_single(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model.add_subsystem("comp", HeatPumpWeight(), promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("power_rating", 1.0, units="kW")
        p.set_val("specific_power", 200.0, units="W/kg")
        p.run_model()

        assert_near_equal(p.get_val("component_weight", units="kg"), np.array([5.0]))
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class HeatPumpWithIntegratedCoolantLoopTestCase(unittest.TestCase):
    """
    Test the convergence of the HeatPumpWithIntegratedCoolantLoop Group
    """

    def test_no_bypass(self):
        # Set up the heat pump problem with 11 evaluation points
        nn = 4
        p = Problem()
        p.model = HeatPumpWithIntegratedCoolantLoop(num_nodes=nn)
        p.model.set_input_defaults("power_rating", val=1.0, units="kW")
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options["solve_subsystems"] = True
        p.setup()
        p.set_val("T_in_hot", 350.0, units="K")
        p.set_val("T_in_cold", 300.0, units="K")
        p.set_val("mdot_coolant", 1.0, units="kg/s")
        p.set_val("control.bypass_start", 0.0)
        p.set_val("control.bypass_end", 0.0)
        p.run_model()

        assert_near_equal(p.get_val("T_out_hot", units="K"), 350.87503524 * np.ones(nn), tolerance=1e-10)
        assert_near_equal(p.get_val("T_out_cold", units="K"), 299.38805342 * np.ones(nn), tolerance=1e-10)
        assert_near_equal(p.get_val("component_weight", units="kg"), np.array([5.0]), tolerance=1e-10)
        assert_near_equal(p.get_val("elec_load", units="W"), 1052.63157895 * np.ones(nn), tolerance=1e-10)

    def test_varying_bypass(self):
        nn = 4
        p = Problem()
        p.model = HeatPumpWithIntegratedCoolantLoop(num_nodes=nn)
        p.model.set_input_defaults("power_rating", val=1.0, units="kW")
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options["solve_subsystems"] = True
        p.setup()
        p.set_val("T_in_hot", 350.0, units="K")
        p.set_val("T_in_cold", 300.0, units="K")
        p.set_val("mdot_coolant", 1.0, units="kg/s")
        p.set_val("control.bypass_start", 0.0)
        p.set_val("control.bypass_end", 1.0)
        p.run_model()

        assert_near_equal(
            p.get_val("T_out_hot", units="K"),
            np.array([350.87503524, 333.91669016, 316.95834508, 300.0]),
            tolerance=1e-10,
        )
        assert_near_equal(
            p.get_val("T_out_cold", units="K"),
            np.array([299.38805342, 316.25870228, 333.12935114, 350.0]),
            tolerance=1e-10,
        )
        assert_near_equal(p.get_val("component_weight", units="kg"), np.array([5.0]), tolerance=1e-10)
        assert_near_equal(
            p.get_val("elec_load", units="W"),
            np.array([1052.63157895, 701.75438596, 350.87719298, 0.0]),
            tolerance=1e-10,
        )


class COPExplicitTestCase(unittest.TestCase):
    def test_single(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model.add_subsystem("comp", COPExplicit(), promotes=["*"])
        p.setup()
        p.set_val("T_c", 300.0 * np.ones(1), units="K")
        p.set_val("T_h", 400.0 * np.ones(1), units="K")
        p.set_val("eff_factor", 0.4)
        p.run_model()

        assert_near_equal(p.get_val("COP"), np.array([1.20001629]), tolerance=1e-8)

    def test_vectorized(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model.add_subsystem("comp", COPExplicit(num_nodes=4), promotes=["*"])
        p.setup()
        p.set_val("T_c", np.array([100.0, 110.0, 120.0, 130.0]), units="K")
        p.set_val("T_h", np.array([200.0, 190.0, 180.0, 170.0]), units="K")
        p.set_val("eff_factor", 0.4)
        p.run_model()

        assert_near_equal(p.get_val("COP"), np.array([0.40000535, 0.5500066, 0.80000934, 1.30001871]), tolerance=1e-8)
