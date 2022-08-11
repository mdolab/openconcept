import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.thermal import (
    PerfectHeatTransferComp,
    ThermalComponentWithMass,
    ConstantSurfaceTemperatureColdPlate_NTU,
    LiquidCooledComp,
    CoolantReservoirRate,
    CoolantReservoir,
)


class PerfectHeatTransferCompTestCase(unittest.TestCase):
    """
    Test the PerfectHeatTransferComp component
    """

    def test_comp(self):
        num_nodes = 3
        prob = Problem()
        prob.model.add_subsystem("test", PerfectHeatTransferComp(num_nodes=num_nodes), promotes=["*"])
        prob.setup(check=True, force_alloc_complex=True)

        # Set the values
        prob["T_in"] = np.array([300.0, 350.0, 400.0])
        prob["q"] = np.array([10000.0, 0.0, -10000.0])
        prob["mdot_coolant"] = np.array([1.0, 1.0, 1.0])

        prob.run_model()

        dT_coolant = 10000.0 / 3801.0

        assert_near_equal(prob["T_out"], np.array([300 + dT_coolant, 350.0, 400 - dT_coolant]))
        assert_near_equal(prob["T_average"], np.array([300 + dT_coolant / 2, 350.0, 400 - dT_coolant / 2]))

        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class ThermalComponentWithMassTestCase(unittest.TestCase):
    """
    Test the ThermalComponentWithMass component
    """

    def test_comp(self):
        num_nodes = 3
        prob = Problem()
        prob.model.add_subsystem("test", ThermalComponentWithMass(num_nodes=num_nodes), promotes=["*"])
        prob.setup(check=True, force_alloc_complex=True)

        # Set the values
        prob.set_val("q_in", np.array([10.0, 14.0, 4.0]), units="kW")
        prob.set_val("q_out", np.array([9.0, 14.0, 12.0]), units="kW")
        prob.set_val("mass", 7.0, units="kg")

        prob.run_model()

        assert_near_equal(
            prob.get_val("dTdt", units="K/s"), np.array([0.1551109043, 0.0, -1.2408872344]), tolerance=1e-9
        )

        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class ColdPlateTestCase(unittest.TestCase):
    """
    Test the ConstantSurfaceTemperatureColdPlate_NTU component
    """

    def test_comp(self):
        num_nodes = 3
        prob = Problem()
        prob.model.add_subsystem("test", ConstantSurfaceTemperatureColdPlate_NTU(num_nodes=num_nodes), promotes=["*"])
        prob.setup(check=True, force_alloc_complex=True)

        # Set the values
        prob.set_val("T_in", np.array([300.0, 350.0, 290.0]), units="K")
        prob.set_val("T_surface", np.array([350.0, 500.0, 250.0]), units="K")
        prob.set_val("mdot_coolant", np.array([9.0, 14.0, 12.0]), units="kg/s")
        prob.set_val("channel_length", 7.0, units="mm")
        prob.set_val("channel_width", 1.0, units="mm")
        prob.set_val("channel_height", 0.5, units="mm")
        prob.set_val("n_parallel", 5)

        prob.run_model()

        assert_near_equal(
            prob.get_val("q", units="W"), np.array([24.04771845, 72.14333648, -19.23820857]), tolerance=1e-9
        )
        assert_near_equal(
            prob.get_val("T_out", units="K"), np.array([300.00070296, 350.00135572, 289.99957822]), tolerance=1e-9
        )


class LiquidCooledCompTestCase(unittest.TestCase):
    """
    Test the LiquidCooledComp component
    """

    def test_comp(self):
        num_nodes = 3
        prob = Problem()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver.options["solve_subsystems"] = True
        prob.model.add_subsystem("test", LiquidCooledComp(num_nodes=num_nodes), promotes=["*"])
        prob.setup(check=True, force_alloc_complex=True)

        # Set the values
        prob.set_val("q_in", np.array([1000.0, 1400.0, 4000.0]), units="W")
        prob.set_val("mdot_coolant", np.array([9.0, 14.0, 12.0]), units="kg/s")
        prob.set_val("T_in", np.array([300.0, 350.0, 290.0]), units="K")
        prob.set_val("mass", 10.0, units="kg")
        prob.set_val("T_initial", 400.0, units="K")
        prob.set_val("duration", 2.0, units="min")
        prob.set_val("channel_length", 7.0, units="mm")
        prob.set_val("channel_width", 1.0, units="mm")
        prob.set_val("channel_height", 0.5, units="mm")
        prob.set_val("n_parallel", 5)

        prob.run_model()

        assert_near_equal(prob.get_val("T", units="K"), np.array([400.0, 406.40945984, 422.53992837]), tolerance=1e-9)
        assert_near_equal(
            prob.get_val("T_out", units="K"), np.array([300.00140593, 350.00050984, 290.00139757]), tolerance=1e-9
        )

        partials = prob.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)


class CoolantReservoirRateTestCase(unittest.TestCase):
    """
    Test the CoolantReservoirRate component
    """

    def test_comp(self):
        num_nodes = 3
        prob = Problem()
        prob.model.add_subsystem("test", CoolantReservoirRate(num_nodes=num_nodes), promotes=["*"])
        prob.setup(check=True, force_alloc_complex=True)

        # Set the values
        prob.set_val("T_in", np.array([300.0, 350.0, 290.0]), units="K")
        prob.set_val("T_out", np.array([350.0, 340.0, 250.0]), units="K")
        prob.set_val("mdot_coolant", np.array([9.0, 14.0, 12.0]), units="kg/s")
        prob.set_val("mass", 7.0, units="kg")

        prob.run_model()

        assert_near_equal(
            prob.get_val("dTdt", units="K/s"), np.array([-64.28571429, 20.0, 68.57142857]), tolerance=1e-9
        )

        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class ReservoirTestCase(unittest.TestCase):
    """
    Test the CoolantReservoir component
    """

    def test_comp(self):
        num_nodes = 3
        prob = Problem()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver.options["solve_subsystems"] = True
        prob.model.add_subsystem("test", CoolantReservoir(num_nodes=num_nodes), promotes=["*"])
        prob.setup(check=True, force_alloc_complex=True)

        # Set the values
        prob.set_val("mdot_coolant", np.array([9.0, 14.0, 12.0]), units="kg/s")
        prob.set_val("T_in", np.array([300.0, 350.0, 290.0]), units="K")
        prob.set_val("mass", 10.0, units="kg")
        prob.set_val("T_initial", 400.0, units="K")
        prob.set_val("duration", 20.0, units="min")

        prob.run_model()

        assert_near_equal(
            prob.get_val("T_out", units="K"), np.array([400.0, 317.9653263, 364.64246726]), tolerance=1e-9
        )

        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
