from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.energy_storage.hydrogen.boil_off import *


class LiquidHeightTestCase(unittest.TestCase):
    def setUp(self):
        self.nn = nn = 7
        self.p = p = om.Problem()
        p.model.add_subsystem("model", LiquidHeight(num_nodes=nn), promotes=["*"])
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, iprint=2, err_on_non_converge=True
        )
        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=True)

    def test_simple(self):
        r = 0.5
        L = 0.3

        # Define height and work backwards to fill level so we
        # can recompute it and check against the original height
        off = 1.0  # deg
        theta = np.linspace(off, 2 * np.pi - off, self.nn)
        h = r * (1 - np.cos(theta / 2))
        V_fill = r**2 / 2 * (theta - np.sin(theta)) * L + np.pi * h**2 / 3 * (3 * r - h)
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        fill = V_fill / V_tank

        self.p.set_val("fill_level", fill)
        self.p.set_val("radius", r, units="m")
        self.p.set_val("length", L, units="m")

        self.p.run_model()

        assert_near_equal(self.p.get_val("h_liq", units="m"), h, tolerance=1e-8)

    def test_derivatives(self):
        self.p.set_val("fill_level", np.linspace(0.1, 0.9, self.nn))
        self.p.set_val("radius", 0.5, units="m")
        self.p.set_val("length", 1.2, units="m")

        self.p.run_model()

        partials = self.p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class BoilOffGeometryTestCase(unittest.TestCase):
    def setup_model(self, nn):
        self.p = p = om.Problem()
        p.model.add_subsystem("model", BoilOffGeometry(num_nodes=nn), promotes=["*"])
        p.setup(force_alloc_complex=True)

        self.r = 0.5
        self.L = 0.3

        self.p.set_val("radius", self.r, units="m")
        self.p.set_val("length", self.L, units="m")

    def test_empty(self):
        self.setup_model(1)

        self.p.set_val("h_liq", 0, units="m")

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), A_tank, tolerance=1e-8)

    def test_full(self):
        self.setup_model(1)

        self.p.set_val("h_liq", 2 * self.r, units="m")

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), A_tank, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), 0.0, tolerance=1e-8)

    def test_half(self):
        self.setup_model(1)

        self.p.set_val("h_liq", self.r, units="m")

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(
            self.p.get_val("A_interface", units="m**2").item(),
            np.pi * self.r**2 + 2 * self.r * self.L,
            tolerance=1e-8,
        )
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 2 * self.r, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), A_tank / 2, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), A_tank / 2, tolerance=1e-8)

    def test_regression(self):
        nn = 5
        self.setup_model(nn)

        self.p.set_val("h_liq", np.linspace(0, 2 * self.r, nn), units="m")

        self.p.run_model()

        A_wet = np.array([0.0, 1.09955743, 2.04203522, 2.98451302, 4.08407045])
        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(
            self.p.get_val("A_interface", units="m**2"),
            np.array([0.0, 0.84885624, 1.08539816, 0.84885624, 0.0]),
            tolerance=1e-8,
        )
        assert_near_equal(
            self.p.get_val("L_interface", units="m"), np.array([0.0, 0.8660254, 1.0, 0.8660254, 0.0]), tolerance=1e-8
        )
        assert_near_equal(self.p.get_val("A_wet", units="m**2"), A_wet, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2"), A_tank - A_wet, tolerance=1e-8)

    def test_derivatives(self):
        nn = 7
        self.setup_model(nn)

        off = 1e-6
        self.p.set_val("h_liq", np.linspace(off, 2 * self.r - off, nn), units="m")

        partials = self.p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class BoilOffFillLevelCalcTestCase(unittest.TestCase):
    def setup_model(self, nn=1):
        self.p = p = om.Problem()
        p.model.add_subsystem("model", BoilOffFillLevelCalc(num_nodes=nn), promotes=["*"])
        p.setup(force_alloc_complex=True)

        self.r = 0.5
        self.L = 0.3

        self.p.set_val("radius", self.r, units="m")
        self.p.set_val("length", self.L, units="m")

    def test_fill_level(self):
        nn = 7
        self.setup_model(nn)

        r = self.r
        L = self.L
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        fill = np.linspace(0.01, 0.99, nn)
        V_gas = (1 - fill) * V_tank
        self.p.set_val("V_gas", V_gas, units="m**3")

        self.p.run_model()

        assert_near_equal(self.p.get_val("fill_level"), fill, tolerance=1e-10)

    def test_derivatives(self):
        nn = 7
        self.setup_model(nn)

        r = self.r
        L = self.L
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        V_gas = np.linspace(0.005, 0.995, nn) * V_tank
        self.p.set_val("V_gas", V_gas, units="m**3")

        partials = self.p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class LH2BoilOffODETestCase(unittest.TestCase):
    def test_mostly_empty(self):
        """
        Compare to values of the EBM model implementation from Eugina Mendez Ramos's thesis.
        """
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(), promotes=["*"])

        p.setup()

        p.set_val("m_gas", 2.0, units="kg")
        p.set_val("m_liq", 4.5161831120531115, units="kg")
        p.set_val("T_gas", 21, units="K")
        p.set_val("T_liq", 20, units="K")
        p.set_val("V_gas", 1.5, units="m**3")
        p.set_val("m_dot_gas_in", 0.0, units="kg/s")
        p.set_val("m_dot_gas_out", 0.0, units="kg/s")
        p.set_val("m_dot_liq_in", 0.0, units="kg/s")
        p.set_val("m_dot_liq_out", 0.0, units="kg/s")
        p.set_val("Q_dot", 50, units="W")
        p.set_val("A_interface", 0.3840421278000321, units="m**2")
        p.set_val("L_interface", 0.17481733751963005, units="m")
        p.set_val("A_wet", 0.6203054996729682, units="m**2")
        p.set_val("A_dry", 5.8941010268108265, units="m**2")

        p.run_model()

        assert_near_equal(p.get_val("m_dot_gas", units="kg/s"), 6.296280265373945e-07, tolerance=1e-12)
        assert_near_equal(p.get_val("m_dot_gas", units="kg/s"), -p.get_val("m_dot_liq", units="kg/s"), tolerance=1e-12)
        assert_near_equal(p.get_val("T_dot_gas", units="K/s"), 0.003488134556611412, tolerance=1e-12)
        assert_near_equal(p.get_val("T_dot_liq", units="K/s"), 0.00010642261101789976, tolerance=1e-12)
        assert_near_equal(p.get_val("V_dot_gas", units="m**3/s"), 8.846997848034505e-09, tolerance=1e-12)
        assert_near_equal(p.get_val("P_gas", units="Pa"), 115486.04083576404, tolerance=1e-12)

    def test_MHTB_mostly_full(self):
        """
        Compare to values of the EBM model implementation from Eugina Mendez Ramos's thesis.
        """
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(), promotes=["*"])

        p.setup()

        p.set_val("m_gas", 1.9411308219846288, units="kg")
        p.set_val("m_liq", 942.0670752834986, units="kg")
        p.set_val("T_gas", 21.239503179127798, units="K")
        p.set_val("T_liq", 20.708930544834377, units="K")
        p.set_val("V_gas", 1.4856099323616818, units="m**3")
        p.set_val("m_dot_gas_in", 0.0, units="kg/s")
        p.set_val("m_dot_gas_out", 0.0, units="kg/s")
        p.set_val("m_dot_liq_in", 0.0, units="kg/s")
        p.set_val("m_dot_liq_out", 0.0, units="kg/s")
        p.set_val("Q_dot", 51.4, units="W")
        p.set_val("A_interface", 4.601815537828035, units="m**2")
        p.set_val("L_interface", 0.6051453090136371, units="m")
        p.set_val("A_wet", 23.502425642397316, units="m**2")
        p.set_val("A_dry", 5.722240017621729, units="m**2")

        p.run_model()

        assert_near_equal(p.get_val("m_dot_gas", units="kg/s"), 1.8147216231146884e-05, tolerance=1e-12)
        assert_near_equal(p.get_val("m_dot_gas", units="kg/s"), -p.get_val("m_dot_liq", units="kg/s"), tolerance=1e-12)
        assert_near_equal(p.get_val("T_dot_gas", units="K/s"), 0.0002533535735070904, tolerance=1e-12)
        assert_near_equal(p.get_val("T_dot_liq", units="K/s"), 4.309889239802682e-06, tolerance=1e-12)
        assert_near_equal(p.get_val("V_dot_gas", units="m**3/s"), 2.575538811629458e-07, tolerance=1e-12)
        assert_near_equal(p.get_val("P_gas", units="Pa"), 114463.18507907697, tolerance=1e-12)

    def test_T_cutoff(self):
        # Minimum pressure of 100,000 Pa puts the saturation temperature below 21 K, so
        # the rate of change of the liquid temperature should be zeroed out
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(P_min=1e5), promotes=["*"])
        p.setup()
        p.set_val("T_liq", 21.0, units="K")
        p.run_model()
        assert_near_equal(p.get_val("T_dot_liq", units="K/s"), 0.0)  # cut off by the T max limiter

    def test_derivatives(self):
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("m_gas", 1.9411308219846288, units="kg")
        p.set_val("m_liq", 942.0670752834986, units="kg")
        p.set_val("T_gas", 21.239503179127798, units="K")
        p.set_val("T_liq", 20.708930544834377, units="K")
        p.set_val("V_gas", 1.4856099323616818, units="m**3")
        p.set_val("m_dot_gas_in", 0.0, units="kg/s")
        p.set_val("m_dot_gas_out", 0.0, units="kg/s")
        p.set_val("m_dot_liq_in", 0.0, units="kg/s")
        p.set_val("m_dot_liq_out", 0.0, units="kg/s")
        p.set_val("Q_dot", 51.4, units="W")
        p.set_val("A_interface", 4.601815537828035, units="m**2")
        p.set_val("L_interface", 0.6051453090136371, units="m")
        p.set_val("A_wet", 23.502425642397316, units="m**2")
        p.set_val("A_dry", 5.722240017621729, units="m**2")

        p.run_model()

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_derivatives_with_mass_flows(self):
        p = om.Problem()
        p.model.add_subsystem("model", LH2BoilOffODE(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("m_gas", 1.9411308219846288, units="kg")
        p.set_val("m_liq", 942.0670752834986, units="kg")
        p.set_val("T_gas", 21.239503179127798, units="K")
        p.set_val("T_liq", 20.708930544834377, units="K")
        p.set_val("V_gas", 1.4856099323616818, units="m**3")
        p.set_val("m_dot_gas_in", 0.1, units="kg/s")
        p.set_val("m_dot_gas_out", 0.2, units="kg/s")
        p.set_val("m_dot_liq_in", 0.05, units="kg/s")
        p.set_val("m_dot_liq_out", 0.5, units="kg/s")
        p.set_val("Q_dot", 51.4, units="W")
        p.set_val("A_interface", 4.601815537828035, units="m**2")
        p.set_val("L_interface", 0.6051453090136371, units="m")
        p.set_val("A_wet", 23.502425642397316, units="m**2")
        p.set_val("A_dry", 5.722240017621729, units="m**2")

        p.run_model()

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
