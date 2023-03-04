from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.utilities.constants import UNIVERSAL_GAS_CONST, MOLEC_WEIGHT_H2
import openconcept.energy_storage.hydrogen.H2_properties as H2_prop
from openconcept.energy_storage.hydrogen.boil_off import *


class BoilOffTestCase(unittest.TestCase):
    def test_integrated(self):
        """
        A regression test for the fully integrated boil-off model.
        """
        nn = 11
        p = om.Problem()
        p.model.add_subsystem(
            "model",
            BoilOff(num_nodes=nn, fill_level_init=0.9, ullage_T_init=21, ullage_P_init=1.2e5, liquid_T_init=20),
            promotes=["*"],
        )

        p.setup(force_alloc_complex=True)

        p.set_val("integ.duration", 3, units="h")
        p.set_val("radius", 0.7, units="m")
        p.set_val("length", 0.3, units="m")
        p.set_val("m_dot_gas_in", 0.7, units="kg/h")
        p.set_val("m_dot_liq_in", 2.0, units="kg/h")
        p.set_val("m_dot_gas_out", 0.2, units="kg/h")
        p.set_val("m_dot_liq_out", 30.0, units="kg/h")
        p.set_val("Q_dot", 50, units="W")

        p.run_model()

        assert_near_equal(
            p.get_val("m_gas", units="kg"),
            np.array(
                [
                    0.26303704192536154,
                    0.42824983628333974,
                    0.6063360966425474,
                    0.7908946454675435,
                    0.979294083766866,
                    1.1715455673363493,
                    1.368037081543731,
                    1.5686835254461788,
                    1.773120630979943,
                    1.9809979081536,
                    2.192022628679719,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("m_liq", units="kg"),
            np.array(
                [
                    121.60657623874415,
                    113.19136344438617,
                    104.76327718402698,
                    96.32871863520197,
                    87.89031919690265,
                    79.44806771333316,
                    71.00157619912578,
                    62.55092975522333,
                    54.096492649689566,
                    45.63861537251589,
                    37.17759065198979,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_gas", units="K"),
            np.array(
                [
                    21.0,
                    22.53189850495305,
                    22.94735860229283,
                    23.068666630450227,
                    23.216090257538593,
                    23.403700047089767,
                    23.591580229207803,
                    23.769746854896383,
                    23.94896718089159,
                    24.14219628817064,
                    24.360037880890946,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_liq", units="K"),
            np.array(
                [
                    20.0,
                    20.027969060527173,
                    20.05567182732004,
                    20.08334541703263,
                    20.11120025717257,
                    20.13942293108154,
                    20.168193926313638,
                    20.197706787043465,
                    20.228186606494212,
                    20.25991173996442,
                    20.293244426674327,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("P_gas", units="Pa"),
            np.array(
                [
                    119999.99999999999,
                    129165.17596077801,
                    134521.05033238672,
                    138018.54598992236,
                    141236.2208481784,
                    144474.95369406184,
                    147632.35781236002,
                    150674.66783696116,
                    153661.93549340914,
                    156674.53736317036,
                    159785.71493562404,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("fill_level"),
            np.array(
                [
                    0.9,
                    0.8377089420319701,
                    0.7753011157497871,
                    0.7128236179012597,
                    0.6502955446437099,
                    0.5877162818761296,
                    0.5250822680760078,
                    0.4623932798529712,
                    0.39965097156234897,
                    0.3368566604342073,
                    0.2740109310097058,
                ]
            ),
            tolerance=1e-9,
        )


class LiquidHeightTestCase(unittest.TestCase):
    def setUp(self):
        self.nn = nn = 7
        self.p = p = om.Problem()
        p.model.add_subsystem("model", LiquidHeight(num_nodes=nn), promotes=["*"])
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, iprint=2, err_on_non_converge=True, maxiter=20
        )
        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=True)

    def test_simple(self):
        r = 0.6
        L = 0.3

        # Define height and work backwards to fill level so we
        # can recompute it and check against the original height
        off = 5.0  # deg
        theta = np.linspace(off, 2 * np.pi - off, self.nn)
        h = r * (1 - np.cos(theta / 2))
        V_fill = r**2 / 2 * (theta - np.sin(theta)) * L + np.pi * h**2 / 3 * (3 * r - h)
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        fill = V_fill / V_tank

        self.p.set_val("fill_level", fill)
        self.p.set_val("radius", r, units="m")
        self.p.set_val("length", L, units="m")

        self.p.run_model()

        assert_near_equal(self.p.get_val("h_liq_frac"), h / (2 * r), tolerance=1e-8)

    def test_derivatives(self):
        self.p.set_val("fill_level", np.linspace(0.1, 0.9, self.nn))
        self.p.set_val("radius", 0.6, units="m")
        self.p.set_val("length", 1.2, units="m")

        self.p.run_model()

        partials = self.p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class BoilOffGeometryTestCase(unittest.TestCase):
    def setup_model(self, nn):
        self.p = p = om.Problem()
        comp = p.model.add_subsystem("model", BoilOffGeometry(num_nodes=nn), promotes=["*"])
        p.setup(force_alloc_complex=True)
        comp.adjust_h_liq_frac = False

        self.r = 0.7
        self.L = 0.3

        self.p.set_val("radius", self.r, units="m")
        self.p.set_val("length", self.L, units="m")

    def test_empty(self):
        self.setup_model(1)

        self.p.set_val("h_liq_frac", 0)

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), A_tank, tolerance=1e-8)

    def test_full(self):
        self.setup_model(1)

        self.p.set_val("h_liq_frac", 1.0)

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), A_tank, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), 0.0, tolerance=1e-8)

    def test_half(self):
        self.setup_model(1)

        self.p.set_val("h_liq_frac", 0.5)

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

        self.p.set_val("h_liq_frac", np.linspace(0, 1.0, nn))

        self.p.run_model()

        A_wet = np.array([0.0, 1.97920337, 3.73849526, 5.49778714, 7.47699052])
        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(
            self.p.get_val("A_interface", units="m**2"),
            np.array([0.0, 1.5182659697837129, 1.9593804002589983, 1.5182659697837138, 0.0]),
            tolerance=1e-8,
        )
        assert_near_equal(
            self.p.get_val("L_interface", units="m"),
            np.array([0.0, 1.212435565298214, 1.4, 1.2124355652982144, 0.0]),
            tolerance=1e-8,
        )
        assert_near_equal(self.p.get_val("A_wet", units="m**2"), A_wet, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2"), A_tank - A_wet, tolerance=1e-8)

    def test_derivatives(self):
        nn = 7
        self.setup_model(nn)

        off = 1e-6
        self.p.set_val("h_liq_frac", np.linspace(off, 1.0 - off, nn))

        partials = self.p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class BoilOffFillLevelCalcTestCase(unittest.TestCase):
    def setup_model(self, nn=1):
        self.p = p = om.Problem()
        p.model.add_subsystem("model", BoilOffFillLevelCalc(num_nodes=nn), promotes=["*"])
        p.setup(force_alloc_complex=True)

        self.r = 0.7
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


class InitialTankStateModificationTestCase(unittest.TestCase):
    def test_init_values(self):
        p = om.Problem()

        fill = 0.6
        T_liq = 20
        T_gas = 21
        P_gas = 2e5

        p.model.add_subsystem(
            "model",
            InitialTankStateModification(
                num_nodes=1,
                fill_level_init=fill,
                ullage_T_init=T_gas,
                ullage_P_init=P_gas,
                liquid_T_init=T_liq,
            ),
            promotes=["*"],
        )

        p.setup()

        r = 1.3  # m
        L = 0.7  # m

        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        # Set all inputs to zero so we can test the computed initial values
        p.set_val("delta_m_gas", 0.0, units="kg")
        p.set_val("delta_m_liq", 0.0, units="kg")
        p.set_val("delta_T_gas", 0.0, units="K")
        p.set_val("delta_T_liq", 0.0, units="K")
        p.set_val("delta_V_gas", 0.0, units="m**3")

        p.run_model()

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        V_gas = V_tank * (1 - fill)
        m_gas = P_gas * V_gas * MOLEC_WEIGHT_H2 / T_gas / UNIVERSAL_GAS_CONST
        m_liq = V_tank * fill * H2_prop.lh2_rho(T_liq)

        assert_near_equal(p.get_val("m_gas", units="kg"), m_gas, tolerance=1e-12)
        assert_near_equal(p.get_val("m_liq", units="kg"), m_liq, tolerance=1e-12)
        assert_near_equal(p.get_val("T_gas", units="K"), T_gas, tolerance=1e-12)
        assert_near_equal(p.get_val("T_liq", units="K"), T_liq, tolerance=1e-12)
        assert_near_equal(p.get_val("V_gas", units="m**3"), V_tank * (1 - fill), tolerance=1e-12)

    def test_vectorized(self):
        p = om.Problem()

        nn = 5
        fill = 0.2
        T_liq = 18
        T_gas = 22
        P_gas = 1.6e5

        p.model.add_subsystem(
            "model",
            InitialTankStateModification(
                num_nodes=nn,
                fill_level_init=fill,
                ullage_T_init=T_gas,
                ullage_P_init=P_gas,
                liquid_T_init=T_liq,
            ),
            promotes=["*"],
        )

        p.setup(force_alloc_complex=True)

        r = 0.7  # m
        L = 0.9  # m

        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        # Add some delta to see that it works properly
        val = np.linspace(-10, 10, nn)
        p.set_val("delta_m_gas", val, units="kg")
        p.set_val("delta_m_liq", val, units="kg")
        p.set_val("delta_T_gas", val, units="K")
        p.set_val("delta_T_liq", val, units="K")
        p.set_val("delta_V_gas", val, units="m**3")

        p.run_model()

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        V_gas = V_tank * (1 - fill)
        m_gas = P_gas * V_gas * MOLEC_WEIGHT_H2 / T_gas / UNIVERSAL_GAS_CONST
        m_liq = V_tank * fill * H2_prop.lh2_rho(T_liq)

        assert_near_equal(p.get_val("m_gas", units="kg"), m_gas + val, tolerance=1e-12)
        assert_near_equal(p.get_val("m_liq", units="kg"), m_liq + val, tolerance=1e-12)
        assert_near_equal(p.get_val("T_gas", units="K"), T_gas + val, tolerance=1e-12)
        assert_near_equal(p.get_val("T_liq", units="K"), T_liq + val, tolerance=1e-12)
        assert_near_equal(p.get_val("V_gas", units="m**3"), V_tank * (1 - fill) + val, tolerance=1e-12)

    def test_partials(self):
        nn = 5
        p = om.Problem()
        p.model.add_subsystem("model", InitialTankStateModification(num_nodes=nn), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("radius", 0.6, units="m")
        p.set_val("length", 1.3, units="m")

        # Add some delta to see that it works properly
        val = np.linspace(-10, 10, nn)
        p.set_val("delta_m_gas", val, units="kg")
        p.set_val("delta_m_liq", val, units="kg")
        p.set_val("delta_T_gas", val, units="K")
        p.set_val("delta_T_liq", val, units="K")
        p.set_val("delta_V_gas", val, units="m**3")

        p.run_model()

        partials = p.check_partials(method="cs")
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
