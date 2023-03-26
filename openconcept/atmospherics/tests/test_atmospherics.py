import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.atmospherics import ComputeAtmosphericProperties


class AtmosTestGroup(Group):
    """This computes pressure, temperature, and density for a given altitude at ISA condtions. Also true airspeed from equivalent ~ indicated airspeed"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]
        iv = self.add_subsystem("conditions", IndepVarComp())
        self.add_subsystem("atmos", ComputeAtmosphericProperties(num_nodes=nn), promotes_outputs=["*"])
        iv.add_output("h", val=np.linspace(0, 30000, nn), units="ft")
        iv.add_output("Ueas", val=np.ones(nn) * 120, units="kn")
        self.connect("conditions.h", "atmos.fltcond|h")
        self.connect("conditions.Ueas", "atmos.fltcond|Ueas")


class VectorAtmosTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(AtmosTestGroup(num_nodes=5))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_sea_level_and_30kft(self):
        # check conditions at sea level
        assert_near_equal(self.prob["fltcond|rho"][0], 1.225, tolerance=1e-4)
        assert_near_equal(self.prob["fltcond|p"][0], 101325, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|T"][0], 288.15, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|Utrue"][0], 61.7333, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|q"][0], 2334.2398, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|M"][0], 0.1814, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|a"][0], 340.294, tolerance=1e-3)

        # check conditions at 30kft (1976 standard atmosphere verified at https://www.digitaldutch.com/atmoscalc/)
        assert_near_equal(self.prob["fltcond|rho"][-1], 0.458312, tolerance=1e-4)
        assert_near_equal(self.prob["fltcond|p"][-1], 30089.6, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|T"][-1], 228.714, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|Utrue"][-1], 61.7333 * np.sqrt(1.225 / 0.458312), tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|q"][-1], 2334.2398, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|M"][-1], 0.3326, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|a"][-1], 303.2301, tolerance=1e-3)

    def test_ISA_temp_offset(self):
        self.prob.set_val("atmos.fltcond|TempIncrement", np.linspace(30, 15, 5), units="degC")
        self.prob.run_model()
        # check conditions at sea level
        assert_near_equal(self.prob["fltcond|rho"][0], 1.10949, tolerance=1e-4)
        assert_near_equal(self.prob["fltcond|p"][0], 101325, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|T"][0], 318.150, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|Utrue"][0], 64.8674, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|q"][0], 2334.2398, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|M"][0], 0.1814, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|a"][0], 357.5698, tolerance=1e-3)

        # #check conditions at 30kft (1976 standard atmosphere verified at https://www.digitaldutch.com/atmoscalc/)
        assert_near_equal(self.prob["fltcond|rho"][-1], 0.430104, tolerance=1e-4)
        assert_near_equal(self.prob["fltcond|p"][-1], 30089.6, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|T"][-1], 243.714, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|Utrue"][-1], 61.7333 * np.sqrt(1.225 / 0.430104), tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|q"][-1], 2334.2398, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|M"][-1], 61.7333 * np.sqrt(1.225 / 0.430104) / 312.957, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|a"][-1], 312.957, tolerance=1e-3)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


class ScalarAtmosTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(AtmosTestGroup(num_nodes=1))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_sea_level(self):
        assert_near_equal(self.prob["fltcond|rho"][0], 1.225, tolerance=1e-4)
        assert_near_equal(self.prob["fltcond|p"][0], 101325, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|T"][0], 288.15, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|Utrue"][0], 61.7333, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|q"][0], 2334.2398, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|M"][0], 0.1814, tolerance=1e-3)
        assert_near_equal(self.prob["fltcond|a"][0], 340.294, tolerance=1e-3)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
