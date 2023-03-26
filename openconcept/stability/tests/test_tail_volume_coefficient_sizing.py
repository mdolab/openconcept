import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.stability import HStabVolumeCoefficientSizing, VStabVolumeCoefficientSizing


class HStabVolumeCoefficientSizingTestCase(unittest.TestCase):
    def test_jet_transport(self):
        S = 10.1
        c = 0.8
        L = 7
        K = 1.0

        p = om.Problem()
        p.model.add_subsystem("comp", HStabVolumeCoefficientSizing(C_ht=K), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("ac|geom|wing|S_ref", S, units="ft**2")
        p.set_val("ac|geom|wing|MAC", c, units="ft")
        p.set_val("ac|geom|hstab|c4_to_wing_c4", L, units="ft")

        p.run_model()

        assert_near_equal(p.get_val("ac|geom|hstab|S_ref", units="ft**2"), K * c * S / L)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)

    def test_twin_turboprop(self):
        S = 6.1
        c = 0.5
        L = 7
        K = 0.9

        p = om.Problem()
        p.model.add_subsystem("comp", HStabVolumeCoefficientSizing(C_ht=K), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("ac|geom|wing|S_ref", S, units="ft**2")
        p.set_val("ac|geom|wing|MAC", c, units="ft")
        p.set_val("ac|geom|hstab|c4_to_wing_c4", L, units="ft")

        p.run_model()

        assert_near_equal(p.get_val("ac|geom|hstab|S_ref", units="ft**2"), K * c * S / L)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


class VStabVolumeCoefficientSizingTestCase(unittest.TestCase):
    def test_jet_transport(self):
        S = 10.1
        AR = 9
        L = 7
        K = 0.09

        p = om.Problem()
        p.model.add_subsystem("comp", VStabVolumeCoefficientSizing(C_vt=K), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("ac|geom|wing|S_ref", S, units="ft**2")
        p.set_val("ac|geom|wing|AR", AR)
        p.set_val("ac|geom|vstab|c4_to_wing_c4", L, units="ft")

        p.run_model()

        span = (AR * S) ** 0.5
        assert_near_equal(p.get_val("ac|geom|vstab|S_ref", units="ft**2"), K * span * S / L)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)

    def test_twin_turboprop(self):
        S = 8.1
        AR = 5
        L = 6
        K = 0.08

        p = om.Problem()
        p.model.add_subsystem("comp", VStabVolumeCoefficientSizing(C_vt=K), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("ac|geom|wing|S_ref", S, units="ft**2")
        p.set_val("ac|geom|wing|AR", AR)
        p.set_val("ac|geom|vstab|c4_to_wing_c4", L, units="ft")

        p.run_model()

        span = (AR * S) ** 0.5
        assert_near_equal(p.get_val("ac|geom|vstab|S_ref", units="ft**2"), K * span * S / L)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


if __name__ == "__main__":
    unittest.main()
