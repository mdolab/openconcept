import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.geometry import WingMACTrapezoidal, WingSpan

class WingMACTrapezoidalTestCase(unittest.TestCase):
    def test_rectangular(self):
        """
        Test a rectangular wing.
        """
        p = om.Problem()
        p.model.add_subsystem("comp", WingMACTrapezoidal(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        b = 10
        c = 1

        p.set_val("S_ref", b * c, units="m**2")
        p.set_val("AR", b / c)
        p.set_val("taper", 1.0)
        
        p.run_model()

        assert_near_equal(p.get_val("MAC", units="m"), c)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)

    def test_tapered(self):
        """
        Test a tapered wing.
        """
        p = om.Problem()
        p.model.add_subsystem("comp", WingMACTrapezoidal(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S_ref", 10, units="m**2")
        p.set_val("AR", 10)
        p.set_val("taper", 0.3)
        
        p.run_model()

        assert_near_equal(p.get_val("MAC", units="m"), 1.09664694, tolerance=1e-8)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


class WingSpanTestCase(unittest.TestCase):
    def test(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingSpan(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S_ref", 10, units="m**2")
        p.set_val("AR", 2.5)
        
        p.run_model()

        assert_near_equal(p.get_val("span", units="m"), 5.0)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


if __name__=="__main__":
    unittest.main()
