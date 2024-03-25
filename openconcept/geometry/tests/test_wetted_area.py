import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.geometry import CylinderSurfaceArea


class CylinderSurfaceAreaTestCase(unittest.TestCase):
    def test(self):
        p = om.Problem()
        p.model.add_subsystem("comp", CylinderSurfaceArea(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        L = 10
        D = 7

        p.set_val("L", L, units="m")
        p.set_val("D", D, units="m")

        p.run_model()

        assert_near_equal(p.get_val("A", units="m**2"), np.pi * L * D)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


if __name__ == "__main__":
    unittest.main()
