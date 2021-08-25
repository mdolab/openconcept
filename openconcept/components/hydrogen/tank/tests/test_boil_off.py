from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.components.hydrogen.tank.boil_off import *

class SimpleBoilOffTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = SimpleBoilOff()
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('m_boil_off', units='kg/s'), 2.239180280883e-04, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = SimpleBoilOff(num_nodes=nn)
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('m_boil_off', units='kg/s'),
                          2.239180280883e-04*np.ones(nn), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

if __name__ == "__main__":
    unittest.main()
