from __future__ import division
import unittest
import os
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem
import openconcept
from openconcept.components.cfm56 import CFM56

# Skip these test cases if the cached surrogate files don't exist
file_root = openconcept.__path__[0] + r'/components/empirical_data/cfm56/'
cached_thrust = os.path.exists(file_root + 'cfm56thrust.pkl')
cached_fuelburn = os.path.exists(file_root + 'cfm56fuelburn.pkl')
cached_T4 = os.path.exists(file_root + 'cfm56T4.pkl')
skip_tests = True
if cached_thrust and cached_fuelburn and cached_T4:
    skip_tests = False

@unittest.skipIf(skip_tests, "CFM56 surrogate model has not been trained (cached data not found), so skipping CFM56 tests")
class CFM56TestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model = CFM56()

        p.setup(force_alloc_complex=True)

        p.set_val('throttle', 0.5)
        p.set_val('fltcond|h', 10e3, units='ft')
        p.set_val('fltcond|M', 0.5)

        p.run_model()

        assert_near_equal(p.get_val('thrust', units='lbf'), 7050.74007329*np.ones(1), tolerance=1e-10)
        assert_near_equal(p.get_val('fuel_flow', units='kg/s'), .50273837866*np.ones(1), tolerance=1e-10)
        assert_near_equal(p.get_val('T4', units='degK'), 1432.06790075*np.ones(1), tolerance=1e-10)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model = CFM56(num_nodes=nn)

        p.setup(force_alloc_complex=True)

        p.set_val('throttle', np.linspace(0.0001, 1., nn))
        p.set_val('fltcond|h', np.linspace(0, 40e3, nn), units='ft')
        p.set_val('fltcond|M', np.linspace(0.1, 0.9, nn))

        p.run_model()

        assert_near_equal(p.get_val('thrust', units='lbf'), np.array([1463.81336747, 3961.41898067, 5278.43601914,
                          5441.45056659, 6478.26284602]), tolerance=1e-10)
        assert_near_equal(p.get_val('fuel_flow', units='kg/s'), 1e-3*np.array([170.25667238, 254.96500022,
                          357.45651444, 405.72535156, 492.41365636]), tolerance=1e-10)
        assert_near_equal(p.get_val('T4', units='degK'), np.array([1005.34531262, 1207.57602501,
                          1381.9478859, 1508.07883526, 1665.04281926]), tolerance=1e-10)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


if __name__=="__main__":
    unittest.main()
