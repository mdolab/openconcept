from __future__ import division
import unittest
import os
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem
import openconcept
from openconcept.components.N3 import N3, N3Hybrid

# Skip these test cases if the cached surrogate files don't exist
# N+3 hybrid
hybrid_file_root = openconcept.__path__[0] + r'/components/empirical_data/n+3_hybrid/'
hybrid_cached_thrust = os.path.exists(hybrid_file_root + r'/n3_hybrid_thrust_trained.zip')
hybrid_cached_fuelburn = os.path.exists(hybrid_file_root + r'n3_hybrid_fuelflow_trained.zip')
hybrid_cached_surge = os.path.exists(hybrid_file_root + 'n3_hybrid_smw_trained.zip')
hybrid_skip_tests = True
if hybrid_cached_thrust and hybrid_cached_fuelburn and hybrid_cached_surge:
    hybrid_skip_tests = False

# N+3
file_root = openconcept.__path__[0] + r'/components/empirical_data/n+3/'
cached_thrust = os.path.exists(file_root + r'/n3_thrust_trained.zip')
cached_fuelburn = os.path.exists(file_root + r'n3_fuelflow_trained.zip')
cached_T4 = os.path.exists(file_root + r'n3_smw_trained.zip')
skip_tests = True
if cached_thrust and cached_fuelburn and cached_T4:
    skip_tests = False

@unittest.skipIf(hybrid_skip_tests, "N+3 hybrid surrogate model has not been trained (cached data not found), so skipping N+3 hybrid tests")
class N3HybridTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model = N3Hybrid()

        p.setup(force_alloc_complex=True)

        p.set_val('throttle', 0.5)
        p.set_val('fltcond|h', 10e3, units='ft')
        p.set_val('fltcond|M', 0.5)
        p.set_val('hybrid_power', 200., units='kW')

        p.run_model()

        assert_near_equal(p.get_val('thrust', units='lbf'), 6965.43502572*np.ones(1), tolerance=1e-10)
        assert_near_equal(p.get_val('fuel_flow', units='kg/s'), .34333923275*np.ones(1), tolerance=1e-10)
        assert_near_equal(p.get_val('surge_margin'), 17.49872079*np.ones(1), tolerance=3e-10)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model = N3Hybrid(num_nodes=nn)

        p.setup(force_alloc_complex=True)

        p.set_val('throttle', np.linspace(0.0001, 1., nn))
        p.set_val('fltcond|h', np.linspace(0, 40e3, nn), units='ft')
        p.set_val('fltcond|M', np.linspace(0.1, 0.9, nn))
        p.set_val('hybrid_power', np.linspace(1e3, 0, nn), units='kW')

        p.run_model()

        assert_near_equal(p.get_val('thrust', units='lbf'), np.array([2139.16344767, 4298.37034408,
                          5731.7340516, 5567.1710359, 6098.6277089 ]), tolerance=1e-10)
        assert_near_equal(p.get_val('fuel_flow', units='kg/s'), 1e-3*np.array([128.30544414,
                          151.07410832, 248.57054231, 280.13559437, 242.69174402]), tolerance=1e-10)
        assert_near_equal(p.get_val('surge_margin'), np.array([3.61771543, 5.13884665,
                          17.61454298, 29.65132244, 17.4655423 ]), tolerance=3e-10)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

@unittest.skipIf(skip_tests, "N+3 surrogate model has not been trained (cached data not found), so skipping N+3 tests")
class N3TestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model = N3()

        p.setup(force_alloc_complex=True)

        p.set_val('throttle', 0.5)
        p.set_val('fltcond|h', 10e3, units='ft')
        p.set_val('fltcond|M', 0.5)

        p.run_model()

        assert_near_equal(p.get_val('thrust', units='lbf'), 6902.32350757*np.ones(1), tolerance=1e-10)
        assert_near_equal(p.get_val('fuel_flow', units='kg/s'), .35176628605*np.ones(1), tolerance=1e-10)
        assert_near_equal(p.get_val('surge_margin'), 18.42442281*np.ones(1), tolerance=3e-10)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model = N3Hybrid(num_nodes=nn)

        p.setup(force_alloc_complex=True)

        p.set_val('throttle', np.linspace(0.0001, 1., nn))
        p.set_val('fltcond|h', np.linspace(0, 40e3, nn), units='ft')
        p.set_val('fltcond|M', np.linspace(0.1, 0.9, nn))

        p.run_model()

        assert_near_equal(p.get_val('thrust', units='lbf'), np.array([2111.4945168, 4298.21260618,
                          5731.73000548, 5567.19105977, 6098.6277089]), tolerance=1e-10)
        assert_near_equal(p.get_val('fuel_flow', units='kg/s'), 1e-3*np.array([154.42603246,
                          185.27455666, 268.86802685, 290.35633843, 242.69174402]), tolerance=1e-10)
        assert_near_equal(p.get_val('surge_margin'), np.array([9.61978054, 9.85226939,
                          21.01362321, 31.54418716, 17.4655423]), tolerance=3e-10)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


if __name__=="__main__":
    unittest.main()
