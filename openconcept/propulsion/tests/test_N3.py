import unittest
import os
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem
import openconcept
from openconcept.propulsion import N3, N3Hybrid

# Skip these test cases if the cached surrogate files don't exist
# N+3 hybrid
hybrid_file_root = openconcept.__path__[0] + r"/propulsion/empirical_data/n+3_hybrid/"
hybrid_cached_thrust = os.path.exists(hybrid_file_root + r"/n3_hybrid_thrust_trained.zip")
hybrid_cached_fuelburn = os.path.exists(hybrid_file_root + r"n3_hybrid_fuelflow_trained.zip")
hybrid_cached_surge = os.path.exists(hybrid_file_root + "n3_hybrid_smw_trained.zip")
hybrid_skip_tests = True
if hybrid_cached_thrust and hybrid_cached_fuelburn and hybrid_cached_surge:
    hybrid_skip_tests = False

# N+3
file_root = openconcept.__path__[0] + r"/propulsion/empirical_data/n+3/"
cached_thrust = os.path.exists(file_root + r"/n3_thrust_trained.zip")
cached_fuelburn = os.path.exists(file_root + r"n3_fuelflow_trained.zip")
cached_T4 = os.path.exists(file_root + r"n3_smw_trained.zip")
skip_tests = True
if cached_thrust and cached_fuelburn and cached_T4:
    skip_tests = False


@unittest.skipIf(
    hybrid_skip_tests,
    "N+3 hybrid surrogate model has not been trained (cached data not found), so skipping N+3 hybrid tests",
)
class N3HybridTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.add_subsystem("comp", N3Hybrid(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("throttle", 0.5)
        p.set_val("fltcond|h", 10e3, units="ft")
        p.set_val("fltcond|M", 0.5)
        p.set_val("hybrid_power", 200.0, units="kW")

        p.run_model()

        assert_near_equal(p.get_val("thrust", units="lbf"), 6965.43674107 * np.ones(1), tolerance=1e-6)
        assert_near_equal(p.get_val("fuel_flow", units="kg/s"), 0.34333925 * np.ones(1), tolerance=1e-6)
        assert_near_equal(p.get_val("surge_margin"), 17.49872296 * np.ones(1), tolerance=1e-6)

    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("comp", N3Hybrid(num_nodes=nn), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("throttle", np.linspace(0.0001, 1.0, nn))
        p.set_val("fltcond|h", np.linspace(0, 40e3, nn), units="ft")
        p.set_val("fltcond|M", np.linspace(0.1, 0.9, nn))
        p.set_val("hybrid_power", np.linspace(1e3, 0, nn), units="kW")

        p.run_model()

        assert_near_equal(
            p.get_val("thrust", units="lbf"),
            np.array([2143.74837065, 4298.37044104, 5731.73572266, 5567.1704698, 6093.64182948]),
            tolerance=5e-3,
        )
        assert_near_equal(
            p.get_val("fuel_flow", units="kg/s"),
            np.array([0.12830921, 0.15107399, 0.24857052, 0.28013556, 0.24271405]),
            tolerance=5e-3,
        )
        assert_near_equal(
            p.get_val("surge_margin"),
            np.array([3.62385489, 5.13891739, 17.61488138, 29.65131358, 17.48630861]),
            tolerance=5e-3,
        )


@unittest.skipIf(skip_tests, "N+3 surrogate model has not been trained (cached data not found), so skipping N+3 tests")
class N3TestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.add_subsystem("comp", N3(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("throttle", 0.5)
        p.set_val("fltcond|h", 10e3, units="ft")
        p.set_val("fltcond|M", 0.5)

        p.run_model()

        assert_near_equal(p.get_val("thrust", units="lbf"), 6902.32371562 * np.ones(1), tolerance=1e-6)
        assert_near_equal(p.get_val("fuel_flow", units="kg/s"), 0.35176628 * np.ones(1), tolerance=1e-6)
        assert_near_equal(p.get_val("surge_margin"), 18.42447377 * np.ones(1), tolerance=1e-6)

    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("comp", N3Hybrid(num_nodes=nn), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("throttle", np.linspace(0.0001, 1.0, nn))
        p.set_val("fltcond|h", np.linspace(0, 40e3, nn), units="ft")
        p.set_val("fltcond|M", np.linspace(0.1, 0.9, nn))

        p.run_model()

        assert_near_equal(
            p.get_val("thrust", units="lbf"),
            np.array([2116.01166807, 4298.21330183, 5731.73026453, 5567.1916333, 6093.64182948]),
            tolerance=5e-3,
        )
        assert_near_equal(
            p.get_val("fuel_flow", units="kg/s"),
            np.array([0.15443523, 0.18527426, 0.26886803, 0.29035632, 0.24271405]),
            tolerance=5e-3,
        )
        assert_near_equal(
            p.get_val("surge_margin"),
            np.array([9.63957356, 9.85223288, 21.01375818, 31.54415929, 17.48630861]),
            tolerance=5e-3,
        )


if __name__ == "__main__":
    unittest.main()
