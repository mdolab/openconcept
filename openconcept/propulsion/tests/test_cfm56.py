import unittest
import os
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem
import openconcept
from openconcept.propulsion import CFM56

# Skip these test cases if the cached surrogate files don't exist
file_root = openconcept.__path__[0] + r"/propulsion/empirical_data/cfm56/"
cached_thrust = os.path.exists(file_root + "cfm56thrust_trained.zip")
cached_fuelburn = os.path.exists(file_root + "cfm56fuelburn_trained.zip")
cached_T4 = os.path.exists(file_root + "cfm56T4_trained.zip")
skip_tests = True
if cached_thrust and cached_fuelburn and cached_T4:
    skip_tests = False


@unittest.skipIf(
    skip_tests, "CFM56 surrogate model has not been trained (cached data not found), so skipping CFM56 tests"
)
class CFM56TestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.add_subsystem("comp", CFM56(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("throttle", 0.5)
        p.set_val("fltcond|h", 10e3, units="ft")
        p.set_val("fltcond|M", 0.5)

        p.run_model()

        assert_near_equal(p.get_val("thrust", units="lbf"), 7050.73840869 * np.ones(1), tolerance=1e-6)
        assert_near_equal(p.get_val("fuel_flow", units="kg/s"), 0.50273824 * np.ones(1), tolerance=1e-6)
        assert_near_equal(p.get_val("T4", units="degK"), 1432.06813946 * np.ones(1), tolerance=1e-6)

    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("comp", CFM56(num_nodes=nn), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("throttle", np.linspace(0.0001, 1.0, nn))
        p.set_val("fltcond|h", np.linspace(0, 40e3, nn), units="ft")
        p.set_val("fltcond|M", np.linspace(0.1, 0.9, nn))

        p.run_model()

        assert_near_equal(
            p.get_val("thrust", units="lbf"),
            np.array([1445.41349482, 3961.46624224, 5278.43191982, 5441.44404298, 6479.00525867]),
            tolerance=5e-3,
        )
        assert_near_equal(
            p.get_val("fuel_flow", units="kg/s"),
            np.array([0.17032429, 0.25496437, 0.35745638, 0.40572545, 0.4924194]),
            tolerance=5e-3,
        )
        assert_near_equal(
            p.get_val("T4", units="degK"),
            np.array([1005.38911171, 1207.57548728, 1381.94820904, 1508.07901676, 1665.37063872]),
            tolerance=5e-3,
        )


if __name__ == "__main__":
    unittest.main()
