from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
import openconcept.components.thermal as thermal

class PerfectHeatTransferCompTestCase(unittest.TestCase):
    """
    Test the PerfectHeatTransferComp component
    """
    def test_comp(self):
        num_nodes = 3
        prob = Problem()
        prob.model.add_subsystem('test', thermal.PerfectHeatTransferComp(num_nodes=num_nodes), promotes=['*'])
        prob.setup(check=True, force_alloc_complex=True)

        # Set the values
        prob['T_in'] = np.array([300., 350., 400.])
        prob['q'] = np.array([10000., 0., -10000.])
        prob['mdot_coolant'] = np.array([1., 1., 1.])

        prob.run_model()

        dT_coolant = 10000./3801.

        assert_near_equal(prob['T_out'], np.array([300 + dT_coolant, 350., 400 - dT_coolant]))
        assert_near_equal(prob['T_average'], np.array([300 + dT_coolant/2, 350., 400 - dT_coolant/2]))

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)