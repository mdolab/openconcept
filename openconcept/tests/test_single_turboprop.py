from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from examples.TBM850 import define_analysis as TBM_define_analysis
from examples.Caravan import define_analysis as caravan_define_analysis

from openconcept.analysis.takeoff import takeoff_check
from openconcept.utilities.nodes import compute_num_nodes

class TBMAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        n_int_per_seg = 5
        self.prob = TBM_define_analysis(n_int_per_seg)
        mission_segments=['climb','cruise','descent']
        nn, nn_tot_to, nn_tot_m, nn_tot = compute_num_nodes(n_int_per_seg, mission_segments)

        self.prob.model.add_objective('design_mission.fuel_burn')
        self.prob.setup(mode='fwd',check=True, force_alloc_complex=True)
        self.prob['OEW.const.structural_fudge'] = 1.67
        self.prob['takeoff.v1_solve.takeoff|v1'] = 30
        self.prob['design_mission.throttle'] = np.ones(nn_tot_m)*0.7
        self.prob['design_mission.cruise|time'] = 1000
        self.prob.run_model()

    def test_values_TBM(self):
        prob = self.prob
        takeoff_check(prob)
        assert_rel_error(self, prob.get_val('design_range', units='NM'), 1250, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('OEW', units='lb'), 4756.77214071, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('design_mission.residuals.MTOW_margin', units='lb'), 27.15401203, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('takeoff.distance_continue', units='ft'), 2847.58822777, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('takeoff.distance_abort', units='ft'), 2847.57692822, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('takeoff.vstall.Vstall_eas', units='kn'), 81.41883299, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('takeoff.takeoff|vr', units='kn'), 89.56071629, tolerance=1e-4)
        assert_rel_error(self, prob.get_val('takeoff.takeoff|v1', units='kn'), 85.31369135, tolerance=1e-4)
        assert_rel_error(self, prob.get_val('takeoff.climb|gamma'), 0.08181493, tolerance=1e-4)

        assert_rel_error(self, prob.get_val('design_mission.mission_total_fuel', units='lb'), 1605.42500307, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('takeoff.total_fuel', units='lb'), 2.75985527, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('design_mission.fuel_burn', units='lb'), 1608.18485834, tolerance=1e-5)

# class CaravanTestCase(unittest.TestCase):
#     def setUp(self):
#         n_int_per_seg = 5
#         self.prob = caravan_define_analysis(n_int_per_seg)
#         mission_segments=['climb','cruise','descent']
#         nn, nn_tot_to, nn_tot_m, nn_tot = compute_num_nodes(n_int_per_seg, mission_segments)

#         self.prob.model.add_objective('design_mission.fuel_burn')
#         self.prob.setup(mode='fwd',check=True, force_alloc_complex=True)
#         self.prob['OEW.const.structural_fudge'] = 1.4
#         self.prob['takeoff.v1_solve.takeoff|v1'] = 30
#         self.prob['design_mission.throttle'] = np.ones(nn_tot_m)*0.7
#         self.prob['design_mission.cruise|time'] = 1000
#         self.prob.run_model()

#     def test_values_TBM(self):
#         prob = self.prob
#         takeoff_check(prob)
#         assert_rel_error(self, prob.get_val('design_range', units='NM'), 1250, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('OEW', units='lb'), 4756.77214071, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('design_mission.residuals.MTOW_margin', units='lb'), 27.14265201, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('takeoff.distance_continue', units='ft'), 2847.58822777, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('takeoff.distance_abort', units='ft'), 2847.82318835, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('takeoff.vstall.Vstall_eas', units='kn'), 81.41883299, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('takeoff.takeoff|vr', units='kn'), 89.56071629, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('takeoff.takeoff|v1', units='kn'), 85.31369135, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('takeoff.climb|gamma'), 0.08181493, tolerance=1e-5)

#         assert_rel_error(self, prob.get_val('design_mission.mission_total_fuel', units='lb'), 1605.42500307, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('takeoff.total_fuel', units='lb'), 2.75985527, tolerance=1e-5)
#         assert_rel_error(self, prob.get_val('design_mission.fuel_burn', units='lb'), 1608.18485834, tolerance=1e-5)