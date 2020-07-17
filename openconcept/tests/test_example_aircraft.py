from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from examples.TBM850 import run_tbm_analysis
from examples.HybridTwin_thermal import run_hybrid_twin_thermal_analysis

class TBMAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_tbm_analysis()

    def test_values_TBM(self):
        prob = self.prob
        assert_rel_error(self, prob.get_val('climb.OEW', units='lb'), 4756.772140709275, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('rotate.range_final', units='ft'), 2489.7142498884746, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('engineoutclimb.gamma',units='deg'), 8.78263, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('descent.fuel_used_final', units='lb'), 1605.32123542, tolerance=1e-5)

class HybridTwinThermalTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_hybrid_twin_thermal_analysis()

    def test_values_TBM(self):
        prob = self.prob
        assert_rel_error(self, prob.get_val('climb.OEW', units='lb'), 6673.001027260613, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('rotate.range_final', units='ft'), 4434.461454141321, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('engineoutclimb.gamma',units='deg'), 1.7508055214855194, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('descent.fuel_used_final', units='lb'), 862.667748103923, tolerance=1e-5)


            # vars_list = ['ac|weights|MTOW','climb.OEW','descent.fuel_used_final',
            #      'rotate.range_final','descent.propmodel.batt1.SOC_final','cruise.hybridization',
            #      'ac|weights|W_battery','margins.MTOW_margin',
            #      'ac|propulsion|motor|rating','ac|propulsion|generator|rating','ac|propulsion|engine|rating',
            #      'ac|geom|wing|S_ref','v0v1.Vstall_eas','v0v1.takeoff|vr',
            #      'engineoutclimb.gamma', 'ac|propulsion|thermal|duct|area_nozzle',
            #      'cruise.propmodel.duct.drag', 'ac|propulsion|thermal|hx|coolant_mass',
            #      'climb.propmodel.duct.mdot']