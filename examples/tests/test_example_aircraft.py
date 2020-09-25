
from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from examples.B738 import run_738_analysis
from examples.TBM850 import run_tbm_analysis
from examples.HybridTwin_thermal import run_hybrid_twin_thermal_analysis
from examples.HybridTwin import run_hybrid_twin_analysis
from examples.Caravan import run_caravan_analysis
from examples.KingAirC90GT import run_kingair_analysis
from examples.ElectricSinglewithThermal import run_electricsingle_analysis


class TBMAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_tbm_analysis()

    def test_values_TBM(self):
        prob = self.prob
        assert_near_equal(prob.get_val('climb.OEW', units='lb'), 4756.772140709275, tolerance=1e-5)
        assert_near_equal(prob.get_val('rotate.range_final', units='ft'), 2489.49501148, tolerance=1e-5)
        assert_near_equal(prob.get_val('engineoutclimb.gamma',units='deg'), 8.78263, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lb'), 1607.84846911, tolerance=1e-5)

class CaravanAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_caravan_analysis()

    def test_values_Caravan(self):
        prob = self.prob
        assert_near_equal(prob.get_val('v1vr.range_final', units='ft'), 1375.59921952, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lb'), 379.90334044, tolerance=1e-5)

class HybridTwinThermalTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_hybrid_twin_thermal_analysis()

    def test_values_thermalhybridtwin(self):
        prob = self.prob
        assert_near_equal(prob.get_val('climb.OEW', units='lb'), 6673.001027260613, tolerance=1e-5)
        assert_near_equal(prob.get_val('rotate.range_final', units='ft'), 4434.461454141321, tolerance=1e-5)
        assert_near_equal(prob.get_val('engineoutclimb.gamma',units='deg'), 1.7508055214855194, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lb'), 862.667748103923, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.propmodel.batt1.SOC_final', units=None), -1.3067363280327982e-06, tolerance=1e-5)

        assert_near_equal(prob.get_val('climb.propmodel.motorheatsink.T', units='degC')[-1], 76.29252153, tolerance=1e-5)
        assert_near_equal(prob.get_val('climb.propmodel.batteryheatsink.T', units='degC')[-1], -0.12704621031633678, tolerance=1e-5)
        assert_near_equal(prob.get_val('cruise.propmodel.duct.drag', units='lbf')[0], 7.935834730687241, tolerance=1e-5)


class HybridTwinTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_hybrid_twin_analysis()
    
    def test_values_hybridtwin(self):
        prob = self.prob
        assert_near_equal(prob.get_val('climb.OEW', units='lb'), 6648.424765080086, tolerance=1e-5)
        assert_near_equal(prob.get_val('rotate.range_final', units='ft'), 4383.871458066499, tolerance=1e-5)
        assert_near_equal(prob.get_val('engineoutclimb.gamma',units='deg'), 1.7659046316724112, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lb'), 854.8937776195904, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.propmodel.batt1.SOC_final', units=None), -0.00030626412, tolerance=1e-5)


class KingAirTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_kingair_analysis()
    
    def test_values_kingair(self):
        prob = self.prob
        assert_near_equal(prob.get_val('climb.OEW', units='lb'), 6471.539115423346, tolerance=1e-5)
        assert_near_equal(prob.get_val('rotate.range_final', units='ft'), 3054.61279799, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lb'), 1666.73459582, tolerance=1e-5)


class ElectricSingleTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_electricsingle_analysis()
    
    def test_values_electricsingle(self):
        prob = self.prob
        assert_near_equal(prob.get_val('rotate.range_final', units='ft'), 2419.111568458725, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.propmodel.batt1.SOC')[-1], 0.1663373102614198, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.propmodel.motorheatsink.T', units='degC')[-1], 14.918329533221709, tolerance=1e-5)

class B738TestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_738_analysis()
    
    def test_values_B738(self):
        prob = self.prob
        # block fuel
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lbm'), 28549.432517, tolerance=2e-5)
        # changelog: 9/2020 - previously 28688.329, updated CFM surrogate model to reject spurious high Mach, low altitude points
        # total fuel
        assert_near_equal(prob.get_val('loiter.fuel_used_final', units='lbm'), 34424.68533072, tolerance=2e-5)
        # changelog: 9/2020 - previously 34555.313, updated CFM surrogate model to reject spurious high Mach, low altitude points
