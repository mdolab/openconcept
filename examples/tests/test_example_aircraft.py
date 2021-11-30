
from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from examples.B738 import run_738_analysis
from examples.B738_VLM_drag import run_738_analysis as run_738VLM_analysis
from examples.TBM850 import run_tbm_analysis
from examples.HybridTwin_thermal import run_hybrid_twin_thermal_analysis
from examples.HybridTwin_active_thermal import run_hybrid_twin_active_thermal_analysis
from examples.HybridTwin import run_hybrid_twin_analysis
from examples.Caravan import run_caravan_analysis
from examples.KingAirC90GT import run_kingair_analysis
from examples.ElectricSinglewithThermal import run_electricsingle_analysis
from examples.N3_HybridSingleAisle_Refrig import run_hybrid_sa_analysis


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
        assert_near_equal(prob.get_val('rotate.range_final', units='ft'), 4434.68545427, tolerance=1e-5)
        assert_near_equal(prob.get_val('engineoutclimb.gamma',units='deg'), 1.75074018, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lb'), 862.69811822, tolerance=1e-5)
        assert_near_equal(prob.get_val('descent.propmodel.batt1.SOC_final', units=None), -3.80158704e-05, tolerance=1e-5)

        assert_near_equal(prob.get_val('climb.propmodel.motorheatsink.T', units='degC')[-1], 76.19938727507775, tolerance=1e-5)
        assert_near_equal(prob.get_val('climb.propmodel.batteryheatsink.T', units='degC')[-1], -0.27586540922391123, tolerance=1e-5)
        assert_near_equal(prob.get_val('cruise.propmodel.duct.drag', units='lbf')[0], 7.968332825694923, tolerance=1e-5)
        # changelog 10/2020 - updated most of the values due to minor update to hydraulic diam calculation in the heat exchanger

# 10/2021 commenting out because does not converge with the new chiller in chiller.py
# class HybridTwinActiveThermalTestCase(unittest.TestCase):
#     def setUp(self):
#         self.prob = run_hybrid_twin_active_thermal_analysis()
    
#     def test_values_hybridtwin(self):
#         prob = self.prob

#         climb_duct_area = np.array([ 0.80614565,  3.25480096,  7.11240858, 12.075577  , 17.55488029,
#                                     23.40694116, 29.15510781, 34.44182758, 39.05343787, 43.00420553, 46.43866073])
#         cruise_duct_area = np.array([17.17611522, 15.22748148, 14.66271227, 14.38669164, 14.2745505 ,
#                                      14.20434496, 14.15713767, 14.11684779, 14.08034799, 14.04524349, 14.01099713])
#         prob.set_val('climb.propmodel.duct.area_nozzle', climb_duct_area, units='inch**2')
#         prob.set_val('cruise.propmodel.duct.area_nozzle', cruise_duct_area, units='inch**2')
#         prob.run_model()

#         assert_near_equal(prob.get_val('climb.OEW', units='lb'), 6673.001027260613, tolerance=1e-5)
#         # assert_near_equal(prob.get_val('descent.fuel_used_final', units='lb'), 871.66394047, tolerance=1e-5)
#         # assert_near_equal(prob.get_val('descent.propmodel.batt1.SOC_final', units=None), 0.00484123, tolerance=1e-5)

#         assert_near_equal(prob.get_val('climb.propmodel.duct.area_nozzle', units='inch**2'), climb_duct_area, tolerance=1e-5)
#         assert_near_equal(prob.get_val('cruise.propmodel.duct.area_nozzle', units='inch**2'), cruise_duct_area, tolerance=1e-5)
#         Wdot = np.array([ 6618.15094465, 17863.48477045, 25558.10458551, 30652.72996714, 33805.46342847, 35538.5460011,
#                           36221.44062722, 36149.9707508, 35539.35428109, 34562.89222503, 33346.05141285])
#         assert_near_equal(prob.get_val('climb.propmodel.refrig.elec_load', units='W'), Wdot, tolerance=1e-5)
#         assert_near_equal(prob.get_val('cruise.propmodel.refrig.elec_load', units='W'), np.zeros(11), tolerance=1e-5)
#         assert_near_equal(prob.get_val('climb.propmodel.motorheatsink.T', units='degC')[-1], 76.48202028574951, tolerance=1e-5)
#         assert_near_equal(prob.get_val('climb.propmodel.batteryheatsink.T', units='degC')[-1], 6.9112870295027165, tolerance=1e-5)
#         assert_near_equal(prob.get_val('cruise.propmodel.duct.drag', units='lbf')[-1], 1.5888992670493287, tolerance=1e-5)
#         # changelog 10/2020 - updated most of the values due to minor update to hydraulic diam calculation in the heat exchanger

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
        assert_near_equal(prob.get_val('descent.propmodel.motorheatsink.T', units='degC')[-1], 14.906950172494192, tolerance=1e-5)
        # changelog 10/2020 - heat sink T now 14.90695 after minor change to hydraulic diameter computation in heat exchanger

class B738TestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_738_analysis()
    
    def test_values_B738(self):
        prob = self.prob
        # block fuel
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lbm'), 28549.432517, tolerance=3e-4)
        # changelog: 9/2020 - previously 28688.329, updated CFM surrogate model to reject spurious high Mach, low altitude points
        # total fuel
        assert_near_equal(prob.get_val('loiter.fuel_used_final', units='lbm'), 34424.68533072, tolerance=3e-4)
        # changelog: 9/2020 - previously 34555.313, updated CFM surrogate model to reject spurious high Mach, low altitude points

class B738VLMTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_738VLM_analysis()
    
    def test_values_B738(self):
        prob = self.prob
        # block fuel
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lbm'), 28443.39604559, tolerance=1e-5)
        # total fuel
        assert_near_equal(prob.get_val('loiter.fuel_used_final', units='lbm'), 34075.30721371, tolerance=1e-5)

class N3HSATestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_hybrid_sa_analysis(plots=False)
    
    def test_values_N3HSA(self):
        prob = self.prob
        # block fuel (no reserve, since the N+3 HSA uses the basic 3-phase mission)
        assert_near_equal(prob.get_val('descent.fuel_used_final', units='lbm'), 9006.52397811, tolerance=1e-5)

if __name__=="__main__":
    unittest.main()
