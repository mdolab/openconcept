import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from openconcept.examples.B738 import run_738_analysis
from openconcept.examples.TBM850 import run_tbm_analysis
from openconcept.examples.HybridTwin_thermal import run_hybrid_twin_thermal_analysis
from openconcept.examples.HybridTwin import run_hybrid_twin_analysis
from openconcept.examples.Caravan import run_caravan_analysis
from openconcept.examples.KingAirC90GT import run_kingair_analysis
from openconcept.examples.ElectricSinglewithThermal import run_electricsingle_analysis
from openconcept.examples.N3_HybridSingleAisle_Refrig import run_hybrid_sa_analysis
from openconcept.examples.minimal import setup_problem as setup_minimal_problem
from openconcept.examples.minimal_integrator import MissionAnalysisWithFuelBurn as MinimalIntegratorMissionAnalysis
from openconcept.examples.B738_sizing import run_738_sizing_analysis

try:
    from openconcept.examples.B738_VLM_drag import run_738_analysis as run_738VLM_analysis
    from openconcept.aerodynamics.openaerostruct import VLMDataGen, OASDataGen
    from openconcept.examples.B738_aerostructural import run_738_analysis as run_738Aerostruct_analysis

    OAS_installed = True
except ImportError:
    OAS_installed = False


class TBMAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_tbm_analysis()
        self.prob.run_model()

    def test_values_TBM(self):
        prob = self.prob
        assert_near_equal(prob.get_val("climb.OEW", units="lb"), 4756.772140709275, tolerance=1e-5)
        assert_near_equal(prob.get_val("rotate.range_final", units="ft"), 2490.89174399, tolerance=1e-5)
        assert_near_equal(prob.get_val("engineoutclimb.gamma", units="deg"), 8.78263, tolerance=1e-5)
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lb"), 633.58800032, tolerance=1e-5)


class CaravanAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_caravan_analysis()

    def test_values_Caravan(self):
        prob = self.prob
        assert_near_equal(prob.get_val("v1vr.range_final", units="ft"), 1375.59921952, tolerance=1e-5)
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lb"), 379.90334044, tolerance=1e-5)


class HybridTwinThermalTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_hybrid_twin_thermal_analysis()

    def test_values_thermalhybridtwin(self):
        prob = self.prob
        assert_near_equal(prob.get_val("climb.OEW", units="lb"), 6673.001027260613, tolerance=1e-5)
        assert_near_equal(prob.get_val("rotate.range_final", units="ft"), 4434.68545427, tolerance=1e-5)
        assert_near_equal(prob.get_val("engineoutclimb.gamma", units="deg"), 1.75074018, tolerance=1e-5)
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lb"), 862.69811822, tolerance=1e-5)
        assert_near_equal(
            prob.get_val("descent.propmodel.batt1.SOC_final", units=None), -3.80158704e-05, tolerance=1e-5
        )

        assert_near_equal(
            prob.get_val("climb.propmodel.motorheatsink.T", units="degC")[-1], 76.19938727507775, tolerance=1e-5
        )
        assert_near_equal(
            prob.get_val("climb.propmodel.batteryheatsink.T", units="degC")[-1], -0.27586540922391123, tolerance=1e-5
        )
        assert_near_equal(prob.get_val("cruise.propmodel.duct.drag", units="lbf")[0], 7.968332825694923, tolerance=1e-5)
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
        assert_near_equal(prob.get_val("climb.OEW", units="lb"), 6648.424765080086, tolerance=1e-5)
        assert_near_equal(prob.get_val("rotate.range_final", units="ft"), 4383.871458066499, tolerance=1e-5)
        assert_near_equal(prob.get_val("engineoutclimb.gamma", units="deg"), 1.7659046316724112, tolerance=1e-5)
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lb"), 854.8937776195904, tolerance=1e-5)
        assert_near_equal(prob.get_val("descent.propmodel.batt1.SOC_final", units=None), -0.00030626412, tolerance=1e-5)


class KingAirTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_kingair_analysis()

    def test_values_kingair(self):
        prob = self.prob
        assert_near_equal(prob.get_val("climb.OEW", units="lb"), 6471.539115423346, tolerance=1e-5)
        assert_near_equal(prob.get_val("rotate.range_final", units="ft"), 3054.61279799, tolerance=1e-5)
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lb"), 1666.73459582, tolerance=1e-5)


class ElectricSingleTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_electricsingle_analysis()

    def test_values_electricsingle(self):
        prob = self.prob
        assert_near_equal(prob.get_val("rotate.range_final", units="ft"), 2419.111568458725, tolerance=1e-5)
        assert_near_equal(prob.get_val("descent.propmodel.batt1.SOC")[-1], 0.1663373102614198, tolerance=1e-5)
        assert_near_equal(
            prob.get_val("descent.propmodel.motorheatsink.T", units="degC")[-1], 14.906950172494192, tolerance=1e-5
        )
        # changelog 10/2020 - heat sink T now 14.90695 after minor change to hydraulic diameter computation in heat exchanger


class B738TestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_738_analysis()

    def test_values_B738(self):
        prob = self.prob
        # block fuel
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lbm"), 28549.432517, tolerance=3e-4)
        # changelog: 9/2020 - previously 28688.329, updated CFM surrogate model to reject spurious high Mach, low altitude points
        # total fuel
        assert_near_equal(prob.get_val("loiter.fuel_used_final", units="lbm"), 34424.68533072, tolerance=3e-4)
        # changelog: 9/2020 - previously 34555.313, updated CFM surrogate model to reject spurious high Mach, low altitude points


class B738SizingTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_738_sizing_analysis(num_nodes=5)

    def test_values_B738(self):
        prob = self.prob
        # block fuel
        assert_near_equal(
            prob.get_val("mission.descent.fuel_burn_integ.fuel_burn_final", units="lbm"),
            35213.7673772348,
            tolerance=1e-4,
        )
        # total fuel
        assert_near_equal(
            prob.get_val("mission.loiter.fuel_burn_integ.fuel_burn_final", units="lbm"),
            40991.187944303405,
            tolerance=1e-4,
        )
        # MTOW
        assert_near_equal(prob.get_val("ac|weights|MTOW", units="lbm"), 172711.3034007032, tolerance=1e-4)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class B738VLMTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_738VLM_analysis()

    def tearDown(self):
        # Get rid of any specified surface options in the VLMDataGen
        # class after every test. This is necessary because the class
        # stores the surface options as a "static" variable and
        # prevents multiple VLMDataGen instances with different
        # surface options. Doing this prevents that error when doing
        # multiple tests with different surface options.
        del VLMDataGen.surf_options

    def test_values_B738(self):
        prob = self.prob
        # block fuel
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lbm"), 27841.06640683, tolerance=1e-5)
        # total fuel
        assert_near_equal(prob.get_val("loiter.fuel_used_final", units="lbm"), 33412.41922187, tolerance=1e-5)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class B738AerostructTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_738Aerostruct_analysis()

    def tearDown(self):
        # Get rid of any specified surface options in the OASDataGen
        # class after every test. This is necessary because the class
        # stores the surface options as a "static" variable and
        # prevents multiple OASDataGen instances with different
        # surface options. Doing this prevents that error when doing
        # multiple tests with different surface options.
        del OASDataGen.surf_options

    def test_values_B738(self):
        prob = self.prob
        # block fuel
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lbm"), 33611.18277099, tolerance=1e-5)


class N3HSATestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_hybrid_sa_analysis(plots=False)

    def test_values_N3HSA(self):
        prob = self.prob
        # block fuel (no reserve, since the N+3 HSA uses the basic 3-phase mission)
        assert_near_equal(prob.get_val("descent.fuel_used_final", units="lbm"), 9006.52397811, tolerance=1e-5)


class MinimalTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = setup_minimal_problem()
        self.prob.run_model()

    def test_values_minimal(self):
        # No fuel burn, so check the throttle from the three phases
        assert_near_equal(
            self.prob.get_val("mission.climb.throttle"),
            np.array(
                [
                    0.651459,
                    0.647949,
                    0.644480,
                    0.641052,
                    0.637664,
                    0.634317,
                    0.631010,
                    0.627744,
                    0.624519,
                    0.621333,
                    0.618189,
                ]
            ),
            tolerance=1e-5,
        )
        assert_near_equal(self.prob.get_val("mission.cruise.throttle"), np.full(11, 0.490333), tolerance=1e-5)
        assert_near_equal(
            self.prob.get_val("mission.descent.throttle"),
            np.array(
                [
                    0.362142,
                    0.358981,
                    0.355778,
                    0.352535,
                    0.349250,
                    0.345924,
                    0.342557,
                    0.339149,
                    0.335699,
                    0.332207,
                    0.328674,
                ]
            ),
            tolerance=1e-5,
        )


class MinimalIntegratorTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = setup_minimal_problem(model=MinimalIntegratorMissionAnalysis)
        self.prob.run_model()

    def test_values_minimal(self):
        assert_near_equal(
            self.prob.get_val("mission.descent.fuel_integrator.fuel_burned_final"), 633.350, tolerance=1e-5
        )


if __name__ == "__main__":
    unittest.main()
