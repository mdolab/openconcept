from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.components.heat_exchanger import OffsetStripFinGeometry, OffsetStripFinData, HydraulicDiameterReynoldsNumber, OutletTemperatures, PressureDrop
from openconcept.components.heat_exchanger import NusseltFromColburnJ, ConvectiveCoefficient, FinEfficiency, UAOverall, NTUMethod, CrossFlowNTUEffectiveness, NTUEffectivenessActualHeatTransfer

class OSFGeometryTestGroup(Group):
    """
    Test the offset strip fin geometry component
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points' )

    def setup(self):
        nn = self.options['num_nodes']

        iv = self.add_subsystem('iv', IndepVarComp(), promotes_outputs=['*'])
        iv.add_output('case_thickness', val=2.0, units='mm')
        iv.add_output('fin_thickness', val=0.102, units='mm')
        iv.add_output('plate_thickness', val=0.2, units='mm')
        iv.add_output('material_k', val=190, units='W/m/K')
        iv.add_output('material_rho', val=2700, units='kg/m**3')

        iv.add_output('mdot_cold', val=np.ones(nn)*1.5, units='kg/s')
        iv.add_output('rho_cold', val=np.ones(nn)*0.5, units='kg/m**3')

        iv.add_output('mdot_hot', val=0.075*np.ones(nn), units='kg/s')
        iv.add_output('rho_hot', val=np.ones(nn)*1020.2, units='kg/m**3')

        iv.add_output('T_in_cold', val=np.ones(nn)*45, units='degC')
        iv.add_output('T_in_hot', val=np.ones(nn)*90, units='degC')
        iv.add_output('n_long_cold', val=3)
        iv.add_output('n_wide_cold', val=430)
        iv.add_output('n_tall', val=19)

        iv.add_output('channel_height_cold', val=14, units='mm')
        iv.add_output('channel_width_cold', val=1.35, units='mm')
        iv.add_output('fin_length_cold', val=6, units='mm')
        iv.add_output('cp_cold', val=1005, units='J/kg/K')
        iv.add_output('k_cold', val=0.02596, units='W/m/K')
        iv.add_output('mu_cold', val=1.789e-5, units='kg/m/s')

        iv.add_output('channel_height_hot', val=1, units='mm')
        iv.add_output('channel_width_hot', val=1, units='mm')
        iv.add_output('fin_length_hot', val=6, units='mm')
        iv.add_output('cp_hot', val=3801, units='J/kg/K')
        iv.add_output('k_hot', val=0.405, units='W/m/K')
        iv.add_output('mu_hot', val=1.68e-3, units='kg/m/s')



        self.add_subsystem('osfgeometry', OffsetStripFinGeometry(), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('redh', HydraulicDiameterReynoldsNumber(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('osfdata', OffsetStripFinData(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('nusselt', NusseltFromColburnJ(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('convection', ConvectiveCoefficient(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('finefficiency', FinEfficiency(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('ua', UAOverall(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('ntu', NTUMethod(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('effectiveness', CrossFlowNTUEffectiveness(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('heat', NTUEffectivenessActualHeatTransfer(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('t_out', OutletTemperatures(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('delta_p', PressureDrop(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

class OSFGeometryTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob['osfgeometry.dh_cold'], 0.00242316, tolerance=1e-6)
        assert_near_equal(prob['heat_transfer'], 10040.9846, tolerance=1e-6 )
        assert_near_equal(prob['delta_p_cold'], -135.15338626, tolerance=1e-6 )
        assert_near_equal(prob['delta_p_hot'], -9112.282754, tolerance=1e-6 )
        assert_near_equal(prob['component_weight'], 1.147605, tolerance=1e-5 )

        partials = prob.check_partials(method='cs',compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_kayslondon_10_61(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True,force_alloc_complex=True)
        prob.set_val('fin_thickness', 0.004, units='inch')
        prob.set_val('plate_thickness', 0.004, units='inch')
        prob.set_val('fin_length_cold', 1./10., units='inch')
        fin_spacing = 1 / 19.35 - 0.004 # fin pitch minus fin thickness
        prob.set_val('channel_height_cold', 0.0750-0.004, units='inch')
        prob.set_val('channel_width_cold', fin_spacing, units='inch')
        prob.set_val('n_long_cold', 2)
        prob.set_val('mdot_cold', 0.0905, units='kg/s')

        prob.run_model()
        prob.model.list_outputs(units=True)
        # test the geometry in Kays and London 3rd Ed Pg 248, Fig 10-61
        assert_near_equal(prob['osfgeometry.dh_cold'], 1.403e-3, tolerance=1e-3)
        assert_near_equal(prob['redh.Re_dh_cold'], 400., tolerance=1e-2)
        # data directly from Kays/London at Redh=400
        assert_near_equal(prob['osfdata.j_cold'], 0.0195, tolerance=2e-1)
        assert_near_equal(prob['osfdata.f_cold'], 0.0750, tolerance=2e-1 )

        prob.set_val('mdot_cold', 0.0905*5, units='kg/s')
        prob.run_model()
        # data directly from Kays/London at Redh=2000
        assert_near_equal(prob['redh.Re_dh_cold'], 2000., tolerance=1e-2)
        assert_near_equal(prob['osfdata.j_cold'], 0.00940, tolerance=2e-1)
        assert_near_equal(prob['osfdata.f_cold'], 0.0303, tolerance=3.5e-1 )

        assert_near_equal(prob['osfgeometry.alpha_cold'], 0.672, tolerance=1e-2)
        assert_near_equal(prob['osfgeometry.delta_cold'], 0.040, tolerance=1e-2)
        assert_near_equal(prob['osfgeometry.gamma_cold'], 0.084, tolerance=1e-2)

    def test_kayslondon_10_55(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True,force_alloc_complex=True)
        prob.set_val('fin_thickness', 0.004, units='inch')
        prob.set_val('plate_thickness', 0.004, units='inch')
        prob.set_val('fin_length_cold', 1./8., units='inch')
        fin_spacing = 1 / 15.61 - 0.004 # fin pitch minus fin thickness
        prob.set_val('channel_height_cold', 0.250-0.004, units='inch')
        prob.set_val('channel_width_cold', fin_spacing, units='inch')
        prob.set_val('n_long_cold', 2)
        prob.set_val('mdot_cold', 0.235, units='kg/s')

        prob.run_model()
        # test the geometry in Kays and London 3rd Ed Pg 248, Fig 10-55
        assert_near_equal(prob['osfgeometry.dh_cold'], 2.383e-3, tolerance=1e-2)
        # data directly from Kays/London at Redh=400
        assert_near_equal(prob['redh.Re_dh_cold'], 400., tolerance=1e-2)
        assert_near_equal(prob['osfdata.j_cold'], 0.0246, tolerance=1e-1)
        assert_near_equal(prob['osfdata.f_cold'], 0.104, tolerance=1e-1)
        prob.set_val('mdot_cold', 0.235*5, units='kg/s')
        prob.run_model()
        # data directly from Kays/London at Redh=2000
        assert_near_equal(prob['redh.Re_dh_cold'], 2000., tolerance=1e-2)
        assert_near_equal(prob['osfdata.j_cold'], 0.0111, tolerance=1e-1)
        assert_near_equal(prob['osfdata.f_cold'], 0.0420, tolerance=1e-1 )

        assert_near_equal(prob['osfgeometry.alpha_cold'], 0.244, tolerance=1e-2)
        assert_near_equal(prob['osfgeometry.delta_cold'], 0.032, tolerance=1e-2)
        assert_near_equal(prob['osfgeometry.gamma_cold'], 0.067, tolerance=1e-2)

    def test_kayslondon_10_60(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True,force_alloc_complex=True)
        prob.set_val('fin_thickness', 0.004, units='inch')
        prob.set_val('plate_thickness', 0.004, units='inch')
        prob.set_val('fin_length_cold', 1./10., units='inch')
        fin_spacing = 1 / 27.03 - 0.004 # fin pitch minus fin thickness
        prob.set_val('channel_height_cold', 0.250-0.004, units='inch')
        prob.set_val('channel_width_cold', fin_spacing, units='inch')
        prob.set_val('n_long_cold', 2)
        prob.set_val('mdot_cold', 0.27, units='kg/s')

        prob.run_model()

        # test the geometry in Kays and London 3rd Ed Pg 248, Fig 10-55
        # assert_near_equal(prob['osfgeometry.dh_cold'], 0.00147796, tolerance=1e-4)
        assert_near_equal(prob['osfgeometry.dh_cold'], 0.001423, tolerance=1e-2)
        # data directly from Kays/London at Redh=500
        assert_near_equal(prob['redh.Re_dh_cold'], 500., tolerance=1e-2)
        assert_near_equal(prob['osfdata.j_cold'], 0.0238, tolerance=1e-1)
        assert_near_equal(prob['osfdata.f_cold'], 0.0922, tolerance=1e-1)
        prob.set_val('mdot_cold', 0.27*4, units='kg/s')
        prob.run_model()
        # data directly from Kays/London at Redh=2000
        assert_near_equal(prob['redh.Re_dh_cold'], 2000., tolerance=1e-2)
        assert_near_equal(prob['osfdata.j_cold'], 0.0113, tolerance=1e-1)
        assert_near_equal(prob['osfdata.f_cold'], 0.0449, tolerance=1e-1 )

        assert_near_equal(prob['osfgeometry.alpha_cold'], 0.134, tolerance=1e-2)
        assert_near_equal(prob['osfgeometry.delta_cold'], 0.040, tolerance=1e-2)
        assert_near_equal(prob['osfgeometry.gamma_cold'], 0.121, tolerance=1e-2)



    def test_kayslondon_10_63(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True,force_alloc_complex=True)
        prob.set_val('fin_thickness', 0.004, units='inch')
        prob.set_val('plate_thickness', 0.004, units='inch')
        prob.set_val('fin_length_cold', 3./32., units='inch')
        fin_spacing = 0.082 - 0.004 # fin pitch minus fin thickness
        prob.set_val('channel_height_cold', 0.485-0.004, units='inch')
        prob.set_val('channel_width_cold', fin_spacing, units='inch')
        prob.set_val('n_long_cold', 4)
        prob.set_val('mdot_cold', 0.54, units='kg/s')

        prob.run_model()
        # test the geometry in Kays and London 3rd Ed Pg 248, Fig 10-55
        # assert_near_equal(prob['osfgeometry.dh_cold'], 0.00341, tolerance=1e-2)
        # data directly from Kays/London at Redh=500
        assert_near_equal(prob['redh.Re_dh_cold'], 500., tolerance=1e-2)
        assert_near_equal(prob['osfdata.j_cold'], 0.0205, tolerance=2e-1)
        assert_near_equal(prob['osfdata.f_cold'], 0.130, tolerance=2e-1)
        prob.set_val('mdot_cold', 0.54*4, units='kg/s')
        prob.run_model()
        # data directly from Kays/London at Redh=2000
        assert_near_equal(prob['redh.Re_dh_cold'], 2000., tolerance=1e-2)
        assert_near_equal(prob['osfdata.j_cold'], 0.0119, tolerance=2e-1)
        assert_near_equal(prob['osfdata.f_cold'], 0.0607, tolerance=2e-1 )

        assert_near_equal(prob['osfgeometry.alpha_cold'], 0.162, tolerance=1e-2)
        assert_near_equal(prob['osfgeometry.delta_cold'], 0.043, tolerance=1e-2)
        assert_near_equal(prob['osfgeometry.gamma_cold'], 0.051, tolerance=1e-2)

if __name__=="__main__":
    unittest.main()