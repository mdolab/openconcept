from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
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
        assert_rel_error(self, prob['osfgeometry.dh_cold'], 0.002462541, tolerance=1e-6)
        assert_rel_error(self, prob['heat_transfer'], 10020.13126, tolerance=1e-6 )
        assert_rel_error(self, prob['delta_p_cold'], -131.9862069, tolerance=1e-6 )
        assert_rel_error(self, prob['delta_p_hot'], -9112.282754, tolerance=1e-6 )
        assert_rel_error(self, prob['component_weight'], 1.147605, tolerance=1e-5 )

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    # def test_nondefault_settings(self):
    #     prob = Problem(BatteryTestGroup(vec_size=10,
    #                                     use_defaults=False,
    #                                     efficiency=0.95,
    #                                     p=3000,
    #                                     e=500,
    #                                     cost_inc=100,
    #                                     cost_base=0))
    #     prob.setup(check=True,force_alloc_complex=True)
    #     prob.run_model()
    #     assert_rel_error(self, prob.get_val('battery.heat_out', units='kW'), np.ones(10)*100*0.05, tolerance=1e-15)
    #     assert_rel_error(self, prob['battery.component_sizing_margin'], np.ones(10)/3, tolerance=1e-15)
    #     assert_rel_error(self, prob['battery.component_cost'], 10000, tolerance=1e-15)
    #     assert_rel_error(self, prob.get_val('battery.max_energy', units='W*h'), 500*100, tolerance=1e-15)

    #     partials = prob.check_partials(method='cs',compact_print=True)
    #     assert_check_partials(partials)
