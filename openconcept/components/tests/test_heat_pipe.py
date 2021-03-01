from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_warning
from openmdao.api import Problem, NewtonSolver, DirectSolver
import openconcept.components.heat_pipe as hp

class HeatPipeIntegrationTestCase(unittest.TestCase):
    """
    Test the HeatPipe group with everything integrated
    """
    def test_simple_scalar(self):
        nn = 1
        theta = 84.
        prob = Problem()
        pipe = prob.model.add_subsystem('test', hp.HeatPipe(num_nodes=nn, theta=theta), promotes=['*'])
        pipe.set_input_defaults('T_evap', units='degC', val=np.linspace(30, 30, nn))
        pipe.set_input_defaults('q', units='W', val=np.linspace(400, 400, nn))
        pipe.set_input_defaults('length', units='m', val=10.22)
        pipe.set_input_defaults('inner_diam', units='inch', val=.902)
        pipe.set_input_defaults('n_pipes', val=1.)
        pipe.set_input_defaults('T_design', units='degC', val=40)

        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob['q_max'], np.ones(nn)*2807.04869547, tolerance=1e-5)
        assert_near_equal(prob['weight'], 0.51463886, tolerance=1e-5)
        assert_near_equal(prob['T_cond'], np.ones(nn)*29.93440845, tolerance=1e-5)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_simple_vector(self):
        nn = 5
        prob = Problem()
        pipe = prob.model.add_subsystem('test', hp.HeatPipe(num_nodes=nn), promotes=['*'])
        pipe.set_input_defaults('T_evap', units='degC', val=np.linspace(30, 60, nn))
        pipe.set_input_defaults('q', units='W', val=np.linspace(400, 1000, nn))
        pipe.set_input_defaults('length', units='m', val=10.22)
        pipe.set_input_defaults('inner_diam', units='inch', val=.902)
        pipe.set_input_defaults('n_pipes', val=1.)
        pipe.set_input_defaults('T_design', units='degC', val=40)

        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob['q_max'], np.array([4936.75193703, 5022.29454826, 5074.04485581, 5095.29546816, 5081.04977578]), tolerance=1e-5)
        assert_near_equal(prob['weight'], 0.51463886, tolerance=1e-5)
        assert_near_equal(prob['T_cond'], np.array([29.93440845, 37.40981609, 44.88522609, 52.36063752, 59.83604972]), tolerance=1e-5)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_two_pipes(self):
        nn = 3
        prob = Problem()

        # Run one and two pipes to compare results
        one = Problem()
        pipe = one.model.add_subsystem('test', hp.HeatPipe(num_nodes=nn), promotes=['*'])
        pipe.set_input_defaults('T_evap', units='degC', val=np.linspace(30, 30, nn))
        pipe.set_input_defaults('q', units='W', val=np.linspace(200, 200, nn))
        pipe.set_input_defaults('length', units='m', val=10.22)
        pipe.set_input_defaults('inner_diam', units='inch', val=.702)
        pipe.set_input_defaults('n_pipes', val=1.)
        pipe.set_input_defaults('T_design', units='degC', val=70)

        one.setup(check=True, force_alloc_complex=True)
        one.run_model()

        # Twice as many pipes with twice as much heat
        two = Problem()
        pipe = two.model.add_subsystem('test', hp.HeatPipe(num_nodes=nn), promotes=['*'])
        pipe.set_input_defaults('T_evap', units='degC', val=np.linspace(30, 30, nn))
        pipe.set_input_defaults('q', units='W', val=np.linspace(400, 400, nn))
        pipe.set_input_defaults('length', units='m', val=10.22)
        pipe.set_input_defaults('inner_diam', units='inch', val=.702)
        pipe.set_input_defaults('n_pipes', val=2.)
        pipe.set_input_defaults('T_design', units='degC', val=70)

        two.setup(check=True, force_alloc_complex=True)
        two.run_model()

        assert_near_equal(one['q_max'], two['q_max']/2)
        assert_near_equal(one['weight'], two['weight']/2)
        assert_near_equal(one['T_cond'], two['T_cond'])

        partials = two.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class HeatPipeThermalResistanceTestCase(unittest.TestCase):
    """
    Basic test for HeatPipeThermalResistance component to ensure no drastic changes in outputs
    """
    def test_default_settings(self):
        nn = 3
        p = Problem()
        p.model.add_subsystem('test', hp.HeatPipeThermalResistance(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p['thermal_resistance'], np.ones(nn)*0.000898000112, tolerance=1e-5)
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class HeatPipeVaporTempDropTestCase(unittest.TestCase):
    """
    Basic test for HeatPipeVaporTempDrop component to ensure no drastic changes in outputs
    """
    def test_default_settings(self):
        nn = 3
        p = Problem()
        p.model.add_subsystem('test', hp.HeatPipeVaporTempDrop(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        
        assert_near_equal(p['delta_T'], np.ones(nn)*2.37127, tolerance=1e-5)
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class HeatPipeWeightTestCase(unittest.TestCase):
    """
    Basic test for HeatPipeWeight component to ensure no drastic changes in outputs
    """
    def test_default_settings(self):
        p = Problem()
        p.model.add_subsystem('test', hp.HeatPipeWeight(), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        
        assert_near_equal(p['heat_pipe_weight'], 0.04074404, tolerance=1e-5)
        assert_near_equal(p['wall_thickness'], 6.99300699e-05, tolerance=1e-5)
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class AmmoniaPropertiesTestCase(unittest.TestCase):
    """
    Basic test for AmmoniaProperties component to ensure no drastic changes in outputs
    """
    def test_on_data(self):
        nn = 3
        p = Problem()
        comp = p.model.add_subsystem('test', hp.AmmoniaProperties(num_nodes=nn), promotes=['*'])
        comp.set_input_defaults('temp', units='degC', val=np.ones(nn)*90.)
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        
        assert_near_equal(p['rho_liquid'], np.ones(nn)*482.9)
        assert_near_equal(p['rho_vapor'], np.ones(nn)*43.9)
        assert_near_equal(p['vapor_pressure'], np.ones(nn)*5123.)
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_interpolated(self):
        nn = 6
        p = Problem()
        comp = p.model.add_subsystem('test', hp.AmmoniaProperties(num_nodes=nn), promotes=['*'])
        comp.set_input_defaults('temp', units='degC', val=np.linspace(-7., 78., nn))
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        
        assert_near_equal(p['rho_liquid'], np.array([648.00187402, 624.69, 599.75101299, 572.91683838, 543.40564864, 509.97440705]), tolerance=1e-5)
        assert_near_equal(p['rho_vapor'], np.array([2.6756274, 4.8593, 8.26745558, 13.40464169, 21.03778235, 32.43929276]), tolerance=1e-5)
        assert_near_equal(p['vapor_pressure'], np.array([327.98889865,  614.9, 1065.92300458, 1733.70063068, 2677.30942723, 3966.79472967]), tolerance=1e-5)
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class QMaxHeatPipeTestCase(unittest.TestCase):
    """
    Basic test for QMaxHeatPipe component to ensure no drastic changes in outputs
    """
    def test_default_settings(self):
        nn = 3
        p = Problem()
        comp = p.model.add_subsystem('test', hp.QMaxHeatPipe(num_nodes=nn), promotes=['*'])
        comp.set_input_defaults('temp', units='degC', val=np.linspace(30, 60, nn))
        comp.set_input_defaults('length', units='m', val=10.22)
        comp.set_input_defaults('inner_diam', units='inch', val=.902)
        comp.set_input_defaults('design_temp', units='degC', val=40)
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        
        assert_near_equal(p['q_max'], np.array([4936.75193703, 5074.04485581, 5081.04977578]), tolerance=1e-5)
        assert_near_equal(p['heat_pipe_weight'], 0.51463886, tolerance=1e-5)
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class QMaxAnalyticalPartTestCase(unittest.TestCase):
    """
    Basic test for QMaxAnalyticalPart component to ensure no drastic changes in outputs
    """
    def test_default_settings(self):
        nn = 3
        p = Problem()
        p.model.add_subsystem('test', hp.QMaxAnalyticalPart(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        
        assert_near_equal(p['q_max'], np.ones(nn)*875.86211677, tolerance=1e-5)
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
