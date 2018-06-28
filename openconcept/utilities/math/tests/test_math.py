import unittest
import numpy as np 
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.utilities.math.simpson_integration import IntegrateQuantity

class SimpsonTestGroup(Group):
    """This computes pressure, temperature, and density for a given altitude at ISA condtions. Also true airspeed from equivalent ~ indicated airspeed
    """
    def initialize(self):
        self.options.declare('n_simp_intervals',default=1,desc="Number of mission analysis points to run")
    def setup(self):
        ni = self.options['n_simp_intervals']
        iv = self.add_subsystem('iv', IndepVarComp())
        self.add_subsystem('integrate', IntegrateQuantity(num_intervals=ni,diff_units='s',quantity_units='m'),promotes_outputs=['*'])
        iv.add_output('start_pt',val=0,units='s')
        iv.add_output('end_pt',val=1,units='s')
        iv.add_output('function',val=np.ones(2*ni+1),units='m/s')
        self.connect('iv.start_pt','integrate.lower_limit')
        self.connect('iv.end_pt','integrate.upper_limit')
        self.connect('iv.function','integrate.rate')
        #output is 'delta quantity'

class VectorAtmosTestCase(unittest.TestCase):
    def test_uniform_single(self):
        prob = Problem(SimpsonTestGroup(n_simp_intervals=1))
        prob.setup(check=True)
        prob.run_model()
        assert_rel_error(self,prob['delta_quantity'],1.,tolerance=1e-15)
        partials = prob.check_partials(method='fd',compact_print=True)
        assert_check_partials(partials)

    def test_uniform(self):
        prob = Problem(SimpsonTestGroup(n_simp_intervals=5))
        prob.setup(check=True)
        prob.run_model()
        assert_rel_error(self,prob['delta_quantity'],1.,tolerance=1e-15)
        partials = prob.check_partials(method='fd',compact_print=True)
        assert_check_partials(partials)

    def test_endpoints(self):
        prob = Problem(SimpsonTestGroup(n_simp_intervals=5))
        prob.setup(check=False)
        prob['iv.end_pt'] = 2
        prob.run_model()
        assert_rel_error(self,prob['delta_quantity'],2.,tolerance=1e-15)    
        
    def test_function_level(self):
        prob = Problem(SimpsonTestGroup(n_simp_intervals=5))
        prob.setup(check=False)
        prob['iv.function'] = 3*np.ones(2*5+1)
        prob.run_model()
        assert_rel_error(self,prob['delta_quantity'],3.,tolerance=1e-15)

    def test_trig_function_approx(self):
        prob = Problem(SimpsonTestGroup(n_simp_intervals=5))
        prob.setup(check=False)
        x = np.linspace(0,np.pi,2*5+1)
        prob['iv.end_pt'] = np.pi
        prob['iv.function'] = np.sin(x)
        prob.run_model()
        assert_rel_error(self,prob['delta_quantity'],2.,tolerance=1e-4)
        partials = prob.check_partials(method='fd',compact_print=True)
        assert_check_partials(partials)

    def test_cubic_polynomial_exact(self):
        prob = Problem(SimpsonTestGroup(n_simp_intervals=5))
        prob.setup(check=False)
        x = np.linspace(0,1,2*5+1)
        prob['iv.function'] = x**3-2*x**2+2.5*x-4
        prob.run_model()
        assert_rel_error(self,prob['delta_quantity'],-3.1666666666666666666666666666666666666,tolerance=1e-16)
        partials = prob.check_partials(method='fd',compact_print=True)
        assert_check_partials(partials)