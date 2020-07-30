from __future__ import division
import unittest
import pytest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.utilities.math.integrals import Integrator

class IntegratorTestGroup(Group):
    """An OpenMDAO group to test the every-node integrator component

    Options
    -------
    num_nodes : int
        Number of analysis points per segment
    quantity_units : str
        The units of the integral quantity q (NOT the rate)
    diff_units : str
        The units of the integrand (none by default)
    integrator : str
        Which integration scheme to use (default 'simpson')
    """

    def initialize(self):
        self.options.declare('quantity_units',default=None, desc="Units of the quantity being differentiated")
        self.options.declare('diff_units',default=None, desc="Units of the differential")
        self.options.declare('rate_units',default=None, desc="Units of the rate")
        self.options.declare('num_nodes',default=11, desc="Number of nodes per segment")
        self.options.declare('integrator',default='simpson', desc="Which simpson integrator to use")
        self.options.declare('time_setup',default='dt')
        self.options.declare('second_integrand',default=False)
        self.options.declare('zero_start', default=False)
        self.options.declare('final_only', default=False)
        self.options.declare('test_auto_names', default=False)
        self.options.declare('val', default=0.0)

    def setup(self):
        quantity_units = self.options['quantity_units']
        diff_units = self.options['diff_units']
        rate_units = self.options['rate_units']
        num_nodes = self.options['num_nodes']
        integrator_option = self.options['integrator']
        time_setup = self.options['time_setup']
        second_integrand = self.options['second_integrand']
        zero_start = self.options['zero_start']
        final_only = self.options['final_only']
        test_auto_names = self.options['test_auto_names']
        val = self.options['val']
        num_nodes = num_nodes

        iv = self.add_subsystem('iv', IndepVarComp())
        integ = Integrator(diff_units=diff_units, num_nodes=num_nodes, method=integrator_option, 
                              time_setup=time_setup)
        if not test_auto_names:
            integ.add_integrand('q', rate_name='dqdt', start_name='q_initial', end_name='q_final',
                                units=quantity_units, rate_units=rate_units, zero_start=zero_start,
                                final_only=final_only, val=val)
        else:
            integ.add_integrand('q', units=quantity_units, rate_units=rate_units, zero_start=zero_start,
                                final_only=final_only)
        if second_integrand:
            integ.add_integrand('q2', rate_name='dq2dt', start_name='q2_initial', end_name='q2_final', units='kJ')
            iv.add_output('rate_to_integrate_2', val=np.ones((num_nodes,)), units='kW')
            iv.add_output('initial_value_2', val=0., units='kJ')
            self.connect('iv.rate_to_integrate_2', 'integral.dq2dt')
            self.connect('iv.initial_value_2', 'integral.q2_initial')
        self.add_subsystem('integral', integ)

        if rate_units and quantity_units:
            # overdetermined and possibly inconsistent
            pass
        elif not rate_units and not quantity_units:
            if diff_units:
                rate_units =  '(' + diff_units +')** -1'
        elif not rate_units:
            # solve for rate_units in terms of quantity_units
            if not diff_units:
                rate_units = quantity_units
            else:
                rate_units = '('+quantity_units+') / (' + diff_units +')'
        elif not quantity_units:
            # solve for quantity units in terms of rate units
            if not diff_units:
                quantity_units = rate_units
            else:
                quantity_units = '('+rate_units+')*('+diff_units+')'

        iv.add_output('rate_to_integrate', val=np.ones((num_nodes,)), units=rate_units)
        iv.add_output('initial_value', val=0, units=quantity_units)

        if not test_auto_names:
            self.connect('iv.rate_to_integrate','integral.dqdt')
        else:
            self.connect('iv.rate_to_integrate','integral.q_rate')
        if not zero_start:
            self.connect('iv.initial_value', 'integral.q_initial')

        if time_setup == 'dt':
            iv.add_output('dt', val=1, units=diff_units)
            self.connect('iv.dt', 'integral.dt')
        elif time_setup == 'duration':
            iv.add_output('duration', val=1*(num_nodes-1), units=diff_units)
            self.connect('iv.duration', 'integral.duration')
        elif time_setup == 'bounds':
            iv.add_output('t_initial', val=2, units=diff_units)
            iv.add_output('t_final', val=2 + 1*(num_nodes-1), units=diff_units)
            self.connect('iv.t_initial','integral.t_initial')
            self.connect('iv.t_final','integral.t_final')

class IntegratorCommonTestCases(object):
    """
    A common set of test cases for the integrator component
    """

    def test_uniform_no_units(self):
        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        assert_near_equal(prob['integral.q'], np.linspace(0, nn_tot-1, nn_tot), tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), nn_tot-1, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_linear_no_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = x
        f = x ** 2 / 2

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob['integral.q'], f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_no_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob['integral.q'], f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_machine_zero_rate(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 0.0 * x
        f = 0.0 * x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob['integral.q'], f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_auto_names(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(test_auto_names=True, num_nodes=self.num_nodes, integrator=self.integrator))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob['integral.q'], f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_qty_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_qty_units_nonzero_start(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x + 25.2

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob['iv.initial_value'] = 25.2
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_qty_units_zero_start(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s',
                                                     zero_start=True))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)
        with self.assertRaises(KeyError):
            # ensure the input hasn't been created
            prob['integral.q_initial'] = -1.0

    def test_quadratic_qty_units_final_only(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s',
                                                     final_only=True))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        with self.assertRaises(KeyError):
            # this output shouldn't exist
            assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_rate_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     rate_units='kg/s', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_multiple_integrands(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime1 = 4 * x **2 - 8*x + 5
        f1 = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x
        fprime2 = -2 * x **2 + 10.5*x - 2
        f2 = -2 * x ** 3 / 3 + 10.5*x**2 /2 - 2*x
        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     rate_units='kg/s', diff_units='s', second_integrand=True))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime1
        prob['iv.rate_to_integrate_2'] = fprime2
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f1, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f1[-1], tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q2', units='kJ'), f2, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q2_final', units='kJ'), f2[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)


    def test_quadratic_both_units_correct(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x
        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                                rate_units='kg/s', quantity_units='kg', diff_units='s'))
        with self.assertRaises(ValueError) as cm:
            prob.setup(check=True)
            
        msg = ('Specify either quantity units or rate units, but not both')
        self.assertEqual(str(cm.exception), msg)

    def test_quadratic_duration(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s',time_setup='duration'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_bounds(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s',time_setup='bounds'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_machine_zero_bounds(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 0.0 * x
        f = 0.0*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s',time_setup='bounds'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_no_rate_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units=None), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

class SimpsonIntegrator5PtTestCases(unittest.TestCase, IntegratorCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 11
        self.integrator = 'simpson'
        super(SimpsonIntegrator5PtTestCases, self).__init__(*args, **kwargs)

class SimpsonIntegrator3PtTestCases(unittest.TestCase, IntegratorCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 7
        self.integrator = 'simpson'
        super(SimpsonIntegrator3PtTestCases, self).__init__(*args, **kwargs)

@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value")
class SimpsonIntegrator1PtTestCases(unittest.TestCase, IntegratorCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 1
        self.integrator = 'simpson'
        super(SimpsonIntegrator1PtTestCases, self).__init__(*args, **kwargs)

class BDFIntegrator5PtTestCases(unittest.TestCase, IntegratorCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 11
        self.integrator = 'bdf3'
        super(BDFIntegrator5PtTestCases, self).__init__(*args, **kwargs)

class BDFIntegrator3PtTestCases(unittest.TestCase, IntegratorCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 7
        self.integrator = 'bdf3'
        super(BDFIntegrator3PtTestCases, self).__init__(*args, **kwargs)

class EdgeCaseTestCases(unittest.TestCase):
    def test_quadratic_even_num_nodes(self):
        num_nodes = 10
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x
        with self.assertRaises(ValueError) as cm:
            prob = Problem(IntegratorTestGroup(num_nodes=num_nodes, integrator='simpson',
                                                                    rate_units='kg/s', diff_units='s'))
            prob.setup()
            
        msg = ('num_nodes is ' +str(num_nodes) + ' and must be odd')
        self.assertEqual(str(cm.exception), msg)

    def test_default_value_scalar(self):
        num_nodes = self.num_nodes = 11
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator='simpson',
                                                     quantity_units='kg', diff_units='s',time_setup='duration',
                                                     val=5.0))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        # DO NOT RUN THE MODEL
        assert_near_equal(prob.get_val('integral.q', units='kg'), 5.0*np.ones((num_nodes,)), tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), 5.0, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_default_value_vector(self):
        num_nodes = self.num_nodes = 11
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(IntegratorTestGroup(num_nodes=self.num_nodes, integrator='simpson',
                                                     quantity_units='kg', diff_units='s',time_setup='duration',
                                                     val=5.0*np.linspace(0.0, 1.0, num_nodes)))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        # DO NOT RUN THE MODEL
        assert_near_equal(prob.get_val('integral.q', units='kg'), 5.0*np.linspace(0.0, 1.0, num_nodes), tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), 5.0, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)
