from __future__ import division
import unittest
import pytest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.utilities.math.integrals import NewIntegrator

class MultiPhaseIntegratorTestGroup(Group):
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
        self.options.declare('num_nodes',default=11, desc="Number of nodes per segment")
        self.options.declare('integrator',default='simpson', desc="Which simpson integrator to use")
        self.options.declare('time_setup',default='dt')

    def setup(self):
        quantity_units = self.options['quantity_units']
        diff_units = self.options['diff_units']
        num_nodes = self.options['num_nodes']
        integrator_option = self.options['integrator']
        time_setup = self.options['time_setup']

        num_nodes = num_nodes

        iv = self.add_subsystem('iv', IndepVarComp())
        integ = NewIntegrator(diff_units=diff_units, num_nodes=num_nodes, method=integrator_option, time_setup=time_setup)
        integ.add_integrand('q', rate_name='dqdt', start_name='q_initial', end_name='q_final',
                            units=quantity_units)
        self.add_subsystem('integral', integ)
        if quantity_units is None and diff_units is None:
            rate_units = None
        elif quantity_units is None:
            rate_units = '(' + diff_units +')** -1'
        elif diff_units is None:
            rate_units = quantity_units
        else:
            rate_units = '('+quantity_units+') / (' + diff_units +')'

        iv.add_output('rate_to_integrate', val=np.ones((num_nodes,)), units=rate_units)
        iv.add_output('initial_value', val=0, units=quantity_units)

        self.connect('iv.rate_to_integrate','integral.dqdt')
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

class IntegratorEveryNodeCommonTestCases(object):
    """
    A common set of test cases for the integrator component
    """

    def test_uniform_single_phase_no_units(self):
        prob = Problem(MultiPhaseIntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        assert_near_equal(prob['integral.q'], np.linspace(0, nn_tot-1, nn_tot), tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), nn_tot-1, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_linear_single_phase_no_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = x
        f = x ** 2 / 2

        prob = Problem(MultiPhaseIntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob['integral.q'], f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_single_phase_no_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(MultiPhaseIntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob['integral.q'], f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_single_phase_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(MultiPhaseIntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_duration(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(MultiPhaseIntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s',time_setup='duration'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    @pytest.mark.filterwarnings("ignore:Input*:UserWarning")
    def test_quadratic_bounds(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(MultiPhaseIntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     quantity_units='kg', diff_units='s',time_setup='bounds'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_single_phase_no_rate_units(self):
        num_nodes = self.num_nodes
        nn_tot = num_nodes
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(MultiPhaseIntegratorTestGroup(num_nodes=self.num_nodes, integrator=self.integrator,
                                                     diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_near_equal(prob.get_val('integral.q', units=None), f, tolerance=1e-14)
        assert_near_equal(prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

class SimpsonIntegratorEveryNode5PtTestCases(unittest.TestCase, IntegratorEveryNodeCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 11
        self.integrator = 'simpson'
        super(SimpsonIntegratorEveryNode5PtTestCases, self).__init__(*args, **kwargs)

class SimpsonIntegratorEveryNode3PtTestCases(unittest.TestCase, IntegratorEveryNodeCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 7
        self.integrator = 'simpson'
        super(SimpsonIntegratorEveryNode3PtTestCases, self).__init__(*args, **kwargs)

class BDFIntegratorEveryNode5PtTestCases(unittest.TestCase, IntegratorEveryNodeCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 11
        self.integrator = 'bdf3'
        super(BDFIntegratorEveryNode5PtTestCases, self).__init__(*args, **kwargs)

class BDFIntegratorEveryNode3PtTestCases(unittest.TestCase, IntegratorEveryNodeCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = 7
        self.integrator = 'bdf3'
        super(BDFIntegratorEveryNode3PtTestCases, self).__init__(*args, **kwargs)