from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.utilities.math.integrals import IntegrateQuantityEveryNode

class MultiPhaseIntegratorTestGroup(Group):
    """An OpenMDAO group to test the every-node integrator component

    Options
    -------
    segment_names : list
        A list of str with the names of the individual segments
        By default, if no segment_names are provided, one segment will be assumed and segment|dt will just be named "dt"
    num_intervals : int
        The number of Simpson intervals per segment
        The total number of points per segment is 2N + 1 where N = num_intervals
        The total length of the vector q is n_segments * (2N + 1)
    quantity_units : str
        The units of the integral quantity q (NOT the rate)
    diff_units : str
        The units of the integrand (none by default)
    """

    def initialize(self):
        self.options.declare('segment_names', default=None, desc="Names of differentiation segments")
        self.options.declare('quantity_units',default=None, desc="Units of the quantity being differentiated")
        self.options.declare('diff_units',default=None, desc="Units of the differential")
        self.options.declare('num_intervals',default=5, desc="Number of Simpsons rule intervals per segment")

    def setup(self):
        segment_names = self.options['segment_names']
        quantity_units = self.options['quantity_units']
        diff_units = self.options['diff_units']
        n_int_per_seg = self.options['num_intervals']
        if segment_names is None:
            nn_tot = (2*n_int_per_seg + 1)
        else:
            nn_tot = (2*n_int_per_seg + 1) * len(segment_names)
        iv = self.add_subsystem('iv', IndepVarComp())

        self.add_subsystem('integral', IntegrateQuantityEveryNode(segment_names=segment_names, quantity_units=quantity_units,
                                                         diff_units=diff_units, num_intervals=n_int_per_seg))
        if quantity_units is None and diff_units is None:
            rate_units = None
        elif quantity_units is None:
            rate_units = '(' + diff_units +')** -1'
        elif diff_units is None:
            rate_units = quantity_units
        else:
            rate_units = '('+quantity_units+') / (' + diff_units +')'

        iv.add_output('rate_to_integrate', val=np.ones((nn_tot,)), units=rate_units)
        iv.add_output('initial_value', val=0, units=quantity_units)

        self.connect('iv.rate_to_integrate','integral.dqdt')
        self.connect('iv.initial_value', 'integral.q_initial')

        if segment_names is None:
            iv.add_output('dt', val=1, units=diff_units)
            self.connect('iv.dt', 'integral.dt')
        else:
            for segment_name in segment_names:
                iv.add_output(segment_name + '|dt', val=1, units=diff_units)
                self.connect('iv.'+segment_name + '|dt','integral.'+segment_name + '|dt')

class IntegratorEveryNodeCommonTestCases(object):
    """
    A common set of test cases for the integrator component
    """

    def test_uniform_single_phase_no_units(self):
        prob = Problem(MultiPhaseIntegratorTestGroup(num_intervals=self.num_intervals))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        n_int_per_seg = self.num_intervals
        nn_tot = (n_int_per_seg*2 + 1)
        assert_rel_error(self, prob['integral.q'], np.linspace(0, nn_tot-1, nn_tot), tolerance=1e-14)
        assert_rel_error(self, prob.get_val('integral.q_final', units=None), nn_tot-1, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_linear_single_phase_no_units(self):
        n_int_per_seg = self.num_intervals
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = x
        f = x ** 2 / 2

        prob = Problem(MultiPhaseIntegratorTestGroup(num_intervals=self.num_intervals))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_rel_error(self, prob['integral.q'], f, tolerance=1e-14)
        assert_rel_error(self, prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_single_phase_no_units(self):
        n_int_per_seg = self.num_intervals
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(MultiPhaseIntegratorTestGroup(num_intervals=self.num_intervals))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_rel_error(self, prob['integral.q'], f, tolerance=1e-14)
        assert_rel_error(self, prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_single_phase_units(self):
        n_int_per_seg = self.num_intervals
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(MultiPhaseIntegratorTestGroup(num_intervals=self.num_intervals,
                                                     quantity_units='kg', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_rel_error(self, prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_rel_error(self, prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_single_phase_no_rate_units(self):
        n_int_per_seg = self.num_intervals
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, nn_tot-1, nn_tot)
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x

        prob = Problem(MultiPhaseIntegratorTestGroup(num_intervals=self.num_intervals,
                                                     diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_rel_error(self, prob.get_val('integral.q', units=None), f, tolerance=1e-14)
        assert_rel_error(self, prob.get_val('integral.q_final', units=None), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_three_phase_units_equal_dt(self):
        n_int_per_seg = self.num_intervals
        nn_tot = (n_int_per_seg*2 + 1)
        x1 = np.linspace(0, nn_tot-1, nn_tot)
        x2 = np.linspace(nn_tot-1, 2*(nn_tot-1), nn_tot)
        x3 = np.linspace(2*(nn_tot-1), 3*(nn_tot-1), nn_tot)
        x = np.concatenate([x1, x2, x3])
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x
        prob = Problem(MultiPhaseIntegratorTestGroup(segment_names=['climb','cruise','descent'],
                                                     num_intervals=self.num_intervals,
                                                     quantity_units='kg', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob.run_model()
        assert_rel_error(self, prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_rel_error(self, prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_three_phase_units_unequal_dt(self):
        n_int_per_seg = self.num_intervals
        nn_tot = (n_int_per_seg*2 + 1)
        x1 = np.linspace(0, nn_tot-1, nn_tot)
        x2 = np.linspace(nn_tot-1, 3*(nn_tot-1), nn_tot)
        x3 = np.linspace(3*(nn_tot-1), 6*(nn_tot-1), nn_tot)
        x = np.concatenate([x1, x2, x3])
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x
        prob = Problem(MultiPhaseIntegratorTestGroup(segment_names=['climb','cruise','descent'],
                                                     num_intervals=self.num_intervals,
                                                     quantity_units='kg', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob['iv.climb|dt'] = 1
        prob['iv.cruise|dt'] = 2
        prob['iv.descent|dt'] = 3
        prob.run_model()
        assert_rel_error(self, prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_rel_error(self, prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

    def test_quadratic_three_phase_units_unequal_dt_initial_val(self):
        n_int_per_seg = self.num_intervals
        nn_tot = (n_int_per_seg*2 + 1)
        x1 = np.linspace(0, nn_tot-1, nn_tot)
        x2 = np.linspace(nn_tot-1, 3*(nn_tot-1), nn_tot)
        x3 = np.linspace(3*(nn_tot-1), 6*(nn_tot-1), nn_tot)
        x = np.concatenate([x1, x2, x3])
        C = 10.
        fprime = 4 * x **2 - 8*x + 5
        f = 4 * x ** 3 / 3 - 8 * x ** 2 / 2 + 5*x + C
        prob = Problem(MultiPhaseIntegratorTestGroup(segment_names=['climb','cruise','descent'],
                                                     num_intervals=self.num_intervals,
                                                     quantity_units='kg', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        prob['iv.rate_to_integrate'] = fprime
        prob['iv.climb|dt'] = 1
        prob['iv.cruise|dt'] = 2
        prob['iv.descent|dt'] = 3
        prob['iv.initial_value'] = C
        prob.run_model()
        assert_rel_error(self, prob.get_val('integral.q', units='kg'), f, tolerance=1e-14)
        assert_rel_error(self, prob.get_val('integral.q_final', units='kg'), f[-1], tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e0)

class IntegratorEveryNode5PtTestCases(unittest.TestCase, IntegratorEveryNodeCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_intervals = 5
        super(IntegratorEveryNode5PtTestCases, self).__init__(*args, **kwargs)

class IntegratorEveryNode3PtTestCases(unittest.TestCase, IntegratorEveryNodeCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.num_intervals = 3
        super(IntegratorEveryNode3PtTestCases, self).__init__(*args, **kwargs)

