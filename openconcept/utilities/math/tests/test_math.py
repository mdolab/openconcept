from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.utilities.math.simpson_integration import IntegrateQuantity
from openconcept.utilities.math.derivatives import FirstDerivative

class SimpsonTestGroup(Group):
    """An OpenMDAO group to test the simpson integration capability
    """
    def initialize(self):
        self.options.declare('n_simp_intervals',default=1,desc="Number of Simpson intevals to use")
    def setup(self):
        ni = self.options['n_simp_intervals']
        iv = self.add_subsystem('iv', IndepVarComp())
        self.add_subsystem('integrate', IntegrateQuantity(num_intervals=ni,diff_units='s',quantity_units='m'),promotes_outputs=['*'])
        iv.add_output('start_pt', val=0, units='s')
        iv.add_output('end_pt', val=1, units='s')
        iv.add_output('function', val=np.ones(2*ni+1), units='m/s')
        self.connect('iv.start_pt','integrate.lower_limit')
        self.connect('iv.end_pt','integrate.upper_limit')
        self.connect('iv.function','integrate.rate')
        #output is 'delta quantity'

class SimpsonTestCase(unittest.TestCase):
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


class FirstDerivativeTestGroup(Group):
    """An OpenMDAO group to test the differentiation tools

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
        The units of the quantity q being differentiated
    diff_units : str
        The units of the derivative being taken (none by default)
    order : int
        Order of accuracy (default 4). May also choose 2 for 2nd order accuracy and sparser structure
    """

    def initialize(self):
        self.options.declare('segment_names', default=None, desc="Names of differentiation segments")
        self.options.declare('quantity_units',default=None, desc="Units of the quantity being differentiated")
        self.options.declare('diff_units',default=None, desc="Units of the differential")
        self.options.declare('num_intervals',default=5, desc="Number of Simpsons rule intervals per segment")
        self.options.declare('order',default=4, desc="Order of accuracy")

    def setup(self):
        segment_names = self.options['segment_names']
        quantity_units = self.options['quantity_units']
        diff_units = self.options['diff_units']
        order = self.options['order']
        n_int_per_seg = self.options['num_intervals']
        if segment_names is None:
            nn_tot = (2*n_int_per_seg + 1)
        else:
            nn_tot = (2*n_int_per_seg + 1) * len(segment_names)
        iv = self.add_subsystem('iv', IndepVarComp())
        self.add_subsystem('derivative', FirstDerivative(segment_names=segment_names, quantity_units=quantity_units,
                                                         diff_units=diff_units, num_intervals=n_int_per_seg, order=order))
        iv.add_output('quant_to_diff', val=np.ones((nn_tot,)), units=quantity_units)
        self.connect('iv.quant_to_diff','derivative.q')

        if segment_names is None:
            iv.add_output('dt', val=1, units=diff_units)
            self.connect('iv.dt', 'derivative.dt')
        else:
            for segment_name in segment_names:
                iv.add_output(segment_name + '|dt', val=1, units=diff_units)
                self.connect('iv.'+segment_name + '|dt','derivative.'+segment_name + '|dt')

class FirstDerivCommonTestCases(object):
    """
    These test cases apply to both the 2nd order and 4th order accurate cases
    """

    def test_uniform_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        assert_rel_error(self, prob['derivative.dqdt'], np.zeros((nn_tot,)),tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e2)

    def test_linear_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        prob['iv.quant_to_diff'] = np.linspace(0,1,nn_tot)
        prob['iv.dt'] = 1 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob['derivative.dqdt'], np.ones((nn_tot,)),tolerance=1e-15)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e2)

    def test_quadratic_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, 2, nn_tot)
        f_test = 5*x ** 2+ 7*x -3
        fp_exact = 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob['derivative.dqdt'], fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quadratic_single_phase_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, quantity_units='m', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, 2, nn_tot)
        f_test = 5*x ** 2+ 7*x -3
        fp_exact = 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob.get_val('derivative.dqdt','m/s'), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quadratic_single_phase_diff_units_only(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, 2, nn_tot)
        f_test = 5*x ** 2+ 7*x -3
        fp_exact = 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob.get_val('derivative.dqdt','s ** -1'), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quadratic_single_named_phase_units(self):
        prob = Problem(FirstDerivativeTestGroup(segment_names=['cruise'], order=self.order, quantity_units='m', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, 2, nn_tot)
        f_test = 5*x ** 2+ 7*x -3
        fp_exact = 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.cruise|dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob.get_val('derivative.dqdt','m/s'), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quadratic_multi_phase_units(self):
        prob = Problem(FirstDerivativeTestGroup(segment_names=['climb','cruise','descent'],
                                                order=self.order, quantity_units='m', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_seg = (n_int_per_seg*2 + 1)
        nn_tot = (n_int_per_seg*2 + 1) * 3
        x = np.concatenate([np.linspace(0, 2, nn_seg), np.linspace(2, 3, nn_seg), np.linspace(3, 6, nn_seg)])
        f_test = 5*x ** 2+ 7*x -3
        fp_exact = 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.climb|dt'] = 2 / (nn_seg - 1)
        prob['iv.cruise|dt'] = 1 / (nn_seg - 1)
        prob['iv.descent|dt'] = 3 / (nn_seg - 1)
        prob.run_model()
        assert_rel_error(self, prob.get_val('derivative.dqdt','m/s'), fp_exact, tolerance=1e-12)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-6, rtol=1e-6)

    def test_quadratic_multi_phase_units_7int(self):
        prob = Problem(FirstDerivativeTestGroup(segment_names=['climb','cruise','descent'],
                                                order=self.order, quantity_units='m', diff_units='s', num_intervals=7))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 7
        nn_seg = (n_int_per_seg*2 + 1)
        nn_tot = (n_int_per_seg*2 + 1) * 3
        x = np.concatenate([np.linspace(0, 2, nn_seg), np.linspace(2, 3, nn_seg), np.linspace(3, 6, nn_seg)])
        f_test = 5*x ** 2+ 7*x -3
        fp_exact = 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.climb|dt'] = 2 / (nn_seg - 1)
        prob['iv.cruise|dt'] = 1 / (nn_seg - 1)
        prob['iv.descent|dt'] = 3 / (nn_seg - 1)
        prob.run_model()
        assert_rel_error(self, prob.get_val('derivative.dqdt','m/s'), fp_exact, tolerance=1e-12)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-6, rtol=1e-6)

class FirstDerivativeSecondOrderTestCases(unittest.TestCase, FirstDerivCommonTestCases):
    """
    Only run the common test cases using second order accuracy because
    it cannot differentiate the quartic accurately
    """
    def __init__(self, *args, **kwargs):
        self.order = 2
        super(FirstDerivativeSecondOrderTestCases, self).__init__(*args, **kwargs)

class FirstDerivativeFourthOrderTestCases(unittest.TestCase, FirstDerivCommonTestCases):
    """
    Add some additional fourth order polynomial test cases.
    """
    def __init__(self, *args, **kwargs):
        self.order = 4
        super(FirstDerivativeFourthOrderTestCases, self).__init__(*args, **kwargs)

    def test_quartic_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, 2, nn_tot)
        f_test = x ** 4 - 3*x **3 + 5*x ** 2+ 7*x -3
        fp_exact = 4*x ** 3 - 9*x ** 2 + 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob['derivative.dqdt'], fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quartic_negative_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(-3, -1, nn_tot)
        f_test = x ** 4 - 3*x **3 + 5*x ** 2+ 7*x -3
        fp_exact = 4*x ** 3 - 9*x ** 2 + 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob['derivative.dqdt'], fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-6, rtol=1e-6)

    def test_quartic_single_phase_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, quantity_units='m', diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, 2, nn_tot)
        f_test = x ** 4 - 3*x **3 + 5*x ** 2+ 7*x -3
        fp_exact = 4*x ** 3 - 9*x ** 2 + 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob.get_val('derivative.dqdt',units='m/s'), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quartic_single_phase_diff_units_only(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, diff_units='s'))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, 2, nn_tot)
        f_test = x ** 4 - 3*x **3 + 5*x ** 2+ 7*x -3
        fp_exact = 4*x ** 3 - 9*x ** 2 + 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob.get_val('derivative.dqdt',units='s ** -1'), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quartic_single_phase_qty_units_only(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, quantity_units='m'))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = (n_int_per_seg*2 + 1)
        x = np.linspace(0, 2, nn_tot)
        f_test = x ** 4 - 3*x **3 + 5*x ** 2+ 7*x -3
        fp_exact = 4*x ** 3 - 9*x ** 2 + 10*x + 7
        prob['iv.quant_to_diff'] = f_test
        prob['iv.dt'] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_rel_error(self, prob.get_val('derivative.dqdt',units='m'), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

