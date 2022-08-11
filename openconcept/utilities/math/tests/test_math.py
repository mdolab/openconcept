import unittest
import pytest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.utilities import FirstDerivative


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
        self.options.declare("segment_names", default=None, desc="Names of differentiation segments")
        self.options.declare("quantity_units", default=None, desc="Units of the quantity being differentiated")
        self.options.declare("diff_units", default=None, desc="Units of the differential")
        self.options.declare("num_intervals", default=5, desc="Number of Simpsons rule intervals per segment")
        self.options.declare("order", default=4, desc="Order of accuracy")

    def setup(self):
        segment_names = self.options["segment_names"]
        quantity_units = self.options["quantity_units"]
        diff_units = self.options["diff_units"]
        order = self.options["order"]
        n_int_per_seg = self.options["num_intervals"]
        if segment_names is None:
            nn_tot = 2 * n_int_per_seg + 1
        else:
            nn_tot = (2 * n_int_per_seg + 1) * len(segment_names)
        iv = self.add_subsystem("iv", IndepVarComp())
        self.add_subsystem(
            "derivative",
            FirstDerivative(
                segment_names=segment_names,
                quantity_units=quantity_units,
                diff_units=diff_units,
                num_intervals=n_int_per_seg,
                order=order,
            ),
        )
        iv.add_output("quant_to_diff", val=np.ones((nn_tot,)), units=quantity_units)
        self.connect("iv.quant_to_diff", "derivative.q")

        if segment_names is None:
            iv.add_output("dt", val=1, units=diff_units)
            self.connect("iv.dt", "derivative.dt")
        else:
            for segment_name in segment_names:
                iv.add_output(segment_name + "|dt", val=1, units=diff_units)
                self.connect("iv." + segment_name + "|dt", "derivative." + segment_name + "|dt")


class FirstDerivCommonTestCases(object):
    """
    These test cases apply to both the 2nd order and 4th order accurate cases
    """

    def test_uniform_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        assert_near_equal(prob["derivative.dqdt"], np.zeros((nn_tot,)), tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e2)

    def test_linear_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        prob["iv.quant_to_diff"] = np.linspace(0, 1, nn_tot)
        prob["iv.dt"] = 1 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob["derivative.dqdt"], np.ones((nn_tot,)), tolerance=2e-15)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e2)

    def test_quadratic_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(0, 2, nn_tot)
        f_test = 5 * x**2 + 7 * x - 3
        fp_exact = 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob["derivative.dqdt"], fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quadratic_single_phase_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, quantity_units="m", diff_units="s"))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(0, 2, nn_tot)
        f_test = 5 * x**2 + 7 * x - 3
        fp_exact = 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob.get_val("derivative.dqdt", "m/s"), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quadratic_single_phase_diff_units_only(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, diff_units="s"))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(0, 2, nn_tot)
        f_test = 5 * x**2 + 7 * x - 3
        fp_exact = 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob.get_val("derivative.dqdt", "s ** -1"), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quadratic_single_named_phase_units(self):
        prob = Problem(
            FirstDerivativeTestGroup(segment_names=["cruise"], order=self.order, quantity_units="m", diff_units="s")
        )
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(0, 2, nn_tot)
        f_test = 5 * x**2 + 7 * x - 3
        fp_exact = 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.cruise|dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob.get_val("derivative.dqdt", "m/s"), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quadratic_multi_phase_units(self):
        prob = Problem(
            FirstDerivativeTestGroup(
                segment_names=["climb", "cruise", "descent"], order=self.order, quantity_units="m", diff_units="s"
            )
        )
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_seg = n_int_per_seg * 2 + 1
        x = np.concatenate([np.linspace(0, 2, nn_seg), np.linspace(2, 3, nn_seg), np.linspace(3, 6, nn_seg)])
        f_test = 5 * x**2 + 7 * x - 3
        fp_exact = 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.climb|dt"] = 2 / (nn_seg - 1)
        prob["iv.cruise|dt"] = 1 / (nn_seg - 1)
        prob["iv.descent|dt"] = 3 / (nn_seg - 1)
        prob.run_model()
        assert_near_equal(prob.get_val("derivative.dqdt", "m/s"), fp_exact, tolerance=1e-12)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-6, rtol=1e-6)

    def test_quadratic_multi_phase_units_7int(self):
        prob = Problem(
            FirstDerivativeTestGroup(
                segment_names=["climb", "cruise", "descent"],
                order=self.order,
                quantity_units="m",
                diff_units="s",
                num_intervals=7,
            )
        )
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 7
        nn_seg = n_int_per_seg * 2 + 1
        x = np.concatenate([np.linspace(0, 2, nn_seg), np.linspace(2, 3, nn_seg), np.linspace(3, 6, nn_seg)])
        f_test = 5 * x**2 + 7 * x - 3
        fp_exact = 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.climb|dt"] = 2 / (nn_seg - 1)
        prob["iv.cruise|dt"] = 1 / (nn_seg - 1)
        prob["iv.descent|dt"] = 3 / (nn_seg - 1)
        prob.run_model()
        assert_near_equal(prob.get_val("derivative.dqdt", "m/s"), fp_exact, tolerance=1e-12)
        partials = prob.check_partials(method="cs", compact_print=True)
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
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(0, 2, nn_tot)
        f_test = x**4 - 3 * x**3 + 5 * x**2 + 7 * x - 3
        fp_exact = 4 * x**3 - 9 * x**2 + 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob["derivative.dqdt"], fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quartic_negative_single_phase_no_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(-3, -1, nn_tot)
        f_test = x**4 - 3 * x**3 + 5 * x**2 + 7 * x - 3
        fp_exact = 4 * x**3 - 9 * x**2 + 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob["derivative.dqdt"], fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-6, rtol=1e-6)

    def test_quartic_single_phase_units(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, quantity_units="m", diff_units="s"))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(0, 2, nn_tot)
        f_test = x**4 - 3 * x**3 + 5 * x**2 + 7 * x - 3
        fp_exact = 4 * x**3 - 9 * x**2 + 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob.get_val("derivative.dqdt", units="m/s"), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    def test_quartic_single_phase_diff_units_only(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, diff_units="s"))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(0, 2, nn_tot)
        f_test = x**4 - 3 * x**3 + 5 * x**2 + 7 * x - 3
        fp_exact = 4 * x**3 - 9 * x**2 + 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob.get_val("derivative.dqdt", units="s ** -1"), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)

    @pytest.mark.filterwarnings("ignore:You have specified*:UserWarning")
    def test_quartic_single_phase_qty_units_only(self):
        prob = Problem(FirstDerivativeTestGroup(order=self.order, quantity_units="m"))
        prob.setup(check=True, force_alloc_complex=True)
        n_int_per_seg = 5
        nn_tot = n_int_per_seg * 2 + 1
        x = np.linspace(0, 2, nn_tot)
        f_test = x**4 - 3 * x**3 + 5 * x**2 + 7 * x - 3
        fp_exact = 4 * x**3 - 9 * x**2 + 10 * x + 7
        prob["iv.quant_to_diff"] = f_test
        prob["iv.dt"] = 2 / (nn_tot - 1)
        prob.run_model()
        assert_near_equal(prob.get_val("derivative.dqdt", units="m"), fp_exact, tolerance=1e-14)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials, atol=1e-8, rtol=1e-8)
