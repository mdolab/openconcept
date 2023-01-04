import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem
from openconcept.thermal import FlowSplit, FlowCombine


class FlowSplitTestCase(unittest.TestCase):
    """
    Test the FlowSplit component
    """

    def test_default_settings(self):
        p = Problem()
        p.model.add_subsystem("test", FlowSplit(), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        assert_near_equal(p["mdot_out_A"], np.array([0.5]))
        assert_near_equal(p["mdot_out_B"], np.array([0.5]))

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        nn = 4
        p = Problem()
        p.model.add_subsystem("test", FlowSplit(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)

        p["mdot_in"] = np.array([-10.0, 0.0, 10.0, 10.0])
        p["mdot_split_fraction"] = np.array([0.0, 0.4, 0.4, 1.0])

        p.run_model()

        assert_near_equal(p["mdot_out_A"], np.array([0.0, 0.0, 4.0, 10.0]))
        assert_near_equal(p["mdot_out_B"], np.array([-10.0, 0.0, 6.0, 0.0]))

    def test_warnings(self):
        nn = 4
        p = Problem()
        p.model.add_subsystem("test", FlowSplit(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)

        p["mdot_in"] = np.array([-10.0, 0.0, 10.0, 10.0])
        p["mdot_split_fraction"] = np.array([-0.0001, 0.4, 0.4, 1.0])
        with self.assertRaises(RuntimeWarning):
            p.run_model()

        p["mdot_split_fraction"] = np.array([1.0001, 0.4, 0.4, 1.0])
        with self.assertRaises(RuntimeWarning):
            p.run_model()


class FlowCombineTestCase(unittest.TestCase):
    """
    Test the FlowCombine component
    """

    def test_default_settings(self):
        p = Problem()
        p.model.add_subsystem("test", FlowCombine(), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        assert_near_equal(p["mdot_out"], np.array([2.0]))
        assert_near_equal(p["T_out"], np.array([1.0]))

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        nn = 4
        p = Problem()
        p.model.add_subsystem("test", FlowCombine(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)

        p["mdot_in_A"] = np.array([0.0, 5.0, 10.0, 10.0])
        p["mdot_in_B"] = np.array([1.0, 0.0, 5.0, 10.0])
        p["T_in_A"] = np.array([1.0, 10.0, 30.0, 500.0])
        p["T_in_B"] = np.array([1.0, 150.0, 60.0, 100.0])

        p.run_model()

        assert_near_equal(p["mdot_out"], np.array([1.0, 5.0, 15.0, 20.0]))
        assert_near_equal(p["T_out"], np.array([1.0, 10.0, 40.0, 300.0]))
