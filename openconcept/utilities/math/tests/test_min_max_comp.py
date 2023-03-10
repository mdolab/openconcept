import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem
from openconcept.utilities import MaxComp, MinComp


class MaxCompTestCase(unittest.TestCase):
    """
    Test the MaxComp component
    """

    def test_one_input(self):
        p = Problem()
        p.model.add_subsystem("test", MaxComp(), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([42.0]))
        p.run_model()
        assert_near_equal(p["max"], 42.0)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_multiple_inputs(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("test", MaxComp(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([42.0, 12.0, -3.0, 58.0, 7.0]))
        p.run_model()
        assert_near_equal(p["max"], 58.0)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_multiple_max(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("test", MaxComp(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([42.0, 58.0, -3.0, 58.0, 7.0]))
        p.run_model()
        assert_near_equal(p["max"], 58.0)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_multiple_very_close_max(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("test", MaxComp(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([-2.0, 2e-45, -3.0, 1e-45, -7.0]))
        p.run_model()
        assert_near_equal(p["max"], 2e-45, tolerance=1e-50)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_units(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("max_comp", MaxComp(num_nodes=nn, units="N"), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([42.0, 58.0, -3.0, 3.0, 7.0]), units="N")
        p.run_model()
        assert_near_equal(p["max"], 58.0)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class MinCompTestCase(unittest.TestCase):
    """
    Test the MinComp component
    """

    def test_one_input(self):
        p = Problem()
        p.model.add_subsystem("test", MinComp(), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([42.0]))
        p.run_model()
        assert_near_equal(p["min"], 42.0)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_multiple_inputs(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("test", MinComp(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([42.0, 12.0, -3.0, 58.0, 7.0]))
        p.run_model()
        assert_near_equal(p["min"], -3.0)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_multiple_min(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("comp", MinComp(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([42.0, 7.0, 30.0, 58.0, 7.0]))
        p.run_model()
        assert_near_equal(p["min"], 7.0)

        # Need to use fd here because cs doesn't support backward and forward does
        # not capture behavior of minimum
        partials = p.check_partials(method="fd", form="backward", compact_print=True)
        assert_check_partials(partials)

    def test_multiple_very_close_min(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("test", MinComp(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([2.0, 2e-45, 3.0, 1e-45, 7.0]))
        p.run_model()
        assert_near_equal(p["min"], 1e-45, tolerance=1e-50)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_units(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem("min_comp", MinComp(num_nodes=nn, units="N"), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val("array", np.array([42.0, 58.0, -3.0, 3.0, 7.0]), units="N")
        p.run_model()
        assert_near_equal(p["min"], -3.0)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)
