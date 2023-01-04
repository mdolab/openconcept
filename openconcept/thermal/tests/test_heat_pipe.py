import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem
from openconcept.thermal.heat_pipe import (
    HeatPipe,
    HeatPipeThermalResistance,
    HeatPipeVaporTempDrop,
    HeatPipeWeight,
    AmmoniaProperties,
    QMaxHeatPipe,
    QMaxAnalyticalPart,
)


class HeatPipeIntegrationTestCase(unittest.TestCase):
    """
    Test the HeatPipe group with everything integrated
    """

    def test_simple_scalar(self):
        nn = 1
        theta = 84.0
        prob = Problem()
        pipe = prob.model.add_subsystem("test", HeatPipe(num_nodes=nn, theta=theta), promotes=["*"])
        pipe.set_input_defaults("T_evap", units="degC", val=np.linspace(30, 30, nn))
        pipe.set_input_defaults("q", units="W", val=np.linspace(400, 400, nn))
        pipe.set_input_defaults("length", units="m", val=10.22)
        pipe.set_input_defaults("inner_diam", units="inch", val=0.902)
        pipe.set_input_defaults("n_pipes", val=1.0)
        pipe.set_input_defaults("T_design", units="degC", val=40)

        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob["q_max"], np.ones(nn) * 2807.01928115, tolerance=1e-5)
        assert_near_equal(prob["weight"], 0.51463886, tolerance=1e-5)
        assert_near_equal(prob["T_cond"], np.ones(nn) * 29.9441105, tolerance=1e-5)

        partials = prob.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)

    def test_simple_vector(self):
        nn = 5
        prob = Problem()
        pipe = prob.model.add_subsystem("test", HeatPipe(num_nodes=nn), promotes=["*"])
        pipe.set_input_defaults("T_evap", units="degC", val=np.linspace(30, 60, nn))
        pipe.set_input_defaults("q", units="W", val=np.linspace(400, 1000, nn))
        pipe.set_input_defaults("length", units="m", val=10.22)
        pipe.set_input_defaults("inner_diam", units="inch", val=0.902)
        pipe.set_input_defaults("n_pipes", val=1.0)
        pipe.set_input_defaults("T_design", units="degC", val=40)

        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(
            prob["q_max"],
            np.array([4936.70020611, 5022.24211291, 5073.99208835, 5095.24271287, 5080.99742858]),
            tolerance=1e-5,
        )
        assert_near_equal(prob["weight"], 0.51463886, tolerance=1e-5)
        assert_near_equal(
            prob["T_cond"], np.array([29.9441105, 37.4231564, 44.90220466, 52.38125436, 59.86030483]), tolerance=1e-5
        )

        partials = prob.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)

    def test_two_pipes(self):
        nn = 3

        # Run one and two pipes to compare results
        one = Problem()
        pipe = one.model.add_subsystem("test", HeatPipe(num_nodes=nn), promotes=["*"])
        pipe.set_input_defaults("T_evap", units="degC", val=np.linspace(30, 30, nn))
        pipe.set_input_defaults("q", units="W", val=np.linspace(200, 200, nn))
        pipe.set_input_defaults("length", units="m", val=10.22)
        pipe.set_input_defaults("inner_diam", units="inch", val=0.702)
        pipe.set_input_defaults("n_pipes", val=1.0)
        pipe.set_input_defaults("T_design", units="degC", val=70)

        one.setup(check=True, force_alloc_complex=True)
        one.run_model()

        partials = one.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)

        # Twice as many pipes with twice as much heat
        two = Problem()
        pipe = two.model.add_subsystem("test", HeatPipe(num_nodes=nn), promotes=["*"])
        pipe.set_input_defaults("T_evap", units="degC", val=np.linspace(30, 30, nn))
        pipe.set_input_defaults("q", units="W", val=np.linspace(400, 400, nn))
        pipe.set_input_defaults("length", units="m", val=10.22)
        pipe.set_input_defaults("inner_diam", units="inch", val=0.702)
        pipe.set_input_defaults("n_pipes", val=2.0)
        pipe.set_input_defaults("T_design", units="degC", val=70)

        two.setup(check=True, force_alloc_complex=True)
        two.run_model()

        partials = two.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)

        assert_near_equal(one["q_max"], two["q_max"] / 2)
        assert_near_equal(one["weight"], two["weight"] / 2)
        assert_near_equal(one["T_cond"], two["T_cond"])


class HeatPipeThermalResistanceTestCase(unittest.TestCase):
    """
    Basic test for HeatPipeThermalResistance component to ensure no drastic changes in outputs
    """

    def test_default_settings(self):
        nn = 3
        p = Problem()
        p.model.add_subsystem("test", HeatPipeThermalResistance(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p["thermal_resistance"], np.ones(nn) * 0.00076513, tolerance=1e-5)

        partials = p.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)


class HeatPipeVaporTempDropTestCase(unittest.TestCase):
    """
    Basic test for HeatPipeVaporTempDrop component to ensure no drastic changes in outputs
    """

    def test_default_settings(self):
        nn = 3
        p = Problem()
        p.model.add_subsystem("test", HeatPipeVaporTempDrop(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p["delta_T"], np.ones(nn) * 2.37127, tolerance=1e-5)

        partials = p.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)


class HeatPipeWeightTestCase(unittest.TestCase):
    """
    Basic test for HeatPipeWeight component to ensure no drastic changes in outputs
    """

    def test_default_settings(self):
        p = Problem()
        p.model.add_subsystem("test", HeatPipeWeight(), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p["heat_pipe_weight"], 0.04074404, tolerance=1e-5)
        assert_near_equal(p["wall_thickness"], 6.99300699e-05, tolerance=1e-5)
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class AmmoniaPropertiesTestCase(unittest.TestCase):
    """
    Basic test for AmmoniaProperties component to ensure no drastic changes in outputs
    """

    def test_on_data(self):
        nn = 3
        p = Problem()
        comp = p.model.add_subsystem("test", AmmoniaProperties(num_nodes=nn), promotes=["*"])
        comp.set_input_defaults("temp", units="degC", val=np.ones(nn) * 90.0)
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p["rho_liquid"], np.ones(nn) * 482.9)
        assert_near_equal(p["rho_vapor"], np.ones(nn) * 43.9)
        assert_near_equal(p["vapor_pressure"], np.ones(nn) * 5123.0)
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_interpolated(self):
        nn = 6
        p = Problem()
        comp = p.model.add_subsystem("test", AmmoniaProperties(num_nodes=nn), promotes=["*"])
        comp.set_input_defaults("temp", units="degC", val=np.linspace(-7.0, 78.0, nn))
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(
            p["rho_liquid"],
            np.array([648.00187402, 624.69, 599.75101299, 572.91683838, 543.40564864, 509.97440705]),
            tolerance=1e-5,
        )
        assert_near_equal(
            p["rho_vapor"],
            np.array([2.6756274, 4.8593, 8.26745558, 13.40464169, 21.03778235, 32.43929276]),
            tolerance=1e-5,
        )
        assert_near_equal(
            p["vapor_pressure"],
            np.array([327.98889865, 614.9, 1065.92300458, 1733.70063068, 2677.30942723, 3966.79472967]),
            tolerance=1e-5,
        )
        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class QMaxHeatPipeTestCase(unittest.TestCase):
    """
    Basic test for QMaxHeatPipe component to ensure no drastic changes in outputs
    """

    def test_default_settings(self):
        nn = 3
        p = Problem()
        comp = p.model.add_subsystem("test", QMaxHeatPipe(num_nodes=nn), promotes=["*"])
        comp.set_input_defaults("temp", units="degC", val=np.linspace(30, 60, nn))
        comp.set_input_defaults("length", units="m", val=10.22)
        comp.set_input_defaults("inner_diam", units="inch", val=0.902)
        comp.set_input_defaults("design_temp", units="degC", val=40)
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p["q_max"], np.array([4936.70020611, 5073.99208835, 5080.99742858]), tolerance=1e-5)
        assert_near_equal(p["heat_pipe_weight"], 0.51463886, tolerance=1e-5)

        partials = p.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)


class QMaxAnalyticalPartTestCase(unittest.TestCase):
    """
    Basic test for QMaxAnalyticalPart component to ensure no drastic changes in outputs
    """

    def test_default_settings(self):
        nn = 3
        p = Problem()
        p.model.add_subsystem("test", QMaxAnalyticalPart(num_nodes=nn), promotes=["*"])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p["q_max"], np.ones(nn) * 875.85211163, tolerance=1e-5)

        partials = p.check_partials(method="cs", compact_print=True, step=1e-50)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
