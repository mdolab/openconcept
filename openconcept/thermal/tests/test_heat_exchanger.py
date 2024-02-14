import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.thermal.heat_exchanger import HXGroup


class OSFGeometryTestGroup(Group):
    """
    Test the offset strip fin geometry component
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]

        iv = self.add_subsystem("iv", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("case_thickness", val=2.0, units="mm")
        iv.add_output("fin_thickness", val=0.102, units="mm")
        iv.add_output("plate_thickness", val=0.2, units="mm")
        iv.add_output("material_k", val=190, units="W/m/K")
        iv.add_output("material_rho", val=2700, units="kg/m**3")

        iv.add_output("mdot_cold", val=np.ones(nn) * 1.5, units="kg/s")
        iv.add_output("rho_cold", val=np.ones(nn) * 0.5, units="kg/m**3")

        iv.add_output("mdot_hot", val=0.075 * np.ones(nn), units="kg/s")
        iv.add_output("rho_hot", val=np.ones(nn) * 1020.2, units="kg/m**3")

        iv.add_output("T_in_cold", val=np.ones(nn) * 45, units="degC")
        iv.add_output("T_in_hot", val=np.ones(nn) * 90, units="degC")
        iv.add_output("ac|propulsion|thermal|hx|n_long_cold", val=3)
        iv.add_output("ac|propulsion|thermal|hx|n_wide_cold", val=430)
        iv.add_output("ac|propulsion|thermal|hx|n_tall", val=19)

        iv.add_output("channel_height_cold", val=14, units="mm")
        iv.add_output("channel_width_cold", val=1.35, units="mm")
        iv.add_output("fin_length_cold", val=6, units="mm")
        iv.add_output("cp_cold", val=1005, units="J/kg/K")
        iv.add_output("k_cold", val=0.02596, units="W/m/K")
        iv.add_output("mu_cold", val=1.789e-5, units="kg/m/s")

        iv.add_output("channel_height_hot", val=1, units="mm")
        iv.add_output("channel_width_hot", val=1, units="mm")
        iv.add_output("fin_length_hot", val=6, units="mm")
        iv.add_output("cp_hot", val=3801, units="J/kg/K")
        iv.add_output("k_hot", val=0.405, units="W/m/K")
        iv.add_output("mu_hot", val=1.68e-3, units="kg/m/s")

        self.add_subsystem("hx", HXGroup(), promotes=["*"])


class OSFGeometryTestCase(unittest.TestCase):
    def test_default_settings(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob["hx.osfgeometry.dh_cold"], 0.00242316, tolerance=1e-6)
        assert_near_equal(prob["hx.heat_transfer"], 10040.9846, tolerance=1e-6)
        assert_near_equal(prob["hx.delta_p_cold"], -135.15338626, tolerance=1e-6)
        assert_near_equal(prob["hx.delta_p_hot"], -9112.282754, tolerance=1e-6)
        assert_near_equal(prob["hx.component_weight"], 1.147605, tolerance=1e-5)

        partials = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=True, step=1e-50)
        assert_check_partials(partials)

    def test_kayslondon_10_61(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True, force_alloc_complex=True)
        prob.set_val("fin_thickness", 0.004, units="inch")
        prob.set_val("plate_thickness", 0.004, units="inch")
        prob.set_val("fin_length_cold", 1.0 / 10.0, units="inch")
        fin_spacing = 1 / 19.35 - 0.004  # fin pitch minus fin thickness
        prob.set_val("channel_height_cold", 0.0750 - 0.004, units="inch")
        prob.set_val("channel_width_cold", fin_spacing, units="inch")
        prob.set_val("n_long_cold", 2)
        prob.set_val("mdot_cold", 0.0905, units="kg/s")

        prob.run_model()
        prob.model.list_outputs(units=True)
        # test the geometry in Kays and London 3rd Ed Pg 248, Fig 10-61
        assert_near_equal(prob["hx.osfgeometry.dh_cold"], 1.403e-3, tolerance=1e-3)
        assert_near_equal(prob["hx.redh.Re_dh_cold"], 400.0, tolerance=1e-2)
        # data directly from Kays/London at Redh=400
        assert_near_equal(prob["hx.osfdata.j_cold"], 0.0195, tolerance=2e-1)
        assert_near_equal(prob["hx.osfdata.f_cold"], 0.0750, tolerance=2e-1)

        prob.set_val("mdot_cold", 0.0905 * 5, units="kg/s")
        prob.run_model()
        # data directly from Kays/London at Redh=2000
        assert_near_equal(prob["hx.redh.Re_dh_cold"], 2000.0, tolerance=1e-2)
        assert_near_equal(prob["hx.osfdata.j_cold"], 0.00940, tolerance=2e-1)
        assert_near_equal(prob["hx.osfdata.f_cold"], 0.0303, tolerance=3.5e-1)

        assert_near_equal(prob["hx.osfgeometry.alpha_cold"], 0.672, tolerance=1e-2)
        assert_near_equal(prob["hx.osfgeometry.delta_cold"], 0.040, tolerance=1e-2)
        assert_near_equal(prob["hx.osfgeometry.gamma_cold"], 0.084, tolerance=1e-2)

    def test_kayslondon_10_55(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True, force_alloc_complex=True)
        prob.set_val("fin_thickness", 0.004, units="inch")
        prob.set_val("plate_thickness", 0.004, units="inch")
        prob.set_val("fin_length_cold", 1.0 / 8.0, units="inch")
        fin_spacing = 1 / 15.61 - 0.004  # fin pitch minus fin thickness
        prob.set_val("channel_height_cold", 0.250 - 0.004, units="inch")
        prob.set_val("channel_width_cold", fin_spacing, units="inch")
        prob.set_val("n_long_cold", 2)
        prob.set_val("mdot_cold", 0.235, units="kg/s")

        prob.run_model()
        # test the geometry in Kays and London 3rd Ed Pg 248, Fig 10-55
        assert_near_equal(prob["hx.osfgeometry.dh_cold"], 2.383e-3, tolerance=1e-2)
        # data directly from Kays/London at Redh=400
        assert_near_equal(prob["hx.redh.Re_dh_cold"], 400.0, tolerance=1e-2)
        assert_near_equal(prob["hx.osfdata.j_cold"], 0.0246, tolerance=1e-1)
        assert_near_equal(prob["hx.osfdata.f_cold"], 0.104, tolerance=1e-1)
        prob.set_val("mdot_cold", 0.235 * 5, units="kg/s")
        prob.run_model()
        # data directly from Kays/London at Redh=2000
        assert_near_equal(prob["hx.redh.Re_dh_cold"], 2000.0, tolerance=1e-2)
        assert_near_equal(prob["hx.osfdata.j_cold"], 0.0111, tolerance=1e-1)
        assert_near_equal(prob["hx.osfdata.f_cold"], 0.0420, tolerance=1e-1)

        assert_near_equal(prob["hx.osfgeometry.alpha_cold"], 0.244, tolerance=1e-2)
        assert_near_equal(prob["hx.osfgeometry.delta_cold"], 0.032, tolerance=1e-2)
        assert_near_equal(prob["hx.osfgeometry.gamma_cold"], 0.067, tolerance=1e-2)

    def test_kayslondon_10_60(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True, force_alloc_complex=True)
        prob.set_val("fin_thickness", 0.004, units="inch")
        prob.set_val("plate_thickness", 0.004, units="inch")
        prob.set_val("fin_length_cold", 1.0 / 10.0, units="inch")
        fin_spacing = 1 / 27.03 - 0.004  # fin pitch minus fin thickness
        prob.set_val("channel_height_cold", 0.250 - 0.004, units="inch")
        prob.set_val("channel_width_cold", fin_spacing, units="inch")
        prob.set_val("n_long_cold", 2)
        prob.set_val("mdot_cold", 0.27, units="kg/s")

        prob.run_model()

        # test the geometry in Kays and London 3rd Ed Pg 248, Fig 10-55
        # assert_near_equal(prob['osfgeometry.dh_cold'], 0.00147796, tolerance=1e-4)
        assert_near_equal(prob["hx.osfgeometry.dh_cold"], 0.001423, tolerance=1e-2)
        # data directly from Kays/London at Redh=500
        assert_near_equal(prob["hx.redh.Re_dh_cold"], 500.0, tolerance=1e-2)
        assert_near_equal(prob["hx.osfdata.j_cold"], 0.0238, tolerance=1e-1)
        assert_near_equal(prob["hx.osfdata.f_cold"], 0.0922, tolerance=1e-1)
        prob.set_val("mdot_cold", 0.27 * 4, units="kg/s")
        prob.run_model()
        # data directly from Kays/London at Redh=2000
        assert_near_equal(prob["hx.redh.Re_dh_cold"], 2000.0, tolerance=1e-2)
        assert_near_equal(prob["hx.osfdata.j_cold"], 0.0113, tolerance=1e-1)
        assert_near_equal(prob["hx.osfdata.f_cold"], 0.0449, tolerance=1e-1)

        assert_near_equal(prob["hx.osfgeometry.alpha_cold"], 0.134, tolerance=1e-2)
        assert_near_equal(prob["hx.osfgeometry.delta_cold"], 0.040, tolerance=1e-2)
        assert_near_equal(prob["hx.osfgeometry.gamma_cold"], 0.121, tolerance=1e-2)

    def test_kayslondon_10_63(self):
        prob = Problem(OSFGeometryTestGroup(num_nodes=1))
        prob.setup(check=True, force_alloc_complex=True)
        prob.set_val("fin_thickness", 0.004, units="inch")
        prob.set_val("plate_thickness", 0.004, units="inch")
        prob.set_val("fin_length_cold", 3.0 / 32.0, units="inch")
        fin_spacing = 0.082 - 0.004  # fin pitch minus fin thickness
        prob.set_val("channel_height_cold", 0.485 - 0.004, units="inch")
        prob.set_val("channel_width_cold", fin_spacing, units="inch")
        prob.set_val("n_long_cold", 4)
        prob.set_val("mdot_cold", 0.54, units="kg/s")

        prob.run_model()
        # test the geometry in Kays and London 3rd Ed Pg 248, Fig 10-55
        # assert_near_equal(prob['osfgeometry.dh_cold'], 0.00341, tolerance=1e-2)
        # data directly from Kays/London at Redh=500
        assert_near_equal(prob["hx.redh.Re_dh_cold"], 500.0, tolerance=1e-2)
        assert_near_equal(prob["hx.osfdata.j_cold"], 0.0205, tolerance=2e-1)
        assert_near_equal(prob["hx.osfdata.f_cold"], 0.130, tolerance=2e-1)
        prob.set_val("mdot_cold", 0.54 * 4, units="kg/s")
        prob.run_model()
        # data directly from Kays/London at Redh=2000
        assert_near_equal(prob["hx.redh.Re_dh_cold"], 2000.0, tolerance=1e-2)
        assert_near_equal(prob["hx.osfdata.j_cold"], 0.0119, tolerance=2e-1)
        assert_near_equal(prob["hx.osfdata.f_cold"], 0.0607, tolerance=2e-1)

        assert_near_equal(prob["hx.osfgeometry.alpha_cold"], 0.162, tolerance=1e-2)
        assert_near_equal(prob["hx.osfgeometry.delta_cold"], 0.043, tolerance=1e-2)
        assert_near_equal(prob["hx.osfgeometry.gamma_cold"], 0.051, tolerance=1e-2)


class OSFManualCheckTestGroup(Group):
    """
    Test the offset strip fin geometry component
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]

        iv = self.add_subsystem("iv", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("case_thickness", val=2.0, units="mm")
        iv.add_output("fin_thickness", val=0.1, units="mm")
        iv.add_output("plate_thickness", val=0.2, units="mm")
        iv.add_output("material_k", val=190, units="W/m/K")
        iv.add_output("material_rho", val=2700, units="kg/m**3")

        iv.add_output("mdot_cold", val=np.ones(nn) * 0.1, units="kg/s")
        iv.add_output("rho_cold", val=np.ones(nn) * 0.5, units="kg/m**3")

        iv.add_output("mdot_hot", val=np.ones(nn) * 0.2, units="kg/s")
        iv.add_output("rho_hot", val=np.ones(nn) * 0.6, units="kg/m**3")

        iv.add_output("T_in_cold", val=np.ones(nn) * 45, units="degC")
        iv.add_output("T_in_hot", val=np.ones(nn) * 90, units="degC")
        iv.add_output("ac|propulsion|thermal|hx|n_long_cold", val=25)
        iv.add_output("ac|propulsion|thermal|hx|n_wide_cold", val=25)
        iv.add_output("ac|propulsion|thermal|hx|n_tall", val=8)

        iv.add_output("channel_height_cold", val=6.0, units="mm")
        iv.add_output("channel_width_cold", val=1.5, units="mm")
        iv.add_output("fin_length_cold", val=3, units="mm")
        iv.add_output("cp_cold", val=1005, units="J/kg/K")
        iv.add_output("k_cold", val=0.02596, units="W/m/K")
        iv.add_output("mu_cold", val=1.789e-5, units="kg/m/s")

        iv.add_output("channel_height_hot", val=8.0, units="mm")
        iv.add_output("channel_width_hot", val=1.7, units="mm")
        iv.add_output("fin_length_hot", val=3.1, units="mm")
        iv.add_output("cp_hot", val=900, units="J/kg/K")
        iv.add_output("k_hot", val=0.024, units="W/m/K")
        iv.add_output("mu_hot", val=1.7e-5, units="kg/m/s")

        self.add_subsystem("hx", HXGroup(), promotes=["*"])


class TestHXByHand(unittest.TestCase):
    def test_by_hand(self):
        """
        This test case verifies that the implementation
        of the equations from Kays and London are correct
        and not mistyped.
        """
        prob = Problem(OSFManualCheckTestGroup(num_nodes=1))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()

        h_c = 6.0
        h_h = 8.0
        w_c = 1.5
        w_h = 1.7
        l_f_c = 3.0
        l_f_h = 3.1
        t_f = 0.1
        t_p = 0.2
        n_cold_wide = 25
        n_cold_tall = 8
        n_cold_long = 25
        cold_one_layer_height = t_p + t_f + h_c
        cold_one_cell_width = t_f + w_c
        cold_length = l_f_c * n_cold_long
        hot_one_layer_height = t_p + t_f + h_h
        hot_one_cell_width = t_f + w_h
        n_hot_wide = cold_length / hot_one_cell_width

        # ######____
        # |    |    |
        # |    |    |
        # |    |    |
        # |____|    |___
        # #########

        height_overall = n_cold_tall * (cold_one_layer_height + hot_one_layer_height)
        # note does not include case
        assert_near_equal(prob["hx.osfgeometry.height_overall"], height_overall / 1000, tolerance=1e-6)
        width_overall = n_cold_wide * cold_one_cell_width
        assert_near_equal(prob["hx.osfgeometry.width_overall"], width_overall / 1000, tolerance=1e-6)
        length_overall = n_cold_long * l_f_c
        assert_near_equal(prob["hx.osfgeometry.length_overall"], length_overall / 1000, tolerance=1e-6)
        frontal_area = width_overall * height_overall
        assert_near_equal(prob["hx.osfgeometry.frontal_area"], frontal_area / 1000**2, tolerance=1e-6)

        xs_area_cold = n_cold_wide * n_cold_tall * w_c * h_c
        assert_near_equal(prob["hx.osfgeometry.xs_area_cold"], xs_area_cold / 1000**2, tolerance=1e-6)
        heat_transfer_area_cold = 2 * (w_c + h_c) * n_cold_tall * n_cold_wide * length_overall
        assert_near_equal(
            prob["hx.osfgeometry.heat_transfer_area_cold"], heat_transfer_area_cold / 1000**2, tolerance=1e-6
        )
        dh_cold = 4 * w_c * h_c * l_f_c / (2 * (w_c * l_f_c + h_c * l_f_c + t_f * h_c) + t_f * w_c)
        assert_near_equal(prob["hx.osfgeometry.dh_cold"], dh_cold / 1000, tolerance=1e-6)
        assert_near_equal(prob["hx.osfgeometry.dh_cold"], 2 * h_c * w_c / (h_c + w_c) / 1000, tolerance=3e-2)

        fin_area_ratio_cold = h_c / (h_c + w_c)
        assert_near_equal(prob["hx.osfgeometry.fin_area_ratio_cold"], fin_area_ratio_cold, tolerance=1e-6)
        contraction_ratio_cold = xs_area_cold / frontal_area
        assert_near_equal(prob["hx.osfgeometry.contraction_ratio_cold"], contraction_ratio_cold, tolerance=1e-6)
        alpha_cold = w_c / h_c
        delta_cold = t_f / l_f_c
        gamma_cold = t_f / w_c
        assert_near_equal(prob["hx.osfgeometry.alpha_cold"], alpha_cold, tolerance=1e-6)
        assert_near_equal(prob["hx.osfgeometry.delta_cold"], delta_cold, tolerance=1e-6)
        assert_near_equal(prob["hx.osfgeometry.gamma_cold"], gamma_cold, tolerance=1e-6)

        n_hot_wide = length_overall / hot_one_cell_width
        xs_area_hot = n_hot_wide * n_cold_tall * w_h * h_h
        assert_near_equal(prob["hx.osfgeometry.xs_area_hot"], xs_area_hot / 1000**2, tolerance=1e-6)
        heat_transfer_area_hot = 2 * (w_h + h_h) * n_cold_tall * n_hot_wide * width_overall
        assert_near_equal(
            prob["hx.osfgeometry.heat_transfer_area_hot"], heat_transfer_area_hot / 1000**2, tolerance=1e-6
        )
        dh_hot = 2 * w_h * h_h / (w_h + h_h)
        assert_near_equal(prob["hx.osfgeometry.dh_hot"], dh_hot / 1000, tolerance=1e-6)
        fin_area_ratio_hot = h_h / (h_h + w_h)
        assert_near_equal(prob["hx.osfgeometry.fin_area_ratio_hot"], fin_area_ratio_hot, tolerance=1e-6)
        contraction_ratio_hot = xs_area_hot / length_overall / height_overall
        assert_near_equal(prob["hx.osfgeometry.contraction_ratio_hot"], contraction_ratio_hot, tolerance=1e-6)
        alpha_hot = w_h / h_h
        delta_hot = t_f / l_f_h
        gamma_hot = t_f / w_h
        assert_near_equal(prob["hx.osfgeometry.alpha_hot"], alpha_hot, tolerance=1e-6)
        assert_near_equal(prob["hx.osfgeometry.delta_hot"], delta_hot, tolerance=1e-6)
        assert_near_equal(prob["hx.osfgeometry.gamma_hot"], gamma_hot, tolerance=1e-6)

        mdot_cold = 0.1
        mdot_hot = 0.2
        mu_cold = 1.789e-5
        mu_hot = 1.7e-5

        redh_cold = mdot_cold / (xs_area_cold / 1000**2) * dh_cold / 1000 / mu_cold
        redh_hot = mdot_hot / (xs_area_hot / 1000**2) * dh_hot / 1000 / mu_hot
        assert_near_equal(prob["hx.redh.Re_dh_cold"], redh_cold, tolerance=1e-6)
        assert_near_equal(prob["hx.redh.Re_dh_hot"], redh_hot, tolerance=1e-6)

        partials = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=True, step=1e-50)
        assert_check_partials(partials)

        j_cold = (
            0.6522
            * redh_cold**-0.5403
            * alpha_cold**-0.1541
            * delta_cold**0.1499
            * gamma_cold**-0.0678
            * (1 + 5.269e-5 * redh_cold**1.340 * alpha_cold**0.504 * delta_cold**0.456 * gamma_cold**-1.055)
            ** 0.1
        )
        j_hot = (
            0.6522
            * redh_hot**-0.5403
            * alpha_hot**-0.1541
            * delta_hot**0.1499
            * gamma_hot**-0.0678
            * (1 + 5.269e-5 * redh_hot**1.340 * alpha_hot**0.504 * delta_hot**0.456 * gamma_hot**-1.055) ** 0.1
        )
        assert_near_equal(prob["hx.osfdata.j_hot"], j_hot, tolerance=1e-6)
        assert_near_equal(prob["hx.osfdata.j_cold"], j_cold, tolerance=1e-6)
        f_cold = (
            9.6243
            * redh_cold**-0.7422
            * alpha_cold**-0.1856
            * delta_cold**0.3053
            * gamma_cold**-0.2659
            * (1 + 7.669e-8 * redh_cold**4.429 * alpha_cold**0.920 * delta_cold**3.767 * gamma_cold**0.236)
            ** 0.1
        )
        f_hot = (
            9.6243
            * redh_hot**-0.7422
            * alpha_hot**-0.1856
            * delta_hot**0.3053
            * gamma_hot**-0.2659
            * (1 + 7.669e-8 * redh_hot**4.429 * alpha_hot**0.920 * delta_hot**3.767 * gamma_hot**0.236) ** 0.1
        )
        assert_near_equal(prob["hx.osfdata.f_hot"], f_hot, tolerance=1e-6)
        assert_near_equal(prob["hx.osfdata.f_cold"], f_cold, tolerance=1e-6)

        cp_cold = 1005
        k_cold = 0.02596
        cp_hot = 900
        k_hot = 0.024

        h_cold = j_cold * cp_cold ** (1 / 3) * (k_cold / mu_cold) ** (2 / 3) * mdot_cold / xs_area_cold * 1000**2
        h_hot = j_hot * cp_hot ** (1 / 3) * (k_hot / mu_hot) ** (2 / 3) * mdot_hot / xs_area_hot * 1000**2

        assert_near_equal(prob["hx.convection.h_conv_cold"], h_cold, tolerance=1e-6)
        assert_near_equal(prob["hx.convection.h_conv_hot"], h_hot, tolerance=1e-6)

        k_alu = 190
        # TODO kays and london has a different expression for fin efficiency. why
        # m = np.sqrt(2*h_cold/k_alu/(t_f/1000))*(1 + t_f / l_f_c)
        m = np.sqrt(2 * h_cold / k_alu / (t_f / 1000))
        eta_f_cold = np.tanh(m * h_c / 2 / 1000) / m / (h_c / 2 / 1000)
        eta_o_cold = 1 - (1 - eta_f_cold) * fin_area_ratio_cold
        assert_near_equal(prob["hx.finefficiency.eta_overall_cold"], eta_o_cold, tolerance=1e-6)
        m = np.sqrt(2 * h_hot / k_alu / (t_f / 1000))
        eta_f_hot = np.tanh(m * h_h / 2 / 1000) / m / (h_h / 2 / 1000)
        eta_o_hot = 1 - (1 - eta_f_hot) * fin_area_ratio_hot
        assert_near_equal(prob["hx.finefficiency.eta_overall_hot"], eta_o_hot, tolerance=1e-6)

        rc = 1 / eta_o_cold / (heat_transfer_area_cold / 1000**2) / h_cold
        rh = 1 / eta_o_hot / (heat_transfer_area_hot / 1000**2) / h_hot
        # rw = (t_f + t_p)/1000/k_alu/(length_overall * width_overall / 1000 ** 2)/n_cold_tall
        # TODO wall resistance not currently accounted for, less than 1% effect
        rw = 0.0
        uaoverall = 1 / (rc + rh + rw)
        assert_near_equal(prob["hx.ua.UA_overall"], uaoverall, tolerance=1e-6)

        cmin = np.minimum(mdot_cold * cp_cold, mdot_hot * cp_hot)
        cmax = np.maximum(mdot_cold * cp_cold, mdot_hot * cp_hot)

        cratio = cmin / cmax
        ntu = uaoverall / cmin
        effectiveness = 1 - np.exp(ntu**0.22 * (np.exp(-cratio * ntu**0.78) - 1) / cratio)
        assert_near_equal(prob["hx.ntu.NTU"], ntu, tolerance=1e-6)
        assert_near_equal(prob["hx.effectiveness.effectiveness"], effectiveness, tolerance=1e-6)

        heat_transfer = cmin * effectiveness * (90 - 45)
        assert_near_equal(prob["hx.heat.heat_transfer"], heat_transfer, tolerance=1e-6)

        tout_cold = 45 + heat_transfer / mdot_cold / cp_cold + 273.15
        tout_hot = 90 - heat_transfer / mdot_hot / cp_hot + 273.15
        assert_near_equal(prob["hx.t_out.T_out_cold"], tout_cold, tolerance=1e-6)
        assert_near_equal(prob["hx.t_out.T_out_hot"], tout_hot, tolerance=1e-6)

        Gcold = mdot_cold / (xs_area_cold / 1000**2)
        Ghot = mdot_hot / (xs_area_hot / 1000**2)

        rho_cold = 0.5
        rho_hot = 0.6
        Kc = 0.3
        Ke = -0.1
        pressure_drop_cold = -(Gcold**2) / 2 / rho_cold * ((Kc + Ke) + f_cold * 4 * length_overall / dh_cold)
        pressure_drop_hot = -(Ghot**2) / 2 / rho_hot * ((Kc + Ke) + f_hot * 4 * width_overall / dh_hot)
        assert_near_equal(prob["hx.delta_p.delta_p_cold"], pressure_drop_cold, tolerance=1e-6)
        assert_near_equal(prob["hx.delta_p.delta_p_hot"], pressure_drop_hot, tolerance=1e-6)


if __name__ == "__main__":
    unittest.main()
