import numpy as np
from openmdao.api import ExplicitComponent, Group
from openconcept.utilities import DVLabel


class OffsetStripFinGeometry(ExplicitComponent):
    """
    Computes geometric and solid parameters of a offset strip fin plate-fin heat exchanger.

    Geometric parameters as published by Manglik and Bergles,
    'Heat Transfer and Pressure Drop Correlations for the Rectangular Offset Strip Fin
    Compact Heat Exchanger', Experimental Thermal and Fluid Science, 1995
    DOI https://doi.org/10.1016/0894-1777(94)00096-Q

    Inputs
    ------
    channel_width_hot : float
        Width of each finned flow channel on the hot side (scalar, m)
    channel_height_hot : float
        Height of each finned flow channel on the hot side (scalar, m)
    fin_length_hot : float
        Length of each offset strip fin on the hot side (scalar, m)
    channel_width_cold : float
        Width of each finned flow channel on the cold side (scalar, m)
    channel_height_cold : float
        Height of each finned flow channel on the cold side (scalar, m)
    fin_length_cold : float
        Length of each offset strip fin on the cold side (scalar, m)
    fin_thickness : float
        Thickness of fin material (scalar, m)
    plate_thickness : float
        Thickness of plate divider material (scalar, m)
    case_thickness : float
        Thickness of the outer case material (scalar, m)
    n_wide_cold : float
        Number of channels wide (cold side) (scalar, dimensionless)
    n_long_cold : float
        Number of fins long (cold side) (scalar, dimensionless)
    n_tall : float
        Number of times to stack the hot/cold combo (scalar, dimensionless)
    material_rho : float
        Density of the heat exchanger material (scalar, kg/m**3)

    Outputs
    -------
    component_weight : float
        Weight / mass of the heat exchanger material (scalar, kg)
    length_overall : float
        Overall heat exchanger length as viewed from cold side (scalar, m)
    height_overall : float
        Overall heat exhcanger height (scalar, m)
    width_overall : float
        Overall heat exchanger width as viewed from cold side (scalar, m)
    frontal_area : float
        Frontal area of the heat exchanger (cold side) (scalar, m**2)
    xs_area_cold : float
        Cross-sectional flow area of the cold side (scalar, m**2)
    heat_transfer_area_cold : float
        Total heat transfer surface area of the cold side (scalar, m**2)
    dh_cold : float
        Hydraulic diameter of the cold side flow channels (scalar, m)
    fin_area_ratio_cold : float
        Ratio of fin area to total heat transfer area (scalar, dimensionless)
    contraction_ratio_cold : float
        Ratio of flow xs area to total xs area of the cold side (scalar, dimensionless)
    alpha_cold : float
        Ratio of fin channel width to height on the cold side (scalar, dimensionless)
    delta_cold : float
        Ratio of fin thickness to length on the cold side (scalar, dimensionless)
    gamma_cold : float
        Ratio of fin thickness to flow width on the cold side (scalar, dimensionless)
    xs_area_hot : float
        Cross-sectional flow area of the hot side (scalar, m**2)
    heat_transfer_area_hot : float
        Total heat transfer surface area of the hot side (scalar, m**2)
    dh_hot : float
        Hydraulic diameter of the hot side flow channels (scalar, m)
    fin_area_ratio_hot : float
        Ratio of fin area to total heat transfer area (scalar, dimensionless)
    contraction_ratio_hot : float
        Ratio of flow xs area to total xs area of the hot side (scalar, dimensionless)
    alpha_hot : float
        Ratio of fin channel width to height on the hot side (scalar, dimensionless)
    delta_hot : float
        Ratio of fin thickness to length on the hot side (scalar, dimensionless)
    gamma_hot : float
        Ratio of fin thickness to flow width on the hot side (scalar, dimensionless)
    """

    # def initialize(self):
    #     self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        self.add_input("channel_width_hot", val=0.001, units="m")
        self.add_input("channel_height_hot", val=0.001, units="m")
        self.add_input("fin_length_hot", val=0.006, units="m")
        self.add_input("channel_width_cold", val=0.001, units="m")
        self.add_input("channel_height_cold", val=0.012, units="m")
        self.add_input("fin_length_cold", val=0.006, units="m")

        self.add_input("fin_thickness", val=0.000102, units="m")
        self.add_input("plate_thickness", val=0.0002, units="m")
        self.add_input("case_thickness", val=0.002, units="m")

        self.add_input("n_wide_cold", val=100)
        self.add_input("n_long_cold", val=3)
        self.add_input("n_tall", val=20)
        self.add_input("material_rho", val=2700.0, units="kg/m**3")

        self.add_output("component_weight", units="kg")
        self.add_output("length_overall", units="m")
        self.add_output("height_overall", units="m")
        self.add_output("width_overall", units="m")
        self.add_output("frontal_area", units="m**2")

        self.add_output("xs_area_cold", units="m**2")
        self.add_output("heat_transfer_area_cold", units="m**2")
        self.add_output("dh_cold", units="m")
        self.add_output("fin_area_ratio_cold")
        self.add_output("contraction_ratio_cold")
        self.add_output("alpha_cold")
        self.add_output("delta_cold")
        self.add_output("gamma_cold")

        self.add_output("xs_area_hot", units="m**2")
        self.add_output("heat_transfer_area_hot", units="m**2")
        self.add_output("dh_hot", units="m")
        self.add_output("fin_area_ratio_hot")
        self.add_output("contraction_ratio_hot")
        self.add_output("alpha_hot")
        self.add_output("delta_hot")
        self.add_output("gamma_hot")

        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        t_f = inputs["fin_thickness"]
        t_p = inputs["plate_thickness"]
        t_c = inputs["case_thickness"]
        n_tall = inputs["n_tall"]

        w_c = inputs["channel_width_cold"]
        h_c = inputs["channel_height_cold"]
        l_c = inputs["fin_length_cold"]
        n_wide_c = inputs["n_wide_cold"]
        n_long_c = inputs["n_long_cold"]

        w_h = inputs["channel_width_hot"]
        h_h = inputs["channel_height_hot"]
        l_h = inputs["fin_length_hot"]

        # compute overall properties
        outputs["height_overall"] = (2 * (t_f + t_p) + h_c + h_h) * n_tall
        outputs["width_overall"] = (t_f + w_c) * n_wide_c
        outputs["length_overall"] = l_c * n_long_c
        outputs["frontal_area"] = outputs["width_overall"] * outputs["height_overall"]

        # compute cold side geometric properties
        # outputs['dh_cold'] = 2 * w_c * h_c / (w_c + h_c)
        # special formula for dh (maybe accounts for bend radii?) from Manglik and Bergles paper
        outputs["dh_cold"] = 4 * w_c * h_c * l_c / (2 * (w_c * l_c + h_c * l_c + t_f * h_c) + t_f * w_c)
        outputs["xs_area_cold"] = w_c * h_c * n_wide_c * n_tall
        outputs["heat_transfer_area_cold"] = 2 * (w_c + h_c) * l_c * n_long_c * n_wide_c * n_tall
        outputs["fin_area_ratio_cold"] = h_c / (h_c + w_c)
        outputs["contraction_ratio_cold"] = outputs["xs_area_cold"] / outputs["frontal_area"]
        outputs["alpha_cold"] = w_c / h_c
        outputs["delta_cold"] = t_f / l_c
        outputs["gamma_cold"] = t_f / w_c

        n_wide_h = outputs["length_overall"] / (w_h + t_f)
        n_long_h = outputs["width_overall"] / l_h

        outputs["dh_hot"] = 2 * w_h * h_h / (w_h + h_h)
        outputs["xs_area_hot"] = w_h * h_h * n_wide_h * n_tall
        outputs["heat_transfer_area_hot"] = 2 * (w_h + h_h) * l_h * n_long_h * n_wide_h * n_tall
        outputs["fin_area_ratio_hot"] = h_h / (h_h + w_h)
        outputs["contraction_ratio_hot"] = (
            outputs["xs_area_hot"] / outputs["height_overall"] / outputs["length_overall"]
        )
        outputs["alpha_hot"] = w_h / h_h
        outputs["delta_hot"] = t_f / l_h
        outputs["gamma_hot"] = t_f / w_h

        plate_volume = outputs["length_overall"] * outputs["width_overall"] * 2 * t_p * n_tall
        fin_volume = (w_c + h_c + t_f) * l_c * n_long_c * t_f * n_wide_c * n_tall + (
            w_h + h_h + t_f
        ) * l_h * n_long_h * t_f * n_wide_h * n_tall
        case_volume = t_c * outputs["length_overall"] * 2 * (outputs["height_overall"] + outputs["width_overall"])
        outputs["component_weight"] = (plate_volume + fin_volume + case_volume) * inputs["material_rho"]


class OffsetStripFinData(ExplicitComponent):
    """
    Computes Fanning friction factor f and Coburn j factor for offset strip fin geometry
    Correlations from empirical data published by Manglik and Bergles,
    'Heat Transfer and Pressure Drop Correlations for the Rectangular Offset Strip Fin
    Compact Heat Exchanger', Experimental Thermal and Fluid Science, 1995
    DOI https://doi.org/10.1016/0894-1777(94)00096-Q
    Equations 34 and 35

    Inputs
    ------
    Re_dh_cold : float
        Hydraulic diameter reynolds number of the cold side (vector, dimensionless)
    alpha_cold : float
        Ratio of fin channel width to height on the cold side (scalar, dimensionless)
    delta_cold : float
        Ratio of fin thickness to length on the cold side (scalar, dimensionless)
    gamma_cold : float
        Ratio of fin thickness to flow width on the cold side (scalar, dimensionless)
    Re_dh_hot : float
        Hydraulic diameter reynolds number of the hot side (vector, dimensionless)
    alpha_hot : float
        Ratio of fin channel width to height on the hot side (scalar, dimensionless)
    delta_hot : float
        Ratio of fin thickness to length on the hot side (scalar, dimensionless)
    gamma_hot : float
        Ratio of fin thickness to flow width on the hot side (scalar, dimensionless)

    Outputs
    -------
    j_cold : float
        Colburn j factor for cold side (vector, dimensionless)
    f_cold : float
        Fanning friction factor for cold side (vector, dimensionless)
    j_hot : float
        Colburn j factor for hot side (vector, dimensionless)
    f_hot : float
        Fanning friction factor for hot side (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("Re_dh_cold", val=np.ones((nn,)), shape=(nn,))
        self.add_input("alpha_cold")
        self.add_input("delta_cold")
        self.add_input("gamma_cold")
        self.add_input("Re_dh_hot", val=np.ones((nn,)), shape=(nn,))
        self.add_input("alpha_hot")
        self.add_input("delta_hot")
        self.add_input("gamma_hot")
        self.add_output("j_cold", shape=(nn,))
        self.add_output("f_cold", shape=(nn,))
        self.add_output("j_hot", shape=(nn,))
        self.add_output("f_hot", shape=(nn,))
        arange = np.arange(0, nn)

        self.declare_partials(["j_cold", "f_cold"], ["alpha_cold", "delta_cold", "gamma_cold"], method="cs")
        self.declare_partials(["j_hot", "f_hot"], ["alpha_hot", "delta_hot", "gamma_hot"], method="cs")
        self.declare_partials(["j_cold", "f_cold"], ["Re_dh_cold"], rows=arange, cols=arange)
        self.declare_partials(["j_hot", "f_hot"], ["Re_dh_hot"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        jc_1 = inputs["alpha_cold"] ** -0.1541 * inputs["delta_cold"] ** 0.1499 * inputs["gamma_cold"] ** -0.0678
        jc_2 = inputs["alpha_cold"] ** 0.504 * inputs["delta_cold"] ** 0.456 * inputs["gamma_cold"] ** -1.055
        fc_1 = inputs["alpha_cold"] ** -0.1856 * inputs["delta_cold"] ** 0.3053 * inputs["gamma_cold"] ** -0.2659
        fc_2 = inputs["alpha_cold"] ** 0.92 * inputs["delta_cold"] ** 3.767 * inputs["gamma_cold"] ** 0.236
        if np.min(inputs["Re_dh_cold"]) <= 0.0:
            raise ValueError(self.msginfo, inputs["Re_dh_cold"])
        outputs["j_cold"] = (
            0.6522
            * inputs["Re_dh_cold"] ** -0.5403
            * jc_1
            * (1 + 5.269e-5 * inputs["Re_dh_cold"] ** 1.34 * jc_2) ** 0.1
        )
        outputs["f_cold"] = (
            9.6243
            * inputs["Re_dh_cold"] ** -0.7422
            * fc_1
            * (1 + 7.669e-8 * inputs["Re_dh_cold"] ** 4.429 * fc_2) ** 0.1
        )

        jh_1 = inputs["alpha_hot"] ** -0.1541 * inputs["delta_hot"] ** 0.1499 * inputs["gamma_hot"] ** -0.0678
        jh_2 = inputs["alpha_hot"] ** 0.504 * inputs["delta_hot"] ** 0.456 * inputs["gamma_hot"] ** -1.055
        fh_1 = inputs["alpha_hot"] ** -0.1856 * inputs["delta_hot"] ** 0.3053 * inputs["gamma_hot"] ** -0.2659
        fh_2 = inputs["alpha_hot"] ** 0.92 * inputs["delta_hot"] ** 3.767 * inputs["gamma_hot"] ** 0.236

        outputs["j_hot"] = (
            0.6522 * inputs["Re_dh_hot"] ** -0.5403 * jh_1 * (1 + 5.269e-5 * inputs["Re_dh_hot"] ** 1.34 * jh_2) ** 0.1
        )
        outputs["f_hot"] = (
            9.6243 * inputs["Re_dh_hot"] ** -0.7422 * fh_1 * (1 + 7.669e-8 * inputs["Re_dh_hot"] ** 4.429 * fh_2) ** 0.1
        )

    def compute_partials(self, inputs, J):
        jc_1 = inputs["alpha_cold"] ** -0.1541 * inputs["delta_cold"] ** 0.1499 * inputs["gamma_cold"] ** -0.0678
        jc_2 = inputs["alpha_cold"] ** 0.504 * inputs["delta_cold"] ** 0.456 * inputs["gamma_cold"] ** -1.055
        fc_1 = inputs["alpha_cold"] ** -0.1856 * inputs["delta_cold"] ** 0.3053 * inputs["gamma_cold"] ** -0.2659
        fc_2 = inputs["alpha_cold"] ** 0.92 * inputs["delta_cold"] ** 3.767 * inputs["gamma_cold"] ** 0.236

        J["j_cold", "Re_dh_cold"] = (
            0.6522
            * -0.5403
            * inputs["Re_dh_cold"] ** -1.5403
            * jc_1
            * (1 + 5.269e-5 * inputs["Re_dh_cold"] ** 1.34 * jc_2) ** 0.1
        ) + (
            0.6522
            * inputs["Re_dh_cold"] ** -0.5403
            * jc_1
            * 0.1
            * (1 + 5.269e-5 * inputs["Re_dh_cold"] ** 1.34 * jc_2) ** -0.9
            * 5.269e-5
            * 1.34
            * inputs["Re_dh_cold"] ** 0.34
            * jc_2
        )
        J["f_cold", "Re_dh_cold"] = (
            9.6243
            * -0.7422
            * inputs["Re_dh_cold"] ** -1.7422
            * fc_1
            * (1 + 7.669e-8 * inputs["Re_dh_cold"] ** 4.429 * fc_2) ** 0.1
        ) + (
            9.6243
            * inputs["Re_dh_cold"] ** -0.7422
            * fc_1
            * 0.1
            * (1 + 7.669e-8 * inputs["Re_dh_cold"] ** 4.429 * fc_2) ** -0.9
        ) * 7.669e-8 * 4.429 * inputs[
            "Re_dh_cold"
        ] ** 3.429 * fc_2

        jh_1 = inputs["alpha_hot"] ** -0.1541 * inputs["delta_hot"] ** 0.1499 * inputs["gamma_hot"] ** -0.0678
        jh_2 = inputs["alpha_hot"] ** 0.504 * inputs["delta_hot"] ** 0.456 * inputs["gamma_hot"] ** -1.055
        fh_1 = inputs["alpha_hot"] ** -0.1856 * inputs["delta_hot"] ** 0.3053 * inputs["gamma_hot"] ** -0.2659
        fh_2 = inputs["alpha_hot"] ** 0.92 * inputs["delta_hot"] ** 3.767 * inputs["gamma_hot"] ** 0.236

        J["j_hot", "Re_dh_hot"] = (
            0.6522
            * -0.5403
            * inputs["Re_dh_hot"] ** -1.5403
            * jh_1
            * (1 + 5.269e-5 * inputs["Re_dh_hot"] ** 1.34 * jh_2) ** 0.1
        ) + (
            0.6522
            * inputs["Re_dh_hot"] ** -0.5403
            * jh_1
            * 0.1
            * (1 + 5.269e-5 * inputs["Re_dh_hot"] ** 1.34 * jh_2) ** -0.9
            * 5.269e-5
            * 1.34
            * inputs["Re_dh_hot"] ** 0.34
            * jh_2
        )
        J["f_hot", "Re_dh_hot"] = (
            9.6243
            * -0.7422
            * inputs["Re_dh_hot"] ** -1.7422
            * fh_1
            * (1 + 7.669e-8 * inputs["Re_dh_hot"] ** 4.429 * fh_2) ** 0.1
        ) + (
            9.6243
            * inputs["Re_dh_hot"] ** -0.7422
            * fh_1
            * 0.1
            * (1 + 7.669e-8 * inputs["Re_dh_hot"] ** 4.429 * fh_2) ** -0.9
        ) * 7.669e-8 * 4.429 * inputs[
            "Re_dh_hot"
        ] ** 3.429 * fh_2


class HydraulicDiameterReynoldsNumber(ExplicitComponent):
    """
    Computes Re_dh for both sides of a heat exchanger

    Inputs
    ------
    mdot_cold : float
        Mass flow rate of the cold side (vector, kg/s)
    mu_cold : float
        Dynamic viscosity of the cold side fluid (scalar, kg/m/s)
    xs_area_cold : float
        Cross-sectional flow area of the cold side (scalar, m**2)
    dh_cold : float
        Hydraulic diameter of the cold side flow channels (scalar, m)
    mdot_hot : float
        Mass flow rate of the hot side (vector, kg/s)
    mu_hot : float
        Dynamic viscosity of the hot side fluid (scalar, kg/m/s)
    xs_area_hot : float
        Cross-sectional flow area of the hot side (scalar, m**2)
    dh_hot : float
        Hydraulic diameter of the hot side flow channels (scalar, m)

    Outputs
    -------
    Re_dh_cold : float
        Reynolds number based on the hydraulic diameter, cold side (vector, dimensionless)
    Re_dh_hot : float
        Reynolds number based on the hydraulic diameter, hot side (vector, dimensionless)


    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("mdot_cold", val=np.ones((nn,)), shape=(nn,), units="kg/s")
        self.add_input("mu_cold", units="kg/m/s")
        self.add_input("xs_area_cold", units="m**2")
        self.add_input("dh_cold", units="m")
        self.add_input("mdot_hot", val=np.ones((nn,)), shape=(nn,), units="kg/s")
        self.add_input("mu_hot", units="kg/m/s")
        self.add_input("xs_area_hot", units="m**2")
        self.add_input("dh_hot", units="m")

        self.add_output("Re_dh_cold", shape=(nn,), lower=0.01)
        self.add_output("Re_dh_hot", shape=(nn,))
        arange = np.arange(0, nn)
        self.declare_partials(["Re_dh_cold"], ["mdot_cold"], rows=arange, cols=arange)
        self.declare_partials(
            ["Re_dh_cold"], ["mu_cold", "xs_area_cold", "dh_cold"], rows=arange, cols=np.zeros((nn,), dtype=np.int32)
        )
        self.declare_partials(["Re_dh_hot"], ["mdot_hot"], rows=arange, cols=arange)
        self.declare_partials(
            ["Re_dh_hot"], ["mu_hot", "xs_area_hot", "dh_hot"], rows=arange, cols=np.zeros((nn,), dtype=np.int32)
        )

    def compute(self, inputs, outputs):
        outputs["Re_dh_cold"] = inputs["mdot_cold"] * inputs["dh_cold"] / inputs["xs_area_cold"] / inputs["mu_cold"]
        outputs["Re_dh_hot"] = inputs["mdot_hot"] * inputs["dh_hot"] / inputs["xs_area_hot"] / inputs["mu_hot"]

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        J["Re_dh_cold", "mdot_cold"] = inputs["dh_cold"] / inputs["xs_area_cold"] / inputs["mu_cold"] * np.ones((nn,))
        J["Re_dh_cold", "dh_cold"] = inputs["mdot_cold"] / inputs["xs_area_cold"] / inputs["mu_cold"]
        J["Re_dh_cold", "xs_area_cold"] = (
            -inputs["mdot_cold"] * inputs["dh_cold"] / inputs["xs_area_cold"] ** 2 / inputs["mu_cold"]
        )
        J["Re_dh_cold", "mu_cold"] = (
            -inputs["mdot_cold"] * inputs["dh_cold"] / inputs["xs_area_cold"] / inputs["mu_cold"] ** 2
        )
        J["Re_dh_hot", "mdot_hot"] = inputs["dh_hot"] / inputs["xs_area_hot"] / inputs["mu_hot"] * np.ones((nn,))
        J["Re_dh_hot", "dh_hot"] = inputs["mdot_hot"] / inputs["xs_area_hot"] / inputs["mu_hot"]
        J["Re_dh_hot", "xs_area_hot"] = (
            -inputs["mdot_hot"] * inputs["dh_hot"] / inputs["xs_area_hot"] ** 2 / inputs["mu_hot"]
        )
        J["Re_dh_hot", "mu_hot"] = (
            -inputs["mdot_hot"] * inputs["dh_hot"] / inputs["xs_area_hot"] / inputs["mu_hot"] ** 2
        )


class NusseltFromColburnJ(ExplicitComponent):
    """
    Computes Nu from the Colburn j factor, Re, and Pr (mu, cp, k).
    Nu = j * Redh * (cp mu / k) ^(1/3)

    Inputs
    ------
    Re_dh_cold : float
        Reynolds number based on the hydraulic diameter, cold side (vector, dimensionless)
    j_cold : float
        Colburn j factor (vector, dimensionless)
    k_cold : float
        Thermal conductivity of the cold side fluid (scalar, W/m/K)
    mu_cold : float
        Dynamic viscosity of the cold side fluid (scalar, kg/m/s)
    cp_cold : float
        Specific heat at constant pressure, cold side (scalar, J/kg/K)
    Re_dh_hot : float
        Reynolds number based on the hydraulic diameter, hot side (vector, dimensionless)
    j_hot : float
        Colburn j factor (vector, dimensionless)
    k_hot : float
        Thermal conductivity of the hot side fluid (scalar, W/m/K)
    mu_hot : float
        Dynamic viscosity of the hot side fluid (scalar, kg/m/s)
    cp_hot : float
        Specific heat at constant pressure, hot side (scalar, J/kg/K)

    Outputs
    -------
    Nu_dh_cold : float
        Hydraulic diameter Nusselt number (vector, dimensionless)
    Nu_dh_hot : float
        Hydraulic diameter Nusselt number (vector, dimensionless

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("Re_dh_cold", shape=(nn,))
        self.add_input("j_cold", shape=(nn,))
        self.add_input("k_cold", units="W/m/K")
        self.add_input("mu_cold", units="kg/m/s")
        self.add_input("cp_cold", units="J/kg/K")
        self.add_input("Re_dh_hot", shape=(nn,))
        self.add_input("j_hot", shape=(nn,))
        self.add_input("k_hot", units="W/m/K")
        self.add_input("mu_hot", units="kg/m/s")
        self.add_input("cp_hot", units="J/kg/K")

        self.add_output("Nu_dh_cold", shape=(nn,), lower=0.001)
        self.add_output("Nu_dh_hot", shape=(nn,))
        arange = np.arange(0, nn)
        self.declare_partials(["Nu_dh_cold"], ["j_cold", "Re_dh_cold"], rows=arange, cols=arange)
        self.declare_partials(
            ["Nu_dh_cold"], ["k_cold", "mu_cold", "cp_cold"], rows=arange, cols=np.zeros((nn,), dtype=np.int32)
        )
        self.declare_partials(["Nu_dh_hot"], ["j_hot", "Re_dh_hot"], rows=arange, cols=arange)
        self.declare_partials(
            ["Nu_dh_hot"], ["k_hot", "mu_hot", "cp_hot"], rows=arange, cols=np.zeros((nn,), dtype=np.int32)
        )

    def compute(self, inputs, outputs):
        outputs["Nu_dh_cold"] = (
            inputs["j_cold"]
            * inputs["Re_dh_cold"]
            * inputs["mu_cold"] ** (1 / 3)
            * inputs["cp_cold"] ** (1 / 3)
            / inputs["k_cold"] ** (1 / 3)
        )
        outputs["Nu_dh_hot"] = (
            inputs["j_hot"]
            * inputs["Re_dh_hot"]
            * inputs["mu_hot"] ** (1 / 3)
            * inputs["cp_hot"] ** (1 / 3)
            / inputs["k_hot"] ** (1 / 3)
        )

    def compute_partials(self, inputs, J):
        J["Nu_dh_cold", "mu_cold"] = (
            1
            / 3
            * (
                inputs["j_cold"]
                * inputs["Re_dh_cold"]
                * inputs["mu_cold"] ** (-2 / 3)
                * inputs["cp_cold"] ** (1 / 3)
                / inputs["k_cold"] ** (1 / 3)
            )
        )
        J["Nu_dh_cold", "cp_cold"] = (
            1
            / 3
            * (
                inputs["j_cold"]
                * inputs["Re_dh_cold"]
                * inputs["mu_cold"] ** (1 / 3)
                * inputs["cp_cold"] ** (-2 / 3)
                / inputs["k_cold"] ** (1 / 3)
            )
        )
        J["Nu_dh_cold", "k_cold"] = (
            -1
            / 3
            * (
                inputs["j_cold"]
                * inputs["Re_dh_cold"]
                * inputs["mu_cold"] ** (1 / 3)
                * inputs["cp_cold"] ** (1 / 3)
                / inputs["k_cold"] ** (4 / 3)
            )
        )
        J["Nu_dh_cold", "j_cold"] = (
            inputs["Re_dh_cold"]
            * inputs["mu_cold"] ** (1 / 3)
            * inputs["cp_cold"] ** (1 / 3)
            / inputs["k_cold"] ** (1 / 3)
        )
        J["Nu_dh_cold", "Re_dh_cold"] = (
            inputs["j_cold"] * inputs["mu_cold"] ** (1 / 3) * inputs["cp_cold"] ** (1 / 3) / inputs["k_cold"] ** (1 / 3)
        )

        J["Nu_dh_hot", "mu_hot"] = (
            1
            / 3
            * (
                inputs["j_hot"]
                * inputs["Re_dh_hot"]
                * inputs["mu_hot"] ** (-2 / 3)
                * inputs["cp_hot"] ** (1 / 3)
                / inputs["k_hot"] ** (1 / 3)
            )
        )
        J["Nu_dh_hot", "cp_hot"] = (
            1
            / 3
            * (
                inputs["j_hot"]
                * inputs["Re_dh_hot"]
                * inputs["mu_hot"] ** (1 / 3)
                * inputs["cp_hot"] ** (-2 / 3)
                / inputs["k_hot"] ** (1 / 3)
            )
        )
        J["Nu_dh_hot", "k_hot"] = (
            -1
            / 3
            * (
                inputs["j_hot"]
                * inputs["Re_dh_hot"]
                * inputs["mu_hot"] ** (1 / 3)
                * inputs["cp_hot"] ** (1 / 3)
                / inputs["k_hot"] ** (4 / 3)
            )
        )
        J["Nu_dh_hot", "j_hot"] = (
            inputs["Re_dh_hot"] * inputs["mu_hot"] ** (1 / 3) * inputs["cp_hot"] ** (1 / 3) / inputs["k_hot"] ** (1 / 3)
        )
        J["Nu_dh_hot", "Re_dh_hot"] = (
            inputs["j_hot"] * inputs["mu_hot"] ** (1 / 3) * inputs["cp_hot"] ** (1 / 3) / inputs["k_hot"] ** (1 / 3)
        )


class ConvectiveCoefficient(ExplicitComponent):
    """
    Computes h from Nu_Dh (hydraulic diam Nusselt number), Dh (hyd diam), and k (thermal conductivity)

    Inputs
    ------
    Nu_dh_cold : float
        Hydraulic diameter Nusselt number (vector, dimensionless)
    dh_cold : float
        Hydraulic diameter of the cold side flow channels (scalar, m)
    k_cold : float
        Thermal conductivity of the cold side fluid (scalar, W/m/K)

    Nu_dh_cold : float
        Hydraulic diameter Nusselt number (vector, dimensionless)
    dh_cold : float
        Hydraulic diameter of the cold side flow channels (scalar, m)
    k_cold : float
        Thermal conductivity of the cold side fluid (scalar, W/m/K)

    Outputs
    -------
    h_conv_cold : float
        Convective heat transfer coefficient (vector, dimensionless)
    h_conv_hot : float
        Convective heat transfer coefficient (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("Nu_dh_cold", shape=(nn,))
        self.add_input("dh_cold", units="m")
        self.add_input("k_cold", units="W/m/K")
        self.add_input("Nu_dh_hot", shape=(nn,))
        self.add_input("dh_hot", units="m")
        self.add_input("k_hot", units="W/m/K")

        self.add_output("h_conv_cold", shape=(nn,), units="W/m**2/K", lower=1.0)
        self.add_output("h_conv_hot", shape=(nn,), units="W/m**2/K")
        arange = np.arange(0, nn)
        self.declare_partials(["h_conv_cold"], ["Nu_dh_cold"], rows=arange, cols=arange)
        self.declare_partials(["h_conv_cold"], ["dh_cold", "k_cold"], rows=arange, cols=np.zeros((nn,), dtype=np.int32))
        self.declare_partials(["h_conv_hot"], ["Nu_dh_hot"], rows=arange, cols=arange)
        self.declare_partials(["h_conv_hot"], ["dh_hot", "k_hot"], rows=arange, cols=np.zeros((nn,), dtype=np.int32))

    def compute(self, inputs, outputs):
        outputs["h_conv_cold"] = inputs["Nu_dh_cold"] * inputs["k_cold"] / inputs["dh_cold"]
        outputs["h_conv_hot"] = inputs["Nu_dh_hot"] * inputs["k_hot"] / inputs["dh_hot"]
        if np.min(outputs["h_conv_cold"]) <= 0.0:
            raise ValueError(self.msginfo)

    def compute_partials(self, inputs, J):
        J["h_conv_cold", "Nu_dh_cold"] = inputs["k_cold"] / inputs["dh_cold"]
        J["h_conv_cold", "k_cold"] = inputs["Nu_dh_cold"] / inputs["dh_cold"]
        J["h_conv_cold", "dh_cold"] = -inputs["Nu_dh_cold"] * inputs["k_cold"] / inputs["dh_cold"] ** 2

        J["h_conv_hot", "Nu_dh_hot"] = inputs["k_hot"] / inputs["dh_hot"]
        J["h_conv_hot", "k_hot"] = inputs["Nu_dh_hot"] / inputs["dh_hot"]
        J["h_conv_hot", "dh_hot"] = -inputs["Nu_dh_hot"] * inputs["k_hot"] / inputs["dh_hot"] ** 2


class FinEfficiency(ExplicitComponent):
    """
    Computes overall heat transfer efficiency eta_0 including fin efficiency
    This accounts for the actual heat transfer being less than if the
    temperature of the fin were uniform.
    If conduction is not perfect, temperature drop along the fin
    results in less than unity efficiency.

    Method described in Fundamentals of Heat and Mass Transfer 6th Edition (Incropera and DeWitt)

    Inputs
    -------
    h_conv_cold : float
        Convective heat transfer coefficient (vector, dimensionless)
    fin_area_ratio_cold : float
        Ratio of fin area to total heat transfer area (scalar, dimensionless)
    channel_height_cold : float
        Height of each finned flow channel on the cold side (scalar, m)
    h_conv_hot : float
        Convective heat transfer coefficient (vector, dimensionless)
    fin_area_ratio_hot : float
        Ratio of fin area to total heat transfer area (scalar, dimensionless)
    channel_height_hot : float
        Height of each finned flow channel on the hot side (scalar, m)
    fin_thickness : float
        Thickness of fin material (scalar, m)
    material_k : float
        Thermal conductivity of fin material (scalar, W/m/K)

    Outputs
    -------
    eta_overall_cold : float
        Overall heat transfer efficiency including fin efficiency (vector, dimensionless)
    eta_overall_hot : float
        Overall heat transfer efficiency including fin efficiency (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("h_conv_cold", shape=(nn,), units="W/m**2/K")
        self.add_input("fin_area_ratio_cold")
        self.add_input("channel_height_cold", units="m")
        self.add_input("h_conv_hot", shape=(nn,), units="W/m**2/K")
        self.add_input("fin_area_ratio_hot")
        self.add_input("channel_height_hot", units="m")
        self.add_input("fin_thickness", units="m")
        self.add_input("material_k", units="W/m/K")

        self.add_output("eta_overall_cold", shape=(nn,))
        self.add_output("eta_overall_hot", shape=(nn,))
        arange = np.arange(0, nn)
        self.declare_partials(["eta_overall_cold"], ["h_conv_cold"], rows=arange, cols=arange)
        self.declare_partials(
            ["eta_overall_cold"],
            ["fin_area_ratio_cold", "channel_height_cold", "fin_thickness", "material_k"],
            rows=arange,
            cols=np.zeros((nn,), dtype=np.int32),
        )
        self.declare_partials(["eta_overall_hot"], ["h_conv_hot"], rows=arange, cols=arange)
        self.declare_partials(
            ["eta_overall_hot"],
            ["fin_area_ratio_hot", "channel_height_hot", "fin_thickness", "material_k"],
            rows=arange,
            cols=np.zeros((nn,), dtype=np.int32),
        )

    def compute(self, inputs, outputs):
        k = inputs["material_k"]
        t_f = inputs["fin_thickness"]

        l_f_c = inputs["channel_height_cold"]
        h_c = inputs["h_conv_cold"]

        m_cold = np.sqrt(2 * h_c / k / t_f)
        if np.min(h_c) <= 0.0:
            raise ValueError(self.msginfo)
        eta_f_cold = 2 * np.tanh(m_cold * l_f_c / 2) / m_cold / l_f_c
        outputs["eta_overall_cold"] = 1 - inputs["fin_area_ratio_cold"] * (1 - eta_f_cold)

        l_f_h = inputs["channel_height_hot"]
        h_h = inputs["h_conv_hot"]

        m_hot = np.sqrt(2 * h_h / k / t_f)
        eta_f_hot = 2 * np.tanh(m_hot * l_f_h / 2) / m_hot / l_f_h
        outputs["eta_overall_hot"] = 1 - inputs["fin_area_ratio_hot"] * (1 - eta_f_hot)

    def compute_partials(self, inputs, J):
        # get some aliases for brevity
        t_f = inputs["fin_thickness"]
        k = inputs["material_k"]

        l_f_c = inputs["channel_height_cold"]
        h_c = inputs["h_conv_cold"]
        afa_c = inputs["fin_area_ratio_cold"]

        m_cold = np.sqrt(2 * h_c / k / t_f)
        eta_f_cold = 2 * np.tanh(m_cold * l_f_c / 2) / m_cold / l_f_c

        # compute partials of m with respect to its inputs
        dmdh_c = (2 * h_c * t_f * k) ** (-1 / 2)
        dmdt_c = -((h_c / t_f**3 / k / 2) ** (1 / 2))
        dmdk_c = -((h_c / k**3 / t_f / 2) ** (1 / 2))
        # compute partials of fin efficiency with respect to its inputs
        ml_c = m_cold * l_f_c
        deta_fdm_c = (ml_c * np.cosh(ml_c / 2) ** -2 - 2 * np.tanh(ml_c / 2)) / ml_c / m_cold
        deta_fdL_c = (ml_c * np.cosh(ml_c / 2) ** -2 - 2 * np.tanh(ml_c / 2)) / ml_c / l_f_c
        # compute partials with respect to overall efficiency
        J["eta_overall_cold", "fin_area_ratio_cold"] = eta_f_cold - 1
        J["eta_overall_cold", "channel_height_cold"] = afa_c * deta_fdL_c
        J["eta_overall_cold", "fin_thickness"] = afa_c * deta_fdm_c * dmdt_c
        J["eta_overall_cold", "material_k"] = afa_c * deta_fdm_c * dmdk_c
        J["eta_overall_cold", "h_conv_cold"] = afa_c * deta_fdm_c * dmdh_c

        l_f_h = inputs["channel_height_hot"]
        h_h = inputs["h_conv_hot"]
        afa_h = inputs["fin_area_ratio_hot"]

        m_hot = np.sqrt(2 * h_h / k / t_f)
        eta_f_hot = 2 * np.tanh(m_hot * l_f_h / 2) / m_hot / l_f_h

        # compute partials of m with respect to its inputs
        dmdh_h = (2 * h_h * t_f * k) ** (-1 / 2)
        dmdt_h = -((h_h / t_f**3 / k / 2) ** (1 / 2))
        dmdk_h = -((h_h / k**3 / t_f / 2) ** (1 / 2))
        # compute partials of fin efficiency with respect to its inputs
        ml_h = m_hot * l_f_h
        deta_fdm_h = (ml_h * np.cosh(ml_h / 2) ** -2 - 2 * np.tanh(ml_h / 2)) / ml_h / m_hot
        deta_fdL_h = (ml_h * np.cosh(ml_h / 2) ** -2 - 2 * np.tanh(ml_h / 2)) / ml_h / l_f_h
        # compute partials with respect to overall efficiency
        J["eta_overall_hot", "fin_area_ratio_hot"] = eta_f_hot - 1
        J["eta_overall_hot", "channel_height_hot"] = afa_h * deta_fdL_h
        J["eta_overall_hot", "fin_thickness"] = afa_h * deta_fdm_h * dmdt_h
        J["eta_overall_hot", "material_k"] = afa_h * deta_fdm_h * dmdk_h
        J["eta_overall_hot", "h_conv_hot"] = afa_h * deta_fdm_h * dmdh_h


class UAOverall(ExplicitComponent):
    """
    Computes overall heat transfer coefficient for a heat exchanger

    Method from Kays and London and Incropera and DeWitt

    Inputs
    -------
    h_conv_cold : float
        Convective heat transfer coefficient (vector, W/m**2/K)
    heat_transfer_area_cold : float
        Total cold-side heat transfer area (scalar, m**2)
    eta_overall_cold : float
        Overall thermal efficiency for the cold side (vector, dimensionless)
    h_conv_hot : float
        Convective heat transfer coefficient (vector, W/m**2/K)
    heat_transfer_area_hot : float
        Total hot-side heat transfer area (scalar, m**2)
    eta_overall_hot : float
        Overall thermal efficiency for the colhotd side (vector, dimensionless)

    Outputs
    -------
    UA_overall : float
        Inverse overall thermal resistance for the entire heat exchanger (vector, W/K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    fouling_factor_hot : float
        Fouling factor, hot side (scalar, m**2 K / W)
    fouling_factor_cold : float
        Fouling factor, cold side (scalar, m**2 K /W)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("fouling_factor_cold", default=0.0)
        self.options.declare("fouling_factor_hot", default=0.0)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("h_conv_cold", shape=(nn,), units="W/m**2/K")
        self.add_input("heat_transfer_area_cold", units="m**2")
        self.add_input("eta_overall_cold", shape=(nn,))
        self.add_input("h_conv_hot", shape=(nn,), units="W/m**2/K")
        self.add_input("heat_transfer_area_hot", units="m**2")
        self.add_input("eta_overall_hot", shape=(nn,))

        self.add_output("UA_overall", shape=(nn,), units="W/K")
        arange = np.arange(0, nn)
        self.declare_partials(
            ["UA_overall"],
            ["h_conv_cold", "eta_overall_cold", "h_conv_hot", "eta_overall_hot"],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            ["UA_overall"],
            ["heat_transfer_area_cold", "heat_transfer_area_hot"],
            rows=arange,
            cols=np.zeros((nn,), dtype=np.int32),
        )

    def compute(self, inputs, outputs):
        Rc = 1 / inputs["h_conv_cold"] / inputs["heat_transfer_area_cold"] / inputs["eta_overall_cold"]
        Rfc = self.options["fouling_factor_cold"] / inputs["heat_transfer_area_cold"] / inputs["eta_overall_cold"]
        Rh = 1 / inputs["h_conv_hot"] / inputs["heat_transfer_area_hot"] / inputs["eta_overall_hot"]
        Rfc = self.options["fouling_factor_hot"] / inputs["heat_transfer_area_hot"] / inputs["eta_overall_hot"]
        outputs["UA_overall"] = 1 / (Rc + Rfc + Rh + Rfc)

    def compute_partials(self, inputs, J):
        Rc = 1 / inputs["h_conv_cold"] / inputs["heat_transfer_area_cold"] / inputs["eta_overall_cold"]
        Rfc = self.options["fouling_factor_cold"] / inputs["heat_transfer_area_cold"] / inputs["eta_overall_cold"]
        Rh = 1 / inputs["h_conv_hot"] / inputs["heat_transfer_area_hot"] / inputs["eta_overall_hot"]
        Rfc = self.options["fouling_factor_hot"] / inputs["heat_transfer_area_hot"] / inputs["eta_overall_hot"]

        dRcdh = -1 / inputs["h_conv_cold"] ** 2 / inputs["heat_transfer_area_cold"] / inputs["eta_overall_cold"]
        dRcdA = -1 / inputs["h_conv_cold"] / inputs["heat_transfer_area_cold"] ** 2 / inputs["eta_overall_cold"]
        dRcdeta = -1 / inputs["h_conv_cold"] / inputs["heat_transfer_area_cold"] / inputs["eta_overall_cold"] ** 2
        dRfcdA = (
            -self.options["fouling_factor_cold"] / inputs["heat_transfer_area_cold"] ** 2 / inputs["eta_overall_cold"]
        )
        dRfcdeta = (
            -self.options["fouling_factor_cold"] / inputs["heat_transfer_area_cold"] / inputs["eta_overall_cold"] ** 2
        )

        dRhdh = -1 / inputs["h_conv_hot"] ** 2 / inputs["heat_transfer_area_hot"] / inputs["eta_overall_hot"]
        dRhdA = -1 / inputs["h_conv_hot"] / inputs["heat_transfer_area_hot"] ** 2 / inputs["eta_overall_hot"]
        dRhdeta = -1 / inputs["h_conv_hot"] / inputs["heat_transfer_area_hot"] / inputs["eta_overall_hot"] ** 2
        dRfhdA = -self.options["fouling_factor_hot"] / inputs["heat_transfer_area_hot"] ** 2 / inputs["eta_overall_hot"]
        dRfhdeta = (
            -self.options["fouling_factor_hot"] / inputs["heat_transfer_area_hot"] / inputs["eta_overall_hot"] ** 2
        )

        J["UA_overall", "h_conv_cold"] = -1 / (Rc + Rfc + Rh + Rfc) ** 2 * (dRcdh)
        J["UA_overall", "heat_transfer_area_cold"] = -1 / (Rc + Rfc + Rh + Rfc) ** 2 * (dRcdA + dRfcdA)
        J["UA_overall", "eta_overall_cold"] = -1 / (Rc + Rfc + Rh + Rfc) ** 2 * (dRcdeta + dRfcdeta)
        J["UA_overall", "h_conv_hot"] = -1 / (Rc + Rfc + Rh + Rfc) ** 2 * (dRhdh)
        J["UA_overall", "heat_transfer_area_hot"] = -1 / (Rc + Rfc + Rh + Rfc) ** 2 * (dRhdA + dRfhdA)
        J["UA_overall", "eta_overall_hot"] = -1 / (Rc + Rfc + Rh + Rfc) ** 2 * (dRhdeta + dRfhdeta)


class NTUMethod(ExplicitComponent):
    """
    Computes number of thermal units and maximum possible heat transfer.

    Method described in Incropera and DeWitt and Kays and London

    Inputs
    -------
    UA_overall : float
        Overall inverse thermal resistance (vector, W/K)
    mdot_cold : float
        Mass flow rate, cold side  (vector, kg/s)
    T_in_cold : float
        Inlet temperature, cold side (vector, K)
    cp_cold : float
        Specific heat at constant pressure, cold side (scalar, J/kg/K)
    mdot_hot : float
        Mass flow rate, cold side  (vector, kg/s)
    T_in_hot : float
        Inlet temperature, hot side (vector, K)
    cp_hot : float
        Specific heat at constant pressure, cold side (scalar, J/kg/K)

    Outputs
    -------
    NTU : float
        Number of thermal units (vector, dimensionless)
    heat_max : float
        Maximum possible heat transfer (vector, W)
    C_ratio : float
        The ratio of the maximum mdot*cp to the maximum (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("UA_overall", shape=(nn,), units="W/K")
        self.add_input("mdot_cold", shape=(nn,), units="kg/s")
        self.add_input("T_in_cold", shape=(nn,), units="K")
        self.add_input("cp_cold", units="J/kg/K")
        self.add_input("mdot_hot", shape=(nn,), units="kg/s")
        self.add_input("T_in_hot", shape=(nn,), units="K")
        self.add_input("cp_hot", units="J/kg/K")

        self.add_output("NTU", shape=(nn,), lower=0.1)
        self.add_output("heat_max", shape=(nn,), units="W")
        self.add_output("C_ratio", shape=(nn,))

        arange = np.arange(0, nn)
        self.declare_partials(["NTU"], ["UA_overall", "mdot_cold", "mdot_hot"], rows=arange, cols=arange)
        self.declare_partials(["C_ratio"], ["mdot_cold", "mdot_hot"], rows=arange, cols=arange)
        self.declare_partials(
            ["heat_max"], ["mdot_cold", "mdot_hot", "T_in_cold", "T_in_hot"], rows=arange, cols=arange
        )
        self.declare_partials(
            ["heat_max", "NTU", "C_ratio"], ["cp_cold", "cp_hot"], rows=arange, cols=np.zeros((nn,), dtype=np.int32)
        )

    def compute(self, inputs, outputs):
        C_cold = inputs["mdot_cold"] * inputs["cp_cold"]
        C_hot = inputs["mdot_hot"] * inputs["cp_hot"]
        C_min_bool = np.less(C_cold, C_hot)
        C_min = np.where(C_min_bool, C_cold, C_hot)
        C_max = np.where(C_min_bool, C_hot, C_cold)
        outputs["NTU"] = inputs["UA_overall"] / C_min
        outputs["heat_max"] = (inputs["T_in_hot"] - inputs["T_in_cold"]) * C_min
        outputs["C_ratio"] = C_min / C_max

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        C_cold = inputs["mdot_cold"] * inputs["cp_cold"]
        C_hot = inputs["mdot_hot"] * inputs["cp_hot"]
        C_min_bool = np.less(C_cold, C_hot)
        C_min = np.where(C_min_bool, C_cold, C_hot)
        C_max = np.where(C_min_bool, C_hot, C_cold)
        dCmindmdotcold = np.where(C_min_bool, inputs["cp_cold"] * np.ones((nn,)), np.zeros((nn,)))
        dCmindcpcold = np.where(C_min_bool, inputs["mdot_cold"], np.zeros((nn,)))
        dCmindmdothot = np.where(C_min_bool, np.zeros((nn,)), inputs["cp_hot"] * np.ones((nn,)))
        dCmindcphot = np.where(C_min_bool, np.zeros((nn,)), inputs["mdot_hot"])

        dCmaxdmdotcold = np.where(C_min_bool, np.zeros((nn,)), inputs["cp_cold"] * np.ones((nn,)))
        dCmaxdcpcold = np.where(C_min_bool, np.zeros((nn,)), inputs["mdot_cold"])
        dCmaxdmdothot = np.where(C_min_bool, inputs["cp_hot"] * np.ones((nn,)), np.zeros((nn,)))
        dCmaxdcphot = np.where(C_min_bool, inputs["mdot_hot"], np.zeros((nn,)))

        J["NTU", "UA_overall"] = 1 / C_min
        J["NTU", "mdot_cold"] = -inputs["UA_overall"] / C_min**2 * dCmindmdotcold
        J["NTU", "cp_cold"] = -inputs["UA_overall"] / C_min**2 * dCmindcpcold
        J["NTU", "mdot_hot"] = -inputs["UA_overall"] / C_min**2 * dCmindmdothot
        J["NTU", "cp_hot"] = -inputs["UA_overall"] / C_min**2 * dCmindcphot

        J["heat_max", "T_in_cold"] = -C_min
        J["heat_max", "T_in_hot"] = C_min
        J["heat_max", "mdot_cold"] = (inputs["T_in_hot"] - inputs["T_in_cold"]) * dCmindmdotcold
        J["heat_max", "cp_cold"] = (inputs["T_in_hot"] - inputs["T_in_cold"]) * dCmindcpcold
        J["heat_max", "mdot_hot"] = (inputs["T_in_hot"] - inputs["T_in_cold"]) * dCmindmdothot
        J["heat_max", "cp_hot"] = (inputs["T_in_hot"] - inputs["T_in_cold"]) * dCmindcphot

        J["C_ratio", "mdot_cold"] = (dCmindmdotcold * C_max - dCmaxdmdotcold * C_min) / C_max**2
        J["C_ratio", "cp_cold"] = (dCmindcpcold * C_max - dCmaxdcpcold * C_min) / C_max**2
        J["C_ratio", "mdot_hot"] = (dCmindmdothot * C_max - dCmaxdmdothot * C_min) / C_max**2
        J["C_ratio", "cp_hot"] = (dCmindcphot * C_max - dCmaxdcphot * C_min) / C_max**2


class CrossFlowNTUEffectiveness(ExplicitComponent):
    """
    Computes the heat transfer effectiveness of a crossflow heat exchanger

    Expression from Kays and London

    Inputs
    -------
    NTU : float
        Number of thermal units (vector, dimensionless)
    C_ratio : float
        Ratio of mdot * cp _min to _max (vector, dimensionless)

    Outputs
    -------
    effectiveness : float
        Heat transfer effectiveness (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("NTU", shape=(nn,))
        self.add_input("C_ratio", shape=(nn,))
        self.add_output("effectiveness", shape=(nn,))
        arange = np.arange(0, nn)
        self.declare_partials(["effectiveness"], ["NTU", "C_ratio"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        Cr = inputs["C_ratio"]
        ntu = inputs["NTU"]
        outputs["effectiveness"] = 1 - np.exp(ntu**0.22 / Cr * (np.exp(-Cr * ntu**0.78) - 1))

    def compute_partials(self, inputs, J):
        Cr = inputs["C_ratio"]
        ntu = inputs["NTU"]
        J["effectiveness", "C_ratio"] = -np.exp((ntu**0.22 * (np.exp(-Cr * ntu**0.78) - 1)) / Cr) * (
            -(ntu**0.22 * (np.exp(-Cr * ntu**0.78) - 1)) / Cr**2 - (ntu * np.exp(-Cr * ntu**0.78)) / Cr
        )
        J["effectiveness", "NTU"] = (
            39 * Cr * ntu**0.78 * np.exp((ntu**0.22 * (np.exp(-Cr * ntu**0.78) - 1)) / Cr - Cr * ntu**0.78)
            - 11 * np.exp((ntu**0.22 * (np.exp(-Cr * ntu**0.78) - 1)) / Cr) * (np.exp(-Cr * ntu**0.78) - 1)
        ) / (50 * Cr * ntu**0.78)


class NTUEffectivenessActualHeatTransfer(ExplicitComponent):
    """
    Computes the actual heat transfer and outlet temperatures of a heat exchanger
    using the NTU-effectiveness method described in Kays and London
    and Incropera and DeWitt

    Inputs
    -------
    effectiveness : float
        Heat transfer effectiveness (vector, dimensionless)
    heat_max : float
        Maximum possible heat transfer (vector, W)

    Outputs
    -------
    heat_transfer : float
        Actual heat transfer from hot to cold side (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("effectiveness", shape=(nn,))
        self.add_input("heat_max", shape=(nn,), units="W")
        self.add_output("heat_transfer", shape=(nn,), units="W")

        arange = np.arange(0, nn)
        self.declare_partials(["heat_transfer"], ["effectiveness", "heat_max"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs["heat_transfer"] = inputs["effectiveness"] * inputs["heat_max"]

    def compute_partials(self, inputs, J):
        J["heat_transfer", "effectiveness"] = inputs["heat_max"]
        J["heat_transfer", "heat_max"] = inputs["effectiveness"]


class OutletTemperatures(ExplicitComponent):
    """
    Computes outlet temperatures of hot and cold streams, given
    mass flow rates, cp, and heat transfer

    Inputs
    -------
    heat_transfer : float
        Actual heat transfer from hot to cold side (vector, W)
    mdot_cold : float
        Mass flow rate, cold side  (vector, kg/s)
    T_in_cold : float
        Inlet temperature, cold side (vector, K)
    cp_cold : float
        Specific heat at constant pressure, cold side (scalar, J/kg/K)
    mdot_hot : float
        Mass flow rate, cold side  (vector, kg/s)
    T_in_hot : float
        Inlet temperature, hot side (vector, K)
    cp_hot : float
        Specific heat at constant pressure, cold side (scalar, J/kg/K)

    Outputs
    -------
    T_out_cold : float
        Outlet temperature, cold side (vector, K)
    T_out_hot : float
        Outlet temperature, hot side (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("heat_transfer", shape=(nn,), units="W")
        self.add_input("mdot_cold", shape=(nn,), units="kg/s")
        self.add_input("T_in_cold", shape=(nn,), units="K")
        self.add_input("cp_cold", units="J/kg/K")
        self.add_input("mdot_hot", shape=(nn,), units="kg/s")
        self.add_input("T_in_hot", shape=(nn,), units="K")
        self.add_input("cp_hot", units="J/kg/K")
        self.add_output("T_out_cold", shape=(nn,), units="K")
        self.add_output("T_out_hot", shape=(nn,), units="K")

        arange = np.arange(0, nn)
        self.declare_partials(["T_out_cold"], ["heat_transfer", "mdot_cold"], rows=arange, cols=arange)
        self.declare_partials(["T_out_cold"], ["T_in_cold"], rows=arange, cols=arange, val=np.ones((nn,)))
        self.declare_partials(["T_out_hot"], ["heat_transfer", "mdot_hot"], rows=arange, cols=arange)
        self.declare_partials(["T_out_hot"], ["T_in_hot"], rows=arange, cols=arange, val=np.ones((nn,)))
        self.declare_partials(["T_out_cold"], ["cp_cold"], rows=arange, cols=np.zeros((nn,), dtype=np.int32))
        self.declare_partials(["T_out_hot"], ["cp_hot"], rows=arange, cols=np.zeros((nn,), dtype=np.int32))

    def compute(self, inputs, outputs):
        outputs["T_out_cold"] = inputs["T_in_cold"] + inputs["heat_transfer"] / inputs["mdot_cold"] / inputs["cp_cold"]
        outputs["T_out_hot"] = inputs["T_in_hot"] - inputs["heat_transfer"] / inputs["mdot_hot"] / inputs["cp_hot"]

    def compute_partials(self, inputs, J):
        J["T_out_cold", "heat_transfer"] = 1 / inputs["mdot_cold"] / inputs["cp_cold"]
        J["T_out_cold", "mdot_cold"] = -inputs["heat_transfer"] / inputs["mdot_cold"] ** 2 / inputs["cp_cold"]
        J["T_out_cold", "cp_cold"] = -inputs["heat_transfer"] / inputs["mdot_cold"] / inputs["cp_cold"] ** 2
        J["T_out_hot", "heat_transfer"] = -1 / inputs["mdot_hot"] / inputs["cp_hot"]
        J["T_out_hot", "mdot_hot"] = inputs["heat_transfer"] / inputs["mdot_hot"] ** 2 / inputs["cp_hot"]
        J["T_out_hot", "cp_hot"] = inputs["heat_transfer"] / inputs["mdot_hot"] / inputs["cp_hot"] ** 2


class PressureDrop(ExplicitComponent):
    """
    Computes total pressure drop in the hot and cold streams

    Method and estimated parameters from Kays and London

    Inputs
    -------
    length_overall : float
        Overall length of the cold side flowpath (scalar, m)
    width_overall : float
        Overall length of the hot side flowpath (scalar, m)
    f_cold : float
        Fanning friction factor (vector, dimensionless)
    mdot_cold : float
        Mass flow rate, cold side  (vector, kg/s)
    rho_cold : float
        Inlet density, cold side (vector, kg/m**3)
    dh_cold : float
        Hydraulic diameter of the cold side flow channels (scalar, m)
    xs_area_cold : float
        Cross-sectional flow area of the cold side (scalar, m**2)
    f_hot : float
        Fanning friction factor (vector, dimensionless)
    mdot_hot : float
        Mass flow rate, hot side  (vector, kg/s)
    rho_hot : float
        Inlet density, hot side (vector, kg/m**3)
    dh_hot : float
        Hydraulic diameter of the hot side flow channels (scalar, m)
    xs_area_hot : float
        Cross-sectional flow area of the hot side (scalar, m**2)

    Outputs
    -------
    delta_p_cold : float
        Pressure drop, cold side. Negative is pressure drop (vector, Pa)
    delta_p_hot : float
        Pressure drop, cold side. Negative is pressure drop (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    Kc_cold : float
        Irreversible contraction loss coefficient (per Kays and London) (scalar, dimensionless)
    Ke_cold : float
        Irreversible expansion loss coefficient (per Kays and London) (scalar, dimensionless)
    Kc_hot : float
        Irreversible contraction loss coefficient (per Kays and London) (scalar, dimensionless)
    Ke_hot : float
        Irreversible expansion loss coefficient (per Kays and London) (scalar, dimensionless)

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("Kc_cold", default=0.3, desc="Irreversible contraction loss coefficient")
        self.options.declare("Ke_cold", default=-0.1, desc="Irreversible expansion loss coefficient")
        self.options.declare("Kc_hot", default=0.3, desc="Irreversible contraction loss coefficient")
        self.options.declare("Ke_hot", default=-0.1, desc="Irreversible expansion loss coefficient")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("length_overall", units="m")
        self.add_input("width_overall", units="m")

        self.add_input("f_cold", shape=(nn,))
        self.add_input("mdot_cold", shape=(nn,), units="kg/s")
        self.add_input("rho_cold", shape=(nn,), units="kg/m**3")
        self.add_input("dh_cold", units="m")
        self.add_input("xs_area_cold", units="m**2")

        self.add_input("f_hot", shape=(nn,))
        self.add_input("mdot_hot", shape=(nn,), units="kg/s")
        self.add_input("rho_hot", shape=(nn,), units="kg/m**3")
        self.add_input("dh_hot", units="m")
        self.add_input("xs_area_hot", units="m**2")

        self.add_output("delta_p_cold", shape=(nn,), units="Pa")
        self.add_output("delta_p_hot", shape=(nn,), units="Pa")

        arange = np.arange(0, nn)
        self.declare_partials(["delta_p_cold"], ["mdot_cold", "rho_cold", "f_cold"], rows=arange, cols=arange)
        self.declare_partials(
            ["delta_p_cold"],
            ["xs_area_cold", "length_overall", "dh_cold"],
            rows=arange,
            cols=np.zeros((nn,), dtype=np.int32),
        )
        self.declare_partials(["delta_p_hot"], ["mdot_hot", "rho_hot", "f_hot"], rows=arange, cols=arange)
        self.declare_partials(
            ["delta_p_hot"],
            ["xs_area_hot", "width_overall", "dh_hot"],
            rows=arange,
            cols=np.zeros((nn,), dtype=np.int32),
        )

    def compute(self, inputs, outputs):
        dyn_press_cold = (1 / 2) * (inputs["mdot_cold"] / inputs["xs_area_cold"]) ** 2 / inputs["rho_cold"]
        dyn_press_hot = (1 / 2) * (inputs["mdot_hot"] / inputs["xs_area_hot"]) ** 2 / inputs["rho_hot"]
        Kec = self.options["Ke_cold"]
        Kcc = self.options["Kc_cold"]
        Keh = self.options["Ke_hot"]
        Kch = self.options["Kc_hot"]
        outputs["delta_p_cold"] = dyn_press_cold * (
            -Kec - Kcc - 4 * inputs["length_overall"] * inputs["f_cold"] / inputs["dh_cold"]
        )
        outputs["delta_p_hot"] = dyn_press_hot * (
            -Keh - Kch - 4 * inputs["width_overall"] * inputs["f_hot"] / inputs["dh_hot"]
        )

    def compute_partials(self, inputs, J):
        dyn_press_cold = (1 / 2) * (inputs["mdot_cold"] / inputs["xs_area_cold"]) ** 2 / inputs["rho_cold"]
        dyn_press_hot = (1 / 2) * (inputs["mdot_hot"] / inputs["xs_area_hot"]) ** 2 / inputs["rho_hot"]
        Kec = self.options["Ke_cold"]
        Kcc = self.options["Kc_cold"]
        Keh = self.options["Ke_hot"]
        Kch = self.options["Kc_hot"]
        losses_cold = -Kec - Kcc - 4 * inputs["length_overall"] * inputs["f_cold"] / inputs["dh_cold"]
        losses_hot = -Keh - Kch - 4 * inputs["width_overall"] * inputs["f_hot"] / inputs["dh_hot"]

        J["delta_p_cold", "mdot_cold"] = (
            (inputs["mdot_cold"] / inputs["xs_area_cold"] ** 2) / inputs["rho_cold"] * losses_cold
        )
        J["delta_p_cold", "rho_cold"] = -dyn_press_cold / inputs["rho_cold"] * losses_cold
        J["delta_p_cold", "f_cold"] = dyn_press_cold * (-4 * inputs["length_overall"] / inputs["dh_cold"])
        J["delta_p_cold", "xs_area_cold"] = -2 * dyn_press_cold / inputs["xs_area_cold"] * losses_cold
        J["delta_p_cold", "length_overall"] = dyn_press_cold * (-4 * inputs["f_cold"] / inputs["dh_cold"])
        J["delta_p_cold", "dh_cold"] = dyn_press_cold * (
            4 * inputs["length_overall"] * inputs["f_cold"] / inputs["dh_cold"] ** 2
        )

        J["delta_p_hot", "mdot_hot"] = (
            (inputs["mdot_hot"] / inputs["xs_area_hot"] ** 2) / inputs["rho_hot"] * losses_hot
        )
        J["delta_p_hot", "rho_hot"] = -dyn_press_hot / inputs["rho_hot"] * losses_hot
        J["delta_p_hot", "f_hot"] = dyn_press_hot * (-4 * inputs["width_overall"] / inputs["dh_hot"])
        J["delta_p_hot", "xs_area_hot"] = -2 * dyn_press_hot / inputs["xs_area_hot"] * losses_hot
        J["delta_p_hot", "width_overall"] = dyn_press_hot * (-4 * inputs["f_hot"] / inputs["dh_hot"])
        J["delta_p_hot", "dh_hot"] = dyn_press_hot * (
            4 * inputs["width_overall"] * inputs["f_hot"] / inputs["dh_hot"] ** 2
        )


class HXGroup(Group):
    """
    A heat exchanger model for use with the duct models
    Note that there are many design variables defined as dvs which could be varied
    in optimization.

    Inputs
    ------
    mdot_cold : float
        Mass flow rate of the cold side (air) (vector, kg/s)
    T_in_cold : float
        Inflow temperature of the cold side (air) (vector, K)
    rho_cold : float
        Inflow density of the cold side (air) (vector, kg/m**3)
    mdot_hot : float
        Mass flow rate of the hot side (liquid) (vector, kg/s)
    T_in_hot : float
        Inflow temperature of the hot side (liquid) (vector, kg/s)
    rho_hot : float
        Inflow density of the hot side (liquid) (vector, kg/m**3)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]

        # Set the default values for promoted variables
        self.set_input_defaults("case_thickness", val=2.0, units="mm")
        self.set_input_defaults("fin_thickness", val=0.102, units="mm")
        self.set_input_defaults("plate_thickness", val=0.2, units="mm")
        self.set_input_defaults("material_k", val=190, units="W/m/K")
        self.set_input_defaults("material_rho", val=2700, units="kg/m**3")

        # self.set_input_defaults('mdot_cold', val=np.ones(nn)*1.5, units='kg/s')
        # self.set_input_defaults('rho_cold', val=np.ones(nn)*0.5, units='kg/m**3')
        # self.set_input_defaults('mdot_hot', val=0.075*np.ones(nn), units='kg/s')
        # self.set_input_defaults('rho_hot', val=np.ones(nn)*1020.2, units='kg/m**3')

        # self.set_input_defaults('T_in_cold', val=np.ones(nn)*45, units='degC')
        # self.set_input_defaults('T_in_hot', val=np.ones(nn)*90, units='degC')
        # self.set_input_defaults('n_long_cold', val=3)
        # self.set_input_defaults('n_wide_cold', val=430)
        # self.set_input_defaults('n_tall', val=19)

        self.set_input_defaults("channel_height_cold", val=14, units="mm")
        self.set_input_defaults("channel_width_cold", val=1.35, units="mm")
        self.set_input_defaults("fin_length_cold", val=6, units="mm")
        self.set_input_defaults("cp_cold", val=1005, units="J/kg/K")
        self.set_input_defaults("k_cold", val=0.02596, units="W/m/K")
        self.set_input_defaults("mu_cold", val=1.789e-5, units="kg/m/s")

        self.set_input_defaults("channel_height_hot", val=1, units="mm")
        self.set_input_defaults("channel_width_hot", val=1, units="mm")
        self.set_input_defaults("fin_length_hot", val=6, units="mm")
        self.set_input_defaults("cp_hot", val=3801, units="J/kg/K")
        self.set_input_defaults("k_hot", val=0.405, units="W/m/K")
        self.set_input_defaults("mu_hot", val=1.68e-3, units="kg/m/s")

        dvlist = [
            ["ac|propulsion|thermal|hx|n_wide_cold", "n_wide_cold", 430, None],
            ["ac|propulsion|thermal|hx|n_long_cold", "n_long_cold", 3, None],
            ["ac|propulsion|thermal|hx|n_tall", "n_tall", 19, None],
        ]

        self.add_subsystem("dvpassthru", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])

        self.add_subsystem("osfgeometry", OffsetStripFinGeometry(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "redh", HydraulicDiameterReynoldsNumber(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("osfdata", OffsetStripFinData(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("nusselt", NusseltFromColburnJ(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "convection", ConvectiveCoefficient(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("finefficiency", FinEfficiency(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("ua", UAOverall(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("ntu", NTUMethod(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "effectiveness", CrossFlowNTUEffectiveness(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "heat", NTUEffectivenessActualHeatTransfer(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("t_out", OutletTemperatures(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("delta_p", PressureDrop(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
