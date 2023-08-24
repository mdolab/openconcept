import numpy as np
import openmdao.api as om
from openconcept.utilities import ElementMultiplyDivideComp


class HeatTransferVacuumTank(om.Group):
    """
    Computes the heat transfered from the environment to the propellant. This model
    assumes the tank is a vacuum-insulated tank with MLI insulation.

    This model assumes that the outer wall temperature is equal to the external
    environment temperature and the inner tank wall temperature is equal to the
    gaseous or liquid hydrogen temperature. This is a reasonably good assumption
    because the thermal resistance of the convective heat transfer to the outer
    wall and from the inner wall is much lower than the thermal resistance of the
    vacuum and MLI insulation.

    Inputs
    ------
    T_env : float
        External environment temperature (vector, K)
    N_layers : float
        Number of reflective sheild layers in the MLI, should be at least ~10 for model
        to retain reasonable accuracy (scalar, dimensionless)
    T_liq : float
        Temperature of the liquid hydrogen (vector, K)
    T_gas : float
        Temperature of the gaseous hydrogen (vector, K)
    A_wet : float
        The area of the tank's surface touching the bulk liquid (vector, m^2)
    A_dry : float
        The area of the tank's surface touching the ullage (vector, m^2)

    Outputs
    -------
    Q_gas : float
        Heat flow rate through the tank walls into the ullage (vector, W)
    Q_liq : float
        Heat flow rate through the tank walls into the bulk liquid (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    heat_multiplier : float
        Multiplier on the output heat to account for heat through supports
        and other connections, by default 1.2 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("heat_multiplier", default=1.2, desc="Multiplier on heat leak")

    def setup(self):
        nn = self.options["num_nodes"]

        # Compute the heat flux through the MLI to the propellant
        self.add_subsystem(
            "mli_heat_liq",
            MLIHeatFlux(num_nodes=nn),
            promotes_inputs=[("T_hot", "T_env"), ("T_cold", "T_liq"), "N_layers"],
        )
        self.add_subsystem(
            "mli_heat_gas",
            MLIHeatFlux(num_nodes=nn),
            promotes_inputs=[("T_hot", "T_env"), ("T_cold", "T_gas"), "N_layers"],
        )

        # Scale each flux by the associated area
        mult = self.add_subsystem(
            "scale_by_area",
            ElementMultiplyDivideComp(vec_size=nn),
            promotes_inputs=["A_wet", "A_dry"],
            promotes_outputs=["Q_liq", "Q_gas"],
        )
        mult.add_equation(
            output_name="Q_liq",
            input_names=["flux_liq", "A_wet"],
            vec_size=nn,
            input_units=["W/m**2", "m**2"],
            scaling_factor=self.options["heat_multiplier"],
        )
        mult.add_equation(
            output_name="Q_gas",
            input_names=["flux_gas", "A_dry"],
            vec_size=nn,
            input_units=["W/m**2", "m**2"],
            scaling_factor=self.options["heat_multiplier"],
        )
        self.connect("mli_heat_liq.heat_flux", "scale_by_area.flux_liq")
        self.connect("mli_heat_gas.heat_flux", "scale_by_area.flux_gas")


class MLIHeatFlux(om.ExplicitComponent):
    """
    Compute the heat flux through vacuum MLI insulation. The methodology is from section 4.2.4 of "Thermal
    Performance of Multilayer Insulations" by Keller et al. (https://ntrs.nasa.gov/citations/19740014451).
    This model assumes the interstitial gas in the vacuum is gaseous nitrogen. The assumption has little
    effect if the vacuum is reasonably good (pressure < ~10^-6 torr). The defaults are based on an
    unperforated double-aluminized mylar MLI with a double silk net spacer system.

    Inputs
    ------
    T_hot : float
        Temperature on the hot side of the insulation, usually the outer vacuum wall temp (vector, K)
    T_cold : float
        Temperature on the cold side of the insulation, usually the inner wall temp (vector, K)
    N_layers : float
        Number of reflective sheild layers in the MLI, should be at least ~10 for model
        to retain reasonable accuracy (scalar, dimensionless)

    Outputs
    -------
    heat_flux : float
        Heat flux through the MLI and vacuum insulation (vector, W/m^2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    layer_density : float
        Number of layers per cm, should be between ~2 and ~50 for valid model, by default 30 (scalar, layers/cm)
    solid_cond_coeff : float
        Coefficient on solid conduction, by default 8.95e-8 from Equation 4-56
        in Keller et al. (https://ntrs.nasa.gov/citations/19740014451)
    gas_cond_coeff : float
        Coefficient on gas conduction, by default 1.46e4 from Equation 4-56
        in Keller et al. (https://ntrs.nasa.gov/citations/19740014451)
    rad_coeff : float
        Coefficient on radiation, by default 5.39e-10 from Equation 4-56
        in Keller et al. (https://ntrs.nasa.gov/citations/19740014451)
    emittance : float
        Emittance of the MLI, by default 0.031 from Equation 4-56 in
        Keller et al. (https://ntrs.nasa.gov/citations/19740014451)
    vacuum_pressure : float
        Pressure of gas in vacuum region in torr, by default 1e-6
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("layer_density", default=30, desc="MLI layers per cm")
        self.options.declare("solid_cond_coeff", default=8.95e-8, desc="Correlation coefficient on solid conduction")
        self.options.declare("gas_cond_coeff", default=1.46e4, desc="Correlation coefficient on gas conduction")
        self.options.declare("rad_coeff", default=5.39e-10, desc="Correlation coefficient on radiation")
        self.options.declare("emittance", default=0.031, desc="Emittance of MLI")
        self.options.declare("vacuum_pressure", default=1e-6, desc="Vacuum pressure in torr")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("T_hot", shape=(nn,), val=300, units="K")
        self.add_input("T_cold", shape=(nn,), val=20, units="K")
        self.add_input("N_layers", val=20, units=None)
        self.add_output("heat_flux", shape=(nn,), val=1.0, units="W/m**2")

        arng = np.arange(nn)
        self.declare_partials("heat_flux", ["T_hot", "T_cold"], rows=arng, cols=arng)
        self.declare_partials("heat_flux", "N_layers", rows=arng, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        T_h = inputs["T_hot"]
        T_c = inputs["T_cold"]
        N_layers = inputs["N_layers"]
        layer_density = self.options["layer_density"]
        C_s = self.options["solid_cond_coeff"]
        C_g = self.options["gas_cond_coeff"]
        C_r = self.options["rad_coeff"]
        emittance = self.options["emittance"]
        P = self.options["vacuum_pressure"]

        # Heat flux due to radiation
        q_rad = C_r * emittance / N_layers * (T_h**4.67 - T_c**4.67)

        # Heat flux due to solid conduction through the MLI material
        q_solid_cond = C_s * layer_density**2.56 / N_layers * (T_h + T_c) / 2 * (T_h - T_c)

        # Heat flux due to gas conduction through the small amount of gas left in the vacuum
        q_gas_cond = C_g * P / N_layers * (T_h**0.52 - T_c**0.52)

        outputs["heat_flux"] = q_rad + q_solid_cond + q_gas_cond

    def compute_partials(self, inputs, J):
        T_h = inputs["T_hot"]
        T_c = inputs["T_cold"]
        N_layers = inputs["N_layers"]
        layer_density = self.options["layer_density"]
        C_s = self.options["solid_cond_coeff"]
        C_g = self.options["gas_cond_coeff"]
        C_r = self.options["rad_coeff"]
        emittance = self.options["emittance"]
        P = self.options["vacuum_pressure"]

        J["heat_flux", "T_hot"] = (
            4.67 * C_r * emittance / N_layers * T_h**3.67
            + 2 * C_s * layer_density**2.56 / N_layers * T_h / 2
            + 0.52 * C_g * P / N_layers * T_h ** (-0.48)
        )
        J["heat_flux", "T_cold"] = (
            -4.67 * C_r * emittance / N_layers * T_c**3.67
            - 2 * C_s * layer_density**2.56 / N_layers * T_c / 2
            - 0.52 * C_g * P / N_layers * T_c ** (-0.48)
        )
        J["heat_flux", "N_layers"] = (
            -1
            / N_layers**2
            * (
                C_r * emittance * (T_h**4.67 - T_c**4.67)
                + C_s * layer_density**2.56 * (T_h + T_c) / 2 * (T_h - T_c)
                + C_g * P * (T_h**0.52 - T_c**0.52)
            )
        )


if __name__ == "__main__":
    p = om.Problem()
    p.model.add_subsystem("model", HeatTransferVacuumTank(num_nodes=1), promotes=["*"])

    p.setup(force_alloc_complex=True)

    r = 2.0
    L = 2.0
    SA = 4 * np.pi * r**2 + 2 * np.pi * r * L
    wet_frac = 0.9
    p.set_val("A_wet", SA * wet_frac, units="m**2")
    p.set_val("A_dry", SA * (1 - wet_frac), units="m**2")
    p.set_val("T_env", 300, units="K")
    p.set_val("T_liq", 21, units="K")
    p.set_val("T_gas", 25, units="K")

    p.run_model()

    om.n2(p)

    p.model.list_inputs(units=True, print_arrays=True)
    p.model.list_outputs(units=True, print_arrays=True)
