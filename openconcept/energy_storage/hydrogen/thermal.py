import numpy as np
import openmdao.api as om
from openconcept.utilities.constants import STEFAN_BOLTZMANN_CONST, GRAV_CONST
from openconcept.energy_storage.hydrogen.H2_properties import lh2_cp, lh2_rho, sat_gh2_k


class HeatTransferVacuumTank(om.Group):
    """
    Computes the heat transfered from the environment to the propellant. This model
    assumes the tank is a vacuum-insulated tank with MLI insulation.

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
    h_liq : float
        Height of the liquid in the tank (vector, m)
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)

    Outputs
    -------
    Q : float
        Heat entering the propellant from the external environment (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")

    def setup(self):
        nn = self.options["num_nodes"]

        # Compute the heat flux through the MLI and from the inner wall to the propellant
        self.add_subsystem("mli_heat", MLIHeatFlux(num_nodes=nn), promotes_inputs=[("T_hot", "T_env"), "N_layers"])
        self.add_subsystem(
            "propellant_heat",
            InnerWallToHydrogenHeatFlux(num_nodes=nn),
            promotes_inputs=["T_liq", "T_gas", "A_wet", "A_dry", "h_liq", "radius"],
        )

        # Determine the inner wall temperature such that the heat flux through the MLI and
        # to the propellant are the same (assume steady state heat transfer)
        self.add_subsystem(
            "wall_temp_balance",
            om.BalanceComp(
                name="T_wall",
                eq_units="W/m**2",
                units="K",
                lower=10,
                val=30,
                lhs_name="q_mli",
                rhs_name="q_prop",
                shape=(nn,),
            ),
        )
        self.connect("mli_heat.heat_flux", "wall_temp_balance.q_mli")
        self.connect("propellant_heat.heat_flux", "wall_temp_balance.q_prop")
        self.connect("wall_temp_balance.T_wall", ["mli_heat.T_cold", "propellant_heat.T_wall"])

        # Given the heat flux, compute the total heat by multiplying by area
        self.add_subsystem(
            "total_heat",
            TotalHeatGivenHeatFlux(num_nodes=nn),
            promotes_inputs=["radius", "length"],
            promotes_outputs=["Q"],
        )
        self.connect("propellant_heat.heat_flux", "total_heat.heat_flux")


class ExternalThermalResistance(om.ExplicitComponent):
    """
    This component computes the effective thermal resistance of the external forced convection
    and radiation around the surface of the tank. This somewhat implicitly assumes that the surface
    of the tank is exposed to the freestream air. This may not be the case in all configurations.

    See section 3.4.3.1 of Dries Verstraete's thesis (http://hdl.handle.net/1826/4089) for more details.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `.                    ,'
         ` -------------- '

    Inputs
    ------
    outer_radius : float
        Radius of the outer surface of the tank's cylindrical portion and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    T_surface : float
        Temperature of the outer skin of the tank (vector, K)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|rho : float
        Density at flight condition (vector, kg/m^3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|k : float
        Thermal conductivity at flight condition (vector, W/(m-K))
    fltcond|mu : float
        Dynamic viscosity at flight condition (vector, N-s/m^2)

    Outputs
    -------
    resistance : float
        Thermal resistance for the external heat transfer (float, K/W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    skin_emissivity : float
        Radiative emissivity of the skin of the tank/aircraft. Verstraete assume an integral
        tank and take emissivity to be 0.95 when the aircraft skin is painted white and 0.09
        if the aircraft is unpainted, by default 0.9 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("skin_emissivity", default=0.9, desc="Tank skin emissivity")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("outer_radius", units="m")
        self.add_input("length", units="m")
        self.add_input("T_surface", shape=(nn,), units="K")
        self.add_input("fltcond|Utrue", shape=(nn,), units="m/s")
        self.add_input("fltcond|rho", shape=(nn,), units="kg / m**3")
        self.add_input("fltcond|T", shape=(nn,), units="K")
        self.add_input("fltcond|k", shape=(nn,), units="W / (m * K)")
        self.add_input("fltcond|mu", shape=(nn,), units="N * s / m**2")

        self.add_output("resistance", shape=(nn,), units="K/W")

        # TODO: analytic partials
        arng = np.arange(nn)
        self.declare_partials("resistance", ["outer_radius", "length"], rows=arng, cols=np.zeros(nn), method="cs")
        self.declare_partials(
            "resistance",
            ["T_surface", "fltcond|Utrue", "fltcond|rho", "fltcond|T", "fltcond|k", "fltcond|mu"],
            rows=arng,
            cols=arng,
            method="cs",
        )

    def compute(self, inputs, outputs):
        r = inputs["outer_radius"]
        L = inputs["length"]
        T_surface = inputs["T_surface"]
        v_air = inputs["fltcond|Utrue"]
        rho_air = inputs["fltcond|rho"]
        T_air = inputs["fltcond|T"]
        k_air = inputs["fltcond|k"]
        mu_air = inputs["fltcond|mu"]
        cp_air = 1.005e3  # J/(kg-K), specific heat at constant pressure of air
        emissivity = self.options["skin_emissivity"]

        # We take characteristic length to include half the length of each hemispherical end cap
        L_char = r + L

        # =================================== Convective heat transfer portion ===================================
        # Prandtl and Reynolds numbers
        Pr = mu_air * cp_air / k_air
        Re_L = rho_air * v_air * L_char / mu_air

        # Nusselt number uses the same relation as Verstraete (see Equation 3.12 in thesis for more details)
        Nu_L = 0.03625 * Pr**0.43 * Re_L**0.8

        # Convective heat transfer coefficient
        h_conv = Nu_L * k_air / L_char

        # =================================== Radiative heat transfer portion ===================================
        h_rad = STEFAN_BOLTZMANN_CONST * emissivity * (T_surface**2 + T_air**2) * (T_surface + T_air)

        # ====================================== Total thermal resistance ======================================
        h_tot = h_conv + h_rad
        outputs["resistance"] = 1 / (2 * np.pi * r * L_char * h_tot)


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
        self.add_output("heat_flux", shape=(nn,), units="W/m**2")

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


class InnerWallToHydrogenHeatFlux(om.ExplicitComponent):
    """
    Compute the heat flux from the tank inner wall into the gaseous and liquid hydrogen propellant.
    See section 3.4.3.3 of Dries Verstraete's thesis (http://hdl.handle.net/1826/4089) for more details.

    Inputs
    ------
    T_wall : float
        Temperature of the inner tank wall (vector, K)
    T_liq : float
        Temperature of the liquid hydrogen (vector, K)
    T_gas : float
        Temperature of the gaseous hydrogen (vector, K)
    A_wet : float
        The area of the tank's surface touching the bulk liquid (vector, m^2)
    A_dry : float
        The area of the tank's surface touching the ullage (vector, m^2)
    h_liq : float
        Height of the liquid in the tank (vector, m)
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m)

    Outputs
    -------
    heat_flux : float
        Heat flux into the propellant (vector, W/m^2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
        Pressure of gas in vacuum region in torr, by default 1e-6
    lh2_conductivity : float
        Conductivity of the liquid hydrogen, by default 0.104 esimated from NIST WebBook
        around 1-2 bar and 20 K (scalar, W/(m-K))
    lh2_expand_coeff : float
        Thermal expansion coefficient of liquid hydrogen, by default 18e-3 roughly
        extrapolated from Shwalbe and Grilly 1977 (https://pubmed.ncbi.nlm.nih.gov/34566132/)
        (scalar, 1/K)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("lh2_conductivity", default=0.104, desc="LH2 thermal conductivity in W/(m-K)")
        self.options.declare("lh2_expand_coeff", default=18e-3, desc="LH2 thermal expansion coefficient in 1/K")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("T_wall", shape=(nn,), val=30, units="K")
        self.add_input("T_liq", shape=(nn,), val=20, units="K")
        self.add_input("T_gas", shape=(nn,), val=25, units="K")
        self.add_input("A_wet", shape=(nn,), val=5.0, units="m**2")
        self.add_input("A_dry", shape=(nn,), val=5.0, units="m**2")
        self.add_input("h_liq", shape=(nn,), units="m")
        self.add_input("radius", units="m")

        self.add_output("heat_flux", shape=(nn,), units="W/m**2")

        # TODO: add analytic partials
        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        T_w = inputs["T_wall"]
        T_l = inputs["T_liq"]
        T_g = inputs["T_gas"]
        A_l = inputs["A_wet"]
        A_g = inputs["A_dry"]
        h_l = inputs["h_liq"]
        r = inputs["radius"]
        k = self.options["lh2_conductivity"]
        beta = self.options["lh2_expand_coeff"]

        # Heat transfer coefficient for liquid phase natural convection
        Ra = GRAV_CONST * beta * (T_w - T_l) * h_l**3 * lh2_rho(T_l) * lh2_cp(T_l) / k
        Nu_liq = 0.0605 * Ra ** (1 / 3)
        h_coeff_liq = k * Nu_liq / h_l

        # Heat transfer coefficient for gaseous phase convection (use saturated gas thermal conductivity)
        Nu_gas = 17.0
        h_coeff_gas = sat_gh2_k(T_g) * Nu_gas / (2 * r - h_l)

        # Scale the heat transfer to liquid and gas by the area touching liquid and gas inside the tank
        outputs["heat_flux"] = (h_coeff_liq * A_l * (T_w - T_l) + h_coeff_gas * A_g * (T_w - T_g)) / (A_l + A_g)


class TotalHeatGivenHeatFlux(om.ExplicitComponent):
    """
    Compute the total heat from the heat flux by multiplying the heat flux by the tank surface area.

    Inputs
    -----
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    heat_flux : float
        Heat flux through the insulation and into the propellant (vector, W/m^2)

    Outputs
    -------
    Q : float
        Heat entering the propellant from the external environment (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("radius", units="m")
        self.add_input("length", units="m")
        self.add_input("heat_flux", shape=(nn,), units="W/m**2")

        self.add_output("Q", shape=(nn,), units="W")

        arng = np.arange(nn)
        self.declare_partials("Q", ["radius", "length"], rows=arng, cols=np.zeros(nn))
        self.declare_partials("Q", "heat_flux", rows=arng, cols=arng)

    def compute(self, inputs, outputs):
        r = inputs["radius"]
        L = inputs["length"]
        q = inputs["heat_flux"]

        # Compute the surface area
        A = 4 * np.pi * r**2 + 2 * np.pi * r * L

        # Heat is surface area times heat flux
        outputs["Q"] = q * A

    def compute_partials(self, inputs, J):
        r = inputs["radius"]
        L = inputs["length"]
        q = inputs["heat_flux"]

        # Compute the surface area
        A = 4 * np.pi * r**2 + 2 * np.pi * r * L

        J["Q", "radius"] = q * (8 * np.pi * r + 2 * np.pi * L)
        J["Q", "length"] = q * 2 * np.pi * r
        J["Q", "heat_flux"] = A


if __name__ == "__main__":
    p = om.Problem()
    p.model.add_subsystem("model", HeatTransferVacuumTank(num_nodes=5), promotes=["*"])
    p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2)
    p.model.linear_solver = om.DirectSolver()

    p.setup()

    p.run_model()

    p.model.list_inputs(units=True)
    p.model.list_outputs(units=True)
