from __future__ import division
import numpy as np
import openmdao.api as om
from openconcept.utilities.constants import GRAV_CONST, UNIVERSAL_GAS_CONST, MOLEC_WEIGHT_H2
import openconcept.energy_storage.hydrogen.H2_properties as H2_prop


class SimpleBoilOff(om.ExplicitComponent):
    """
    Simplest possible model for boil-off. Boil-off
    mass flow rate equals Q/h where Q is heat entering
    liquid and h is latent heat of vaporization.

    Inputs
    ------
    heat_into_liquid : float
        Heat entering liquid propellant (vector, W)
    LH2_heat_added : float
        Additional heat intentionally added to liquid (vector, W)

    Outputs
    -------
    m_boil_off : float
        Mass flow rate of boil-off (vector, kg/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    h_vap : float
        Latent heat of vaporization of propellant, default hydrogen 446592 J/kg (scalar, J/kg)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("h_vap", default=446592.0, desc="Latent heat of vaporization (J/kg)")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("heat_into_liquid", val=100.0, units="W", shape=(nn,))
        self.add_input("LH2_heat_added", val=0.0, units="W", shape=(nn,))
        self.add_output("m_boil_off", val=0.1, units="kg/s", shape=(nn,), lower=0.0)
        self.declare_partials(
            "m_boil_off",
            ["heat_into_liquid", "LH2_heat_added"],
            val=np.ones(nn) / self.options["h_vap"],
            rows=np.arange(nn),
            cols=np.arange(nn),
        )

    def compute(self, inputs, outputs):
        outputs["m_boil_off"] = (inputs["LH2_heat_added"] + inputs["heat_into_liquid"]) / self.options["h_vap"]


class LH2BoilOffODE(om.ExplicitComponent):
    """
    This portion of the code leans on much of the work from Eugina Mendez Ramos's thesis.
    """

    def setup(self):
        # Compute the maximum allowable temperature of the liquid.
        # The maximum allowable temperature of the liquid is the saturation temperature
        # at the minimum pressure. If it is at this temperature, don't let it increase further.
        self.T_liq_max = H2_prop.sat_gh2_T(self.options["P_min"])

    def compute(self, inputs, outputs):
        # Unpack the states from the inputs
        m_gas = inputs["m_gas"]
        m_liq = inputs["m_liq"]
        T_gas = inputs["T_gas"]
        T_liq = inputs["T_liq"]
        V_gas = inputs["V_gas"]

        m_dot_gas_in = 0.0  # external input to ullage (excluding boil off), don't think this would ever be nonzero
        m_dot_gas_out = inputs[
            "m_dot_gas_out"
        ]  # with venting this will be nonzero (is this still valid from a physics standpoint given her assumptions?)
        m_dot_liq_in = 0.0  # not sure when this would ever by nonzero, but could keep in for generality
        m_dot_liq_out = inputs["m_dot_liq_out"]  # liquid leaving the tank (e.g., for fuel to the engines)

        # ============================== Compute geometric quantities ==============================
        A_int = None  # area of the surface of the bulk liquid (the interface)
        L_int = None  # characteristic length of the interface

        # =============================== Compute physical properties ===============================
        # Ullage gas properties
        h_gas = H2_prop.gh2_h(P_gas, T_gas)  # enthalpy
        u_gas = H2_prop.gh2_u(P_gas, T_gas)  # internal energy
        cv_gas = H2_prop.gh2_cv(P_gas, T_gas)  # specific heat at constant volume
        cp_gas = H2_prop.gh2_cp(P_gas, T_gas)  # TODO: this may be unnecesary; specific heat at constant pressure

        # Use ideal gas law to compute ullage pressure (real gas properties are used elsewhere)
        P_gas = m_gas * T_gas * UNIVERSAL_GAS_CONST / (V_gas * MOLEC_WEIGHT_H2)

        # Bulk liquid properties
        h_liq = H2_prop.lh2_h(T_liq)  # enthalpy
        u_liq = H2_prop.lh2_u(T_liq)  # internal energy
        cp_liq = H2_prop.lh2_cp(T_liq)  # specific heat at constant pressure
        rho_liq = H2_prop.lh2_rho(T_liq)  # density
        P_liq = H2_prop.lh2_P(T_liq)  # pressure

        # Temperature of the interface assumes saturated hydrogen with same pressure as the ullage
        T_int = H2_prop.sat_gh2_T(P_gas)  # use saturated GH2 temperature
        h_int = H2_prop.lh2_h(T_int)  # use saturated LH2 enthalpy

        # Saturated gas properties at the mean film temperature
        T_mean_film = 0.5 * (T_gas + T_int)
        cp_sat_gas = H2_prop.sat_gh2_cp(T_mean_film)  # specific heat at constant pressure
        visc_sat_gas = H2_prop.sat_gh2_viscosity(T_mean_film)  # viscosity
        k_sat_gas = H2_prop.sat_gh2_k(T_mean_film)  # thermal conductivity
        beta_sat_gas = H2_prop.sat_gh2_beta(T_mean_film)  # coefficient of thermal expansion
        rho_sat_gas = H2_prop.sat_gh2_rho(T_mean_film)  # density
        h_sat_gas = H2_prop.sat_gh2_h(T_mean_film)  # enthalpy

        # ==================== Compute heat transfer between ullage and interface ====================
        # Some of the heat from the environment goes to heating the walls, make sure you're accounting for this.
        # Turns out though that this may be small (only a few percent), so we'll ignore it (see Van Dresar paper).
        # TODO: decide whether to use the area ratio approach to figure out how heat is distributed between the
        # ullage and bulk liquid (like EBM model) or use current approach that takes empirical data from Van Dresar.
        Q_dot_env_gas = inputs["Q_dot_env_gas"]
        Q_dot_env_liq = inputs["Q_dot_env_liq"]
        Q_dot_gas_int = heat_transfer_coeff_gas_int * A_int * (T_gas - T_int)

        # Compute the heat transfer coefficient for the heat transfer from the ullage to the interface.
        # Evaluate the heat transfer coefficient between the ullage and interface using the saturated gas
        # properties at the mean film temperature (average of ullage and interface temps) because the ullage
        # near the interface is close to saturated due to thermal stratification effects.
        # Use constants associated with the top of a cold horizontal surface
        C = 0.27
        n = 0.25

        # Compute the fluid properties for heat transfer
        prandtl = cp_sat_gas * visc_sat_gas / k_sat_gas
        grashof = GRAV_CONST * beta_sat_gas * rho_sat_gas**2 * np.abs(T_gas - T_int) * L_int**3 / visc_sat_gas**2
        nusselt = C * (prandtl * grashof) ** n
        heat_transfer_coeff_gas_int = k_sat_gas / L_int * nusselt

        # ============================================ ODEs ============================================
        # Compute the boil off mass flow rate
        # TODO: these two should be the same I think? Check.
        m_dot_boil_off = Q_dot_gas_int / (h_gas - h_liq)
        m_dot_boil_off = Q_dot_gas_int / (cp_liq * (T_int - T_liq) + (h_gas - h_int) + (h_gas - h_sat_gas))

        # Mass flows
        m_dot_gas = m_dot_boil_off + m_dot_gas_in - m_dot_gas_out
        m_dot_liq = m_dot_liq_in - m_dot_boil_off - m_dot_liq_out

        V_dot_liq = m_dot_liq / rho_liq
        V_dot_gas = -V_dot_liq

        # TODO: is there a reason she differentiates between mdot_v and mdot_l/g in these equations? (See eqns 4.17 and 4.25 in the thesis)
        T_dot_gas = (Q_dot_env_gas - Q_dot_gas_int - P_gas * V_dot_gas + m_dot_gas * (h_gas - u_gas)) / (m_gas * cv_gas)
        T_dot_liq = (Q_dot_env_liq + P_liq * V_dot_liq - m_dot_liq * (h_liq + u_liq)) / (m_liq * cp_liq)
        m_dot_liq = -m_dot_gas

        # The maximum allowable temperature of the liquid is the saturation temperature
        # at the minimum pressure. If it is at this temperature, don't let it increase further.
        T_dot_liq[T_liq >= self.T_liq_max] *= 0.0

        # We got em!
        outputs["m_dot_gas"] = m_dot_gas
        outputs["m_dot_liq"] = m_dot_liq
        outputs["T_dot_gas"] = T_dot_gas
        outputs["T_dot_liq"] = T_dot_liq
        outputs["V_dot_gas"] = V_dot_gas
