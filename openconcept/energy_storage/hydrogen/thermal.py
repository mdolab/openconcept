import numpy as np
import openmdao.api as om
from openconcept.utilities.constants import STEFAN_BOLTZMANN_CONST

# ===============================================================================================
#      Below are the various thermal resistance computations for the thermal circuit model
# ===============================================================================================

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
        self.declare_partials("resistance", ["T_surface", "fltcond|Utrue", "fltcond|rho", "fltcond|T", "fltcond|k", "fltcond|mu"], rows=arng, cols=arng, method="cs")

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