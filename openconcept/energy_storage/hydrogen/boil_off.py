from __future__ import division
import numpy as np
import openmdao.api as om
from openconcept.utilities.constants import GRAV_CONST, UNIVERSAL_GAS_CONST, MOLEC_WEIGHT_H2
import openconcept.energy_storage.hydrogen.H2_properties as H2_prop
from openconcept.utilities import Integrator


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


class BoilOff(om.Group):
    """
    Time-integrated properties of the ullage and bulk liquid due to heat
    and mass flows into, out of, and within the liquid hydrogen tank.
    The model used is heavily based on work in Eugina Mendez Ramos's thesis
    (http://hdl.handle.net/1853/64797). See Chapter 4 and Appendix E
    for more relevant details.

    Due to geometric computations, this model can get tempermental when the
    tank is nearly empty or nearly full. If used in an optimization problem,
    it may help to constrain the tank fill level to be greater than 1% or so.

    Inputs
    ------
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    m_dot_gas_in : float
        Mass flow rate of gaseous hydrogen into the ullage EXCLUDING any boil off (this is
        handled internally in this component); unlikely to ever be nonzero but left here
        to maintain generality (vector, kg/s)
    m_dot_gas_out : float
        Mass flow rate of gaseous hydrogen out of the ullage; could be for venting
        or gaseous hydrogen consumption (vector, kg/s)
    m_dot_liq_in : float
        Mass flow rate of liquid hydrogen into the bulk liquid; unlikely to ever be nonzero
        but left here to maintain generality (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid hydrogen out of the tank; this is where fuel being consumed
        is bookkept, assuming it is removed from the tank as a liquid (vector, kg/s)
    Q_dot : float
        Total heat flow rate into tank (vector, W)

    Outputs
    -------
    m_gas : float
        Mass of the gaseous hydrogen in the tank ullage (vector, kg)
    m_liq : float
        Mass of liquid hydrogen in the tank (vector, kg)
    T_gas : float
        Temperature of the gaseous hydrogen in the ullage (vector, K)
    T_liq : float
        Temperature of the bulk liquid hydrogen (vector, K)
    P_gas : float
        Pressure of the gas in the ullage (vector, Pa)
    fill_level : float
        Fraction of tank volume filled with liquid (vector)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    init_fill_level : float
        Initial fill level (in range 0-1) of the tank, default 0.97
        to leave space for boil off gas; 3% adopted from Cryoplane study (scalar, dimensionless)
    ullage_T_init : float
        Initial temperature of gas in ullage, default 21 K (scalar, K)
    ullage_P_init : float
        Initial pressure of gas in ullage, default 120,000 Pa; ullage pressure must be higher than ambient
        to prevent air leaking in and creating a combustible mixture (scalar, Pa)
    liquid_T_init : float
        Initial temperature of bulk liquid hydrogen, default 20 K (scalar, K)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("init_fill_level", default=0.97, desc="Initial fill level")
        self.options.declare("ullage_T_init", default=21.0, desc="Initial ullage temp (K)")
        self.options.declare("ullage_P_init", default=1.2e5, desc="Initial ullage pressure (Pa)")
        self.options.declare("liquid_T_init", default=20.0, desc="Initial bulk liquid temp (K)")

    def setup(self):
        nn = self.options["num_nodes"]

        # Compute the fill level in the tank
        self.add_subsystem(
            "level_calc",
            BoilOffFillLevelCalc(num_nodes=nn),
            promotes_inputs=["radius", "length"],
            promotes_outputs=["fill_level"],
        )

        # Compute the required geometric properties
        self.add_subsystem("liq_height_calc", LiquidHeight(num_nodes=nn), promotes_inputs=["radius", "length"])
        self.add_subsystem("interface_params", BoilOffGeometry(num_nodes=nn), promotes_inputs=["radius", "length"])
        self.connect("fill_level", "liq_height_calc.fill_level")
        self.connect("liq_height_calc.h_liq", "interface_params.h_liq")

        # Compute the ODE equations to be integrated
        self.add_subsystem(
            "boil_off_ode",
            LH2BoilOffODE(num_nodes=nn),
            promotes_inputs=[
                "m_dot_gas_in",
                "m_dot_gas_out",
                "m_dot_liq_in",
                "m_dot_liq_out",
                "Q_dot",
            ],
            promotes_outputs=["P_gas"],
        )
        self.connect("interface_params.A_interface", "boil_off_ode.A_interface")
        self.connect("interface_params.L_interface", "boil_off_ode.L_interface")
        self.connect("interface_params.A_wet", "boil_off_ode.A_wet")
        self.connect("interface_params.A_dry", "boil_off_ode.A_dry")

        # Integrate the ODE
        integ = self.add_subsystem(
            "integ",
            Integrator(num_nodes=nn, diff_units="s", time_setup="duration", method="simpson"),
            promotes_outputs=["m_gas", "m_liq", "T_gas", "T_liq"],
        )
        integ.add_integrand("m_gas", rate_name="m_dot_gas", units="kg", lower=1e-4)
        integ.add_integrand("m_liq", rate_name="m_dot_liq", units="kg", lower=1e-3)
        integ.add_integrand("T_gas", rate_name="T_dot_gas", units="K", lower=15, upper=1e3)
        integ.add_integrand("T_liq", rate_name="T_dot_liq", units="K", lower=1e-3, upper=30)
        integ.add_integrand("V_gas", rate_name="V_dot_gas", units="m**3", lower=1e-3)

        # Connect the ODE to the integrator
        self.connect("boil_off_ode.m_dot_gas", "integ.m_dot_gas")
        self.connect("boil_off_ode.m_dot_liq", "integ.m_dot_liq")
        self.connect("boil_off_ode.T_dot_gas", "integ.T_dot_gas")
        self.connect("boil_off_ode.T_dot_liq", "integ.T_dot_liq")
        self.connect("boil_off_ode.V_dot_gas", "integ.V_dot_gas")

        self.connect("m_gas", "boil_off_ode.m_gas")
        self.connect("m_liq", "boil_off_ode.m_liq")
        self.connect("T_gas", "boil_off_ode.T_gas")
        self.connect("T_liq", "boil_off_ode.T_liq")
        self.connect("integ.V_gas", ["boil_off_ode.V_gas", "level_calc.V_gas"])

        # Set defaults for inputs promoted from multiple sources
        self.set_input_defaults("radius", 2.0, units="m")
        self.set_input_defaults("length", 0.5, units="m")

        # Set a solver specifically for this component in an attempt to increase robustness
        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options["solve_subsystems"] = False
        self.nonlinear_solver.options["maxiter"] = 100
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(alpha=1.0, iprint=2, print_bound_enforce=False)

    def guess_nonlinear(self, inputs, outputs, _):
        """
        Set both the guesses and the initial values at the beginning of the phase
        for the integrated states. The initial state values should get overwritten
        if this component lives in an intermediate phase and OpenConcept has linked
        the states from a previous phase to this one.
        """
        r = inputs["level_calc.radius"]
        L = inputs["level_calc.length"]
        fill_init = self.options["init_fill_level"]
        T_gas_init = self.options["ullage_T_init"]
        P_gas_init = self.options["ullage_P_init"]
        T_liq_init = self.options["liquid_T_init"]

        # Compute the initial gas mass from the given initial pressure
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        V_gas = V_tank * (1 - fill_init)
        m_gas = P_gas_init / T_gas_init / UNIVERSAL_GAS_CONST * V_gas * MOLEC_WEIGHT_H2
        m_liq = (V_tank - V_gas) * H2_prop.lh2_rho(T_liq_init)

        # Initialize the states for the solver
        outputs["m_gas"] = m_gas
        outputs["m_liq"] = m_liq
        outputs["T_gas"] = T_gas_init
        outputs["T_liq"] = T_liq_init
        outputs["integ.V_gas"] = V_gas

        # Set the initial values of the states for integration. Doing this here allows the values
        # to be set properly while enabling them to be overwritten if this component lives within
        # an intermediate phase and OpenConcept has connected the final states from the previous
        # phase the the initial state of this one.
        inputs["integ.m_gas_initial"] = m_gas
        inputs["integ.m_liq_initial"] = m_liq
        inputs["integ.T_gas_initial"] = T_gas_init
        inputs["integ.T_liq_initial"] = T_liq_init
        inputs["integ.V_gas_initial"] = V_gas


class LiquidHeight(om.ImplicitComponent):
    """
    Implicitly compute the height of liquid in the tank.

    Inputs
    ------
    fill_level : float
        Fraction of tank volume filled with liquid (vector)
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)

    Outputs
    -------
    h_liq : float
        Height of the liquid in the tank (vector, m)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("fill_level", shape=(nn,))
        self.add_input("radius", units="m")
        self.add_input("length", units="m")

        self.add_output("h_liq", val=1e-3, units="m", shape=(nn,))

        arng = np.arange(nn)
        self.declare_partials("h_liq", ["radius", "length"], rows=arng, cols=np.zeros(nn))
        self.declare_partials("h_liq", ["h_liq", "fill_level"], rows=arng, cols=arng)

    def apply_nonlinear(self, inputs, outputs, residuals):
        fill = inputs["fill_level"]
        r = inputs["radius"]
        L = inputs["length"]
        h = outputs["h_liq"]

        # For the current guess of the liquid height, compute the
        # volume of fluid in the hemispherical and cylindrical
        # portions of the tank
        V_sph = np.pi * h**2 / 3 * (3 * r - h)

        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment
        V_cyl = r**2 / 2 * (th - np.sin(th)) * L

        # Total tank volume
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L

        # Residual is difference between liquid volume given current
        # height guess and actual liquid volume computed with fill level
        residuals["h_liq"] = V_sph + V_cyl - V_tank * fill

    def linearize(self, inputs, outputs, J):
        fill = inputs["fill_level"]
        r = inputs["radius"]
        L = inputs["length"]
        h = outputs["h_liq"]

        # Compute partials of spherical volume w.r.t. inputs and height
        Vsph_r = np.pi * h**2
        Vsph_h = 2 * np.pi * h * r - np.pi * h**2

        # Compute partials of cylindrical volume w.r.t. inputs and height
        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment
        th_r = -2 / np.sqrt(1 - (1 - h / r) ** 2) * h / r**2
        th_h = 2 / np.sqrt(1 - (1 - h / r) ** 2) / r

        Vcyl_r = (
            r * (th - np.sin(th)) * L
            + r**2 / 2 * (1 - np.cos(th)) * L * th_r  # pV_cyl / pr  # pV_cyl / pth * pth / pr
        )
        Vcyl_h = r**2 / 2 * (1 - np.cos(th)) * L * th_h
        Vcyl_L = r**2 / 2 * (th - np.sin(th))

        # Total tank volume
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        Vtank_r = 4 * np.pi * r**2 + 2 * np.pi * r * L
        Vtank_L = np.pi * r**2

        J["h_liq", "radius"] = Vsph_r + Vcyl_r - Vtank_r * fill
        J["h_liq", "length"] = Vcyl_L - Vtank_L * fill
        J["h_liq", "fill_level"] = -V_tank
        J["h_liq", "h_liq"] = Vsph_h + Vcyl_h

    def guess_nonlinear(self, inputs, outputs, residuals):
        # Guess the height initially using a linear approximation of height w.r.t. fill level
        outputs["h_liq"] = 2 * inputs["fill_level"] * inputs["radius"]


class BoilOffGeometry(om.ExplicitComponent):
    """
    Compute areas and volumes in the tank from fill level.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `. ~~~~~~~~~~~~~~~~~~ ,'      -.- h_liq
         ` -------------- '         -'-

    Inputs
    ------
    h_liq : float
        Height of the liquid in the tank (vector, m)
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)

    Outputs
    -------
    A_interface : float
        Area of the surface of the liquid in the tank. This is the area of
        the interface between the ullage and bulk liquid portions, hence
        the name (vector, m^2)
    L_interface : float
        Characteristic length of the interface between the ullage and the
        bulk liquid (vector, m)
    A_wet : float
        The area of the tank's surface touching the bulk liquid (vector, m^2)
    A_dry : float
        The area of the tank's surface touching the ullage (vector, m^2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("h_liq", units="m", shape=(nn,))
        self.add_input("radius", units="m")
        self.add_input("length", units="m")

        self.add_output("A_interface", units="m**2", shape=(nn,), lower=0.0, val=3.0)
        self.add_output("L_interface", units="m", shape=(nn,), lower=0.0, val=1.0)
        self.add_output("A_wet", units="m**2", shape=(nn,), lower=0.0, val=5.0)
        self.add_output("A_dry", units="m**2", shape=(nn,), lower=0.0, val=5.0)

        arng = np.arange(nn)
        self.declare_partials(["*"], "h_liq", rows=arng, cols=arng)
        self.declare_partials(["*"], ["radius", "length"], rows=arng, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        h = inputs["h_liq"]
        r = inputs["radius"]
        L = inputs["length"]

        # Total area of the tank
        A_tank = 4 * np.pi * r**2 + 2 * np.pi * r * L

        # Some useful geometric parameters
        c = 2 * np.sqrt(2 * r * h - h**2)  # chord length of circular segment
        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment

        # Interface area
        outputs["A_interface"] = np.pi * (c / 2) ** 2 + c * L
        outputs["L_interface"] = c  # take the chord as the characteristic length

        # Wet and dry areas
        outputs["A_wet"] = 2 * np.pi * r * h + th * r * L
        outputs["A_dry"] = A_tank - outputs["A_wet"]

    def compute_partials(self, inputs, J):
        h = inputs["h_liq"]
        r = inputs["radius"]
        L = inputs["length"]

        # Derivatives of chord and central angle of segment w.r.t. height and radius
        c = 2 * np.sqrt(2 * r * h - h**2)  # chord length of circular segment
        c_r = 2 * h / np.sqrt(2 * r * h - h**2)
        c_h = (2 * r - 2 * h) / np.sqrt(2 * r * h - h**2)

        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment
        th_r = -2 / np.sqrt(1 - (1 - h / r) ** 2) * h / r**2
        th_h = 2 / np.sqrt(1 - (1 - h / r) ** 2) / r

        J["A_interface", "h_liq"] = c_h * (np.pi * c / 2 + L)
        J["A_interface", "radius"] = c_r * (np.pi * c / 2 + L)
        J["A_interface", "length"] = c

        J["L_interface", "h_liq"] = c_h
        J["L_interface", "radius"] = c_r
        J["L_interface", "length"] *= 0.0

        J["A_wet", "h_liq"] = 2 * np.pi * r + th_h * r * L
        J["A_wet", "radius"] = 2 * np.pi * h + th * L + th_r * r * L
        J["A_wet", "length"] = th * r

        J["A_dry", "h_liq"] = -J["A_wet", "h_liq"]
        J["A_dry", "radius"] = 8 * np.pi * r + 2 * np.pi * L - J["A_wet", "radius"]
        J["A_dry", "length"] = 2 * np.pi * r - J["A_wet", "length"]


class LH2BoilOffODE(om.ExplicitComponent):
    """
    Compute the derivatives of the state values for the liquid hydrogen boil off process
    given the current states values and other related inputs. The states are the mass of
    gaseous and liquid hydrogen, the temperature of the gas and liquid, and the volume
    of gas (volume of the ullage).

    This portion of the code leans on much of the work from Eugina Mendez Ramos's thesis
    (http://hdl.handle.net/1853/64797). See Chapter 4 and Appendix E for more relevant details.

    Inputs
    ------
    m_gas : float
        Mass of the gaseous hydrogen in the tank ullage (vector, kg)
    m_liq : float
        Mass of liquid hydrogen in the tank (vector, kg)
    T_gas : float
        Temperature of the gaseous hydrogen in the ullage (vector, K)
    T_liq : float
        Temperature of the bulk liquid hydrogen (vector, K)
    V_gas : float
        Volume of the ullage (vector, m^3)
    m_dot_gas_in : float
        Mass flow rate of gaseous hydrogen into the ullage EXCLUDING any boil off (this is
        handled internally in this component); unlikely to ever be nonzero but left here
        to maintain generality (vector, kg/s)
    m_dot_gas_out : float
        Mass flow rate of gaseous hydrogen out of the ullage; could be for venting
        or gaseous hydrogen consumption (vector, kg/s)
    m_dot_liq_in : float
        Mass flow rate of liquid hydrogen into the bulk liquid; unlikely to ever be nonzero
        but left here to maintain generality (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid hydrogen out of the tank; this is where fuel being consumed
        is bookkept, assuming it is removed from the tank as a liquid (vector, kg/s)
    Q_dot : float
        Total heat flow rate into tank (vector, W)
    A_interface : float
        Area of the surface of the liquid in the tank. This is the area of
        the interface between the ullage and bulk liquid portions, hence
        the name (vector, m^2)
    L_interface : float
        Characteristic length of the interface between the ullage and the
        bulk liquid (vector, m)
    A_wet : float
        The area of the tank's surface touching the bulk liquid (vector, m^2)
    A_dry : float
        The area of the tank's surface touching the ullage (vector, m^2)

    Outputs
    -------
    m_dot_gas : float
        Rate of change of ullage gas mass (vector, kg/s)
    m_dot_liq : float
        Rate of change of bulk liquid mass (vector, kg/s)
    T_dot_gas : float
        Rate of change of ullage gas temperature (vector, K/s)
    T_dot_liq : float
        Rate of change of bulk liquid temperature (vector, K/s)
    V_dot_gas : float
        Rate of change of ullage volume (vector, m^3/s)
    P_gas : float
        Pressure in the ullage (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    P_min : float
        Minimum operating pressure of the tank, by default 110,000 Pa; slightly above atmospheric (scalar, Pa)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("P_min", default=110e3, desc="Minimum operating pressure of the tank")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("m_gas", units="kg", shape=(nn,))
        self.add_input("m_liq", units="kg", shape=(nn,))
        self.add_input("T_gas", units="K", shape=(nn,))
        self.add_input("T_liq", units="K", shape=(nn,))
        self.add_input("V_gas", units="m**3", shape=(nn,))
        self.add_input("m_dot_gas_in", units="kg/s", shape=(nn,), val=0.0)
        self.add_input("m_dot_gas_out", units="kg/s", shape=(nn,), val=0.0)
        self.add_input("m_dot_liq_in", units="kg/s", shape=(nn,), val=0.0)
        self.add_input("m_dot_liq_out", units="kg/s", shape=(nn,), val=0.0)
        self.add_input("Q_dot", units="W", shape=(nn,), val=0.0)
        self.add_input("A_interface", units="m**2", shape=(nn,))
        self.add_input("L_interface", units="m", shape=(nn,))
        self.add_input("A_wet", units="m**2", shape=(nn,))
        self.add_input("A_dry", units="m**2", shape=(nn,))

        self.add_output("m_dot_gas", units="kg/s", shape=(nn,), val=0.0)
        self.add_output("m_dot_liq", units="kg/s", shape=(nn,), val=0.0)
        self.add_output("T_dot_gas", units="K/s", shape=(nn,), val=0.0)
        self.add_output("T_dot_liq", units="K/s", shape=(nn,), val=0.0)
        self.add_output("V_dot_gas", units="m**3/s", shape=(nn,), val=0.0)
        self.add_output("P_gas", units="Pa", shape=(nn,), val=1e5, lower=1e3)

        # TODO: analytic partials
        self.declare_partials(["*"], ["*"], method="cs")

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

        m_dot_gas_in = inputs["m_dot_gas_in"]  # external input to ullage excluding boil off, almost always zero
        m_dot_gas_out = inputs["m_dot_gas_out"]  # gas released for venting or consumption
        m_dot_liq_in = inputs["m_dot_liq_in"]  # not sure when this would ever by nonzero, but keep in for generality
        m_dot_liq_out = inputs["m_dot_liq_out"]  # liquid leaving the tank (e.g., for fuel to the engines)

        # ============================== Compute geometric quantities ==============================
        A_int = inputs["A_interface"]  # area of the surface of the bulk liquid (the interface)
        L_int = inputs["L_interface"]  # characteristic length of the interface
        A_wet = inputs["A_wet"]
        A_dry = inputs["A_dry"]

        # =============================== Compute physical properties ===============================
        # Use ideal gas law to compute ullage pressure (real gas properties are used elsewhere)
        P_gas = m_gas * T_gas * UNIVERSAL_GAS_CONST / (V_gas * MOLEC_WEIGHT_H2)

        # Ullage gas properties
        h_gas = H2_prop.gh2_h(P_gas, T_gas)  # enthalpy
        u_gas = H2_prop.gh2_u(P_gas, T_gas)  # internal energy
        cv_gas = H2_prop.gh2_cv(P_gas, T_gas)  # specific heat at constant volume

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
        prandtl[prandtl < 0] = 0.0
        grashof[grashof < 0] = 0.0
        nusselt = C * (prandtl * grashof) ** n
        heat_transfer_coeff_gas_int = k_sat_gas / L_int * nusselt

        # Heat from the environment that goes to heating the walls is likely be small (only a few percent),
        # so we'll ignore it (see Van Dresar paper).
        Q_dot = inputs["Q_dot"]
        Q_dot_gas_int = heat_transfer_coeff_gas_int * A_int * (T_gas - T_int)

        # Determine heat flows into bulk liquid and ullage
        Q_dot_gas = Q_dot * A_dry / (A_wet + A_dry)
        Q_dot_liq = Q_dot * A_wet / (A_wet + A_dry)

        # ============================================ ODEs ============================================
        # Compute the boil off mass flow rate
        m_dot_boil_off = Q_dot_gas_int / (cp_liq * (T_int - T_liq) + (h_gas - h_int) + (h_gas - h_sat_gas))

        # Mass flows
        m_dot_gas = m_dot_boil_off + m_dot_gas_in - m_dot_gas_out
        m_dot_liq = m_dot_liq_in - m_dot_boil_off - m_dot_liq_out

        V_dot_liq = m_dot_liq / rho_liq
        V_dot_gas = -V_dot_liq

        T_dot_gas = (Q_dot_gas - Q_dot_gas_int - P_gas * V_dot_gas + m_dot_gas * (h_gas - u_gas)) / (m_gas * cv_gas)
        T_dot_liq = (Q_dot_liq + P_liq * V_dot_liq - m_dot_liq * (h_liq + u_liq)) / (m_liq * cp_liq)
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

        # Ullage pressure (useful for other stuff)
        outputs["P_gas"] = P_gas


class BoilOffFillLevelCalc(om.ExplicitComponent):
    """
    Computes the fill level in the tank given the
    weight of the liquid.

    Inputs
    ------
    m_liq : float
        Mass of the liquid (vector, kg)
    radius : float
        Radius inside of tank for the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLINDRICAL part of the tank (scalar, m)
    T_liq : float
        Temperature of the bulk liquid (vector, K)

    Outputs
    -------
    fill_level : float
        Volume fraction of tank (in range 0-1) filled with liquid propellant (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("radius", val=2.0, units="m")
        self.add_input("length", val=0.5, units="m")
        self.add_input("V_gas", units="m**3", shape=(nn,))
        self.add_output("fill_level", val=0.5, shape=(nn,), lower=0.005, upper=0.995)

        arng = np.arange(nn)
        self.declare_partials("fill_level", ["V_gas"], rows=arng, cols=arng)
        self.declare_partials("fill_level", ["radius", "length"], rows=arng, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        r = inputs["radius"]
        L = inputs["length"]

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        outputs["fill_level"] = 1 - inputs["V_gas"] / V_tank

    def compute_partials(self, inputs, J):
        r = inputs["radius"]
        L = inputs["length"]

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        J["fill_level", "V_gas"] = 1 - 1 / V_tank
        J["fill_level", "radius"] = 1 + inputs["V_gas"] / V_tank**2 * (4 * np.pi * r**2 + 2 * np.pi * r * L)
        J["fill_level", "length"] = 1 + inputs["V_gas"] / V_tank**2 * (np.pi * r**2)