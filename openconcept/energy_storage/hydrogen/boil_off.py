import numpy as np
from copy import deepcopy
import openmdao.api as om
from openconcept.utilities.constants import GRAV_CONST, UNIVERSAL_GAS_CONST, MOLEC_WEIGHT_H2
from openconcept.utilities import Integrator

# TODO: These curve fits are poorly conditioned (hurt nonlinear solver performance)
#       and might not cover a large enough range
import openconcept.energy_storage.hydrogen.H2_properties as H2_prop


# Sometimes OpenMDAO's bound enforcement doesn't work properly,
# so enforce these bounds within compute methods to avoid divide
# by zero errors.
LIQ_HEIGHT_FRAC_LOWER_ENFORCE = 1e-7
LIQ_HEIGHT_FRAC_UPPER_ENFORCE = 1.0 - 1e-7


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

    WARNING: Do not modify or connect anything to the initial integrated delta state values
             ("integ.delta_m_gas_initial", "integ.delta_m_liq_initial", etc.). They must
             remain zero for the initial tank state to be the expected value. Set the initial
             tank condition using the BoilOff options.

    Inputs
    ------
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    m_dot_gas_out : float
        Mass flow rate of gaseous hydrogen out of the ullage; could be for venting
        or gaseous hydrogen consumption (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid hydrogen out of the tank; this is where fuel being consumed
        is bookkept, assuming it is removed from the tank as a liquid (vector, kg/s)
    Q_gas : float
        Heat flow rate through the tank walls into the ullage (vector, W)
    Q_liq : float
        Heat flow rate through the tank walls into the bulk liquid (vector, W)
    Q_add : float
        Additional heat added directly to the bulk liquid by a heater this heat
        is assumed to go directly to boiling the liquid, rather than also heating
        the bulk liquid as Q_liq does (vector, W)

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
    fill_level_init : float
        Initial fill level (in range 0-1) of the tank, default 0.95
        to leave space for boil-off gas; 5% adopted from Millis et al. 2009 (scalar, dimensionless)
    ullage_T_init : float
        Initial temperature of gas in ullage, default 21 K (scalar, K)
    ullage_P_init : float
        Initial pressure of gas in ullage, default 150,000 Pa; ullage pressure must be higher than ambient
        to prevent air leaking in and creating a combustible mixture (scalar, Pa)
    liquid_T_init : float
        Initial temperature of bulk liquid hydrogen, default 20 K (scalar, K)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("fill_level_init", default=0.95, desc="Initial fill level")
        self.options.declare("ullage_T_init", default=21.0, desc="Initial ullage temp (K)")
        self.options.declare("ullage_P_init", default=1.5e5, desc="Initial ullage pressure (Pa)")
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
        self.connect("liq_height_calc.h_liq_frac", "interface_params.h_liq_frac")

        # Compute the ODE equations to be integrated
        self.add_subsystem(
            "boil_off_ode",
            LH2BoilOffODE(num_nodes=nn),
            promotes_inputs=[
                "m_dot_gas_out",
                "m_dot_liq_out",
                "Q_gas",
                "Q_liq",
                "Q_add",
            ],
            promotes_outputs=["P_gas"],
        )
        self.connect("interface_params.A_interface", "boil_off_ode.A_interface")
        self.connect("interface_params.L_interface", "boil_off_ode.L_interface")

        # The initial tank states are specified indirectly by the fill_level_init, ullage_T_init, ullage_P_init,
        # and liquid_T_init options, along with the input tank radius and length. We can't connect a component
        # directly to the integrator's inputs because those initial values are linked between phases. Thus, we
        # use a bit of a trick where we actually integrate the change in the state values since the beginning
        # of the mission and then add their correct initial values in the add_init_state_values component.
        integ = self.add_subsystem(
            "integ",
            Integrator(num_nodes=nn, diff_units="s", time_setup="duration", method="bdf3"),
        )
        integ.add_integrand("delta_m_gas", rate_name="m_dot_gas", units="kg", val=0, start_val=0)
        integ.add_integrand("delta_m_liq", rate_name="m_dot_liq", units="kg", val=0, start_val=0)
        integ.add_integrand("delta_T_gas", rate_name="T_dot_gas", units="K", val=0, start_val=0)
        integ.add_integrand("delta_T_liq", rate_name="T_dot_liq", units="K", val=0, start_val=0)
        integ.add_integrand("delta_V_gas", rate_name="V_dot_gas", units="m**3", val=0, start_val=0)

        self.add_subsystem(
            "add_init_state_values",
            InitialTankStateModification(
                num_nodes=nn,
                fill_level_init=self.options["fill_level_init"],
                ullage_T_init=self.options["ullage_T_init"],
                ullage_P_init=self.options["ullage_P_init"],
                liquid_T_init=self.options["liquid_T_init"],
            ),
            promotes_inputs=["radius", "length"],
            promotes_outputs=["m_liq", "m_gas", "T_liq", "T_gas"],
        )

        # Connect the integrated delta states to the component that increments them by their computed initial values
        self.connect("integ.delta_m_gas", "add_init_state_values.delta_m_gas")
        self.connect("integ.delta_m_liq", "add_init_state_values.delta_m_liq")
        self.connect("integ.delta_T_gas", "add_init_state_values.delta_T_gas")
        self.connect("integ.delta_T_liq", "add_init_state_values.delta_T_liq")
        self.connect("integ.delta_V_gas", "add_init_state_values.delta_V_gas")

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
        self.connect("add_init_state_values.V_gas", ["boil_off_ode.V_gas", "level_calc.V_gas"])

        # Set defaults for inputs promoted from multiple sources
        self.set_input_defaults("radius", 1.0, units="m")
        self.set_input_defaults("length", 0.5, units="m")

        # Set a solver specifically for this component in an attempt to increase robustness
        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options["solve_subsystems"] = False
        self.nonlinear_solver.options["maxiter"] = 10
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["rtol"] = 1e-9
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
            bound_enforcement="scalar", alpha=1.0, iprint=0, print_bound_enforce=False
        )

    def guess_nonlinear(self, inputs, outputs, resids):
        # If the model is already converged, don't change anything
        norm = resids.get_norm()
        if norm < 1e-2 and norm != 0.0:
            return

        # If the initial integrated state values are not zero, this component lives in
        # the middle or at the end of a mission where the initial integrated state is
        # set by the final integrated state in the previous mission phase. In any of
        # these cases, just use those initial values as the guess.
        if (
            inputs["integ.delta_m_gas_initial"].item() != 0.0
            and inputs["integ.delta_m_liq_initial"].item() != 0.0
            and inputs["integ.delta_T_gas_initial"].item() != 0.0
            and inputs["integ.delta_T_liq_initial"].item() != 0.0
            and inputs["integ.delta_V_gas_initial"].item() != 0.0
        ):
            outputs["integ.delta_m_gas"] = inputs["integ.delta_m_gas_initial"]
            outputs["integ.delta_m_liq"] = inputs["integ.delta_m_liq_initial"]
            outputs["integ.delta_T_gas"] = inputs["integ.delta_T_gas_initial"]
            outputs["integ.delta_T_liq"] = inputs["integ.delta_T_liq_initial"]
            outputs["integ.delta_V_gas"] = inputs["integ.delta_V_gas_initial"]
            return

        # The remaining case is that the model is not yet converged and the
        # initial values must be set, so do the thing
        r = inputs["level_calc.radius"]
        L = inputs["level_calc.length"]
        fill_init = self.options["fill_level_init"]
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
        outputs["add_init_state_values.V_gas"] = V_gas


class LiquidHeight(om.ImplicitComponent):
    """
    Implicitly compute the height of liquid in the tank.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `. ~~~~~~~~~~~~~~~~~~ ,'      -.- h  -->  h_liq_frac = h / (2 * radius)
         ` -------------- '         -'-

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
    h_liq_frac : float
        Height of the liquid in the tank nondimensionalized by the height of
        the tank; 1.0 indicates the height is two radii (at the top of the tank)
        and 0.0 indicates the height is zero (vector, dimensionless)

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

        self.add_output("h_liq_frac", val=0.5, shape=(nn,), lower=1e-3, upper=1.0 - 1e-3)

        arng = np.arange(nn)
        self.declare_partials("h_liq_frac", ["radius", "length"], rows=arng, cols=np.zeros(nn))
        self.declare_partials("h_liq_frac", ["h_liq_frac", "fill_level"], rows=arng, cols=arng)

    def apply_nonlinear(self, inputs, outputs, residuals):
        fill = inputs["fill_level"]
        r = inputs["radius"]
        L = inputs["length"]
        h_frac = outputs["h_liq_frac"]
        h_frac[h_frac <= LIQ_HEIGHT_FRAC_LOWER_ENFORCE] = LIQ_HEIGHT_FRAC_LOWER_ENFORCE
        h_frac[h_frac >= LIQ_HEIGHT_FRAC_UPPER_ENFORCE] = LIQ_HEIGHT_FRAC_UPPER_ENFORCE
        h = h_frac * 2 * r

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
        residuals["h_liq_frac"] = V_sph + V_cyl - V_tank * fill

    def linearize(self, inputs, outputs, J):
        fill = inputs["fill_level"]
        r = inputs["radius"]
        L = inputs["length"]
        h_frac = outputs["h_liq_frac"]
        h_frac[h_frac <= LIQ_HEIGHT_FRAC_LOWER_ENFORCE] = LIQ_HEIGHT_FRAC_LOWER_ENFORCE
        h_frac[h_frac >= LIQ_HEIGHT_FRAC_UPPER_ENFORCE] = LIQ_HEIGHT_FRAC_UPPER_ENFORCE
        h = h_frac * 2 * r

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

        J["h_liq_frac", "radius"] = Vsph_r + Vcyl_r - Vtank_r * fill + (Vsph_h + Vcyl_h) * 2 * outputs["h_liq_frac"]
        J["h_liq_frac", "length"] = Vcyl_L - Vtank_L * fill
        J["h_liq_frac", "fill_level"] = -V_tank
        J["h_liq_frac", "h_liq_frac"] = (Vsph_h + Vcyl_h) * 2 * r

    def guess_nonlinear(self, inputs, outputs, residuals):
        # Guess the height initially using a linear approximation of height w.r.t. fill level
        outputs["h_liq_frac"] = inputs["fill_level"]


class BoilOffGeometry(om.ExplicitComponent):
    """
    Compute areas and volumes in the tank from fill level.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `. ~~~~~~~~~~~~~~~~~~ ,'      -.- h  -->  h_liq_frac = h / (2 * radius)
         ` -------------- '         -'-

    Inputs
    ------
    h_liq_frac : float
        Height of the liquid in the tank nondimensionalized by the height of
        the tank; 1.0 indicates the height is two radii (at the top of the tank)
        and 0.0 indicates the height is zero (vector, dimensionless)
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
        self.add_input("h_liq_frac", shape=(nn,))
        self.add_input("radius", units="m")
        self.add_input("length", units="m")

        self.add_output("A_interface", units="m**2", shape=(nn,), lower=1e-5, val=3.0)
        self.add_output("L_interface", units="m", shape=(nn,), lower=1e-5, val=1.0)
        self.add_output("A_wet", units="m**2", shape=(nn,), lower=1e-5, val=5.0)
        self.add_output("A_dry", units="m**2", shape=(nn,), lower=1e-5, val=5.0)

        arng = np.arange(nn)
        self.declare_partials(["*"], "h_liq_frac", rows=arng, cols=arng)
        self.declare_partials(["*"], ["radius", "length"], rows=arng, cols=np.zeros(nn))

        # Prevent h_liq_frac input from being evaluated at 0 or 1 (made a variable
        # here so can be turned off for unit testing)
        self.adjust_h_liq_frac = True

    def compute(self, inputs, outputs):
        r = inputs["radius"]
        L = inputs["length"]
        h_frac = inputs["h_liq_frac"]
        if self.adjust_h_liq_frac:
            h_frac[h_frac <= LIQ_HEIGHT_FRAC_LOWER_ENFORCE] = LIQ_HEIGHT_FRAC_LOWER_ENFORCE
            h_frac[h_frac >= LIQ_HEIGHT_FRAC_UPPER_ENFORCE] = LIQ_HEIGHT_FRAC_UPPER_ENFORCE
        h = h_frac * 2 * r

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
        r = inputs["radius"]
        L = inputs["length"]
        h_frac = inputs["h_liq_frac"]
        if self.adjust_h_liq_frac:
            h_frac[h_frac <= LIQ_HEIGHT_FRAC_LOWER_ENFORCE] = LIQ_HEIGHT_FRAC_LOWER_ENFORCE
            h_frac[h_frac >= LIQ_HEIGHT_FRAC_UPPER_ENFORCE] = LIQ_HEIGHT_FRAC_UPPER_ENFORCE
        h = h_frac * 2 * r

        # Derivatives of chord and central angle of segment w.r.t. height and radius
        c = 2 * np.sqrt(2 * r * h - h**2)  # chord length of circular segment
        c_r = 2 * h / np.sqrt(2 * r * h - h**2)
        c_h = (2 * r - 2 * h) / np.sqrt(2 * r * h - h**2)

        th = 2 * np.arccos(1 - h / r)  # central angle of circular segment
        th_r = -2 / np.sqrt(1 - (1 - h / r) ** 2) * h / r**2
        th_h = 2 / np.sqrt(1 - (1 - h / r) ** 2) / r

        J["A_interface", "h_liq_frac"] = c_h * (np.pi * c / 2 + L) * 2 * r
        J["A_interface", "radius"] = (
            c_r * (np.pi * c / 2 + L) + J["A_interface", "h_liq_frac"] / r * inputs["h_liq_frac"]
        )
        J["A_interface", "length"] = c

        J["L_interface", "h_liq_frac"] = c_h * 2 * r
        J["L_interface", "radius"] = c_r + J["L_interface", "h_liq_frac"] / r * inputs["h_liq_frac"]
        J["L_interface", "length"] *= 0.0

        J["A_wet", "h_liq_frac"] = (2 * np.pi * r + th_h * r * L) * 2 * r
        J["A_wet", "radius"] = (
            2 * np.pi * h + th * L + th_r * r * L + J["A_wet", "h_liq_frac"] / r * inputs["h_liq_frac"]
        )
        J["A_wet", "length"] = th * r

        J["A_dry", "h_liq_frac"] = -J["A_wet", "h_liq_frac"]
        J["A_dry", "radius"] = 8 * np.pi * r + 2 * np.pi * L - J["A_wet", "radius"]
        J["A_dry", "length"] = 2 * np.pi * r - J["A_wet", "length"]


class LH2BoilOffODE(om.ExplicitComponent):
    """
    Compute the derivatives of the state values for the liquid hydrogen boil-off process
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
    m_dot_gas_out : float
        Mass flow rate of gaseous hydrogen out of the ullage; could be for venting
        or gaseous hydrogen consumption (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid hydrogen out of the tank; this is where fuel being consumed
        is bookkept, assuming it is removed from the tank as a liquid (vector, kg/s)
    Q_gas : float
        Heat flow rate through the tank walls into the ullage (vector, W)
    Q_liq : float
        Heat flow rate through the tank walls into the bulk liquid (vector, W)
    Q_add : float
        Additional heat added directly to the bulk liquid by a heater this heat
        is assumed to go directly to boiling the liquid, rather than also heating
        the bulk liquid as Q_liq does (vector, W)
    A_interface : float
        Area of the surface of the liquid in the tank. This is the area of
        the interface between the ullage and bulk liquid portions, hence
        the name (vector, m^2)
    L_interface : float
        Characteristic length of the interface between the ullage and the
        bulk liquid (vector, m)

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
        Minimum operating pressure of the tank, by default 120,000 Pa; slightly above atmospheric (scalar, Pa)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("P_min", default=120e3, desc="Minimum operating pressure of the tank")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("m_gas", units="kg", shape=(nn,))
        self.add_input("m_liq", units="kg", shape=(nn,))
        self.add_input("T_gas", units="K", shape=(nn,))
        self.add_input("T_liq", units="K", shape=(nn,))
        self.add_input("V_gas", units="m**3", shape=(nn,))
        self.add_input("m_dot_gas_out", units="kg/s", shape=(nn,), val=0.0)
        self.add_input("m_dot_liq_out", units="kg/s", shape=(nn,), val=0.0)
        self.add_input("Q_gas", units="W", shape=(nn,), val=0.0)
        self.add_input("Q_liq", units="W", shape=(nn,), val=0.0)
        self.add_input("Q_add", units="W", shape=(nn,), val=0.0)
        self.add_input("A_interface", units="m**2", shape=(nn,))
        self.add_input("L_interface", units="m", shape=(nn,))

        self.add_output("m_dot_gas", units="kg/s", shape=(nn,), val=0.0)
        self.add_output("m_dot_liq", units="kg/s", shape=(nn,), val=0.0)
        self.add_output("T_dot_gas", units="K/s", shape=(nn,), val=0.0)
        self.add_output("T_dot_liq", units="K/s", shape=(nn,), val=0.0)
        self.add_output("V_dot_gas", units="m**3/s", shape=(nn,), val=0.0)
        self.add_output("P_gas", units="Pa", shape=(nn,), val=1e5, lower=1e3)

        arng = np.arange(nn)
        self.declare_partials(
            "m_dot_gas",
            ["Q_add", "A_interface", "L_interface", "T_gas", "T_liq", "V_gas", "m_dot_gas_out", "m_gas"],
            rows=arng,
            cols=arng,
            method="cs",
        )
        self.declare_partials(
            ["m_dot_liq", "V_dot_gas"],
            ["Q_add", "A_interface", "L_interface", "T_gas", "T_liq", "V_gas", "m_dot_liq_out", "m_gas"],
            rows=arng,
            cols=arng,
            method="cs",
        )
        self.declare_partials(
            "T_dot_liq",
            [
                "Q_liq",
                "Q_add",
                "A_interface",
                "L_interface",
                "T_gas",
                "T_liq",
                "V_gas",
                "m_dot_liq_out",
                "m_gas",
                "m_liq",
            ],
            rows=arng,
            cols=arng,
            method="cs",
        )
        self.declare_partials(
            "T_dot_gas",
            [
                "Q_gas",
                "Q_add",
                "A_interface",
                "L_interface",
                "T_gas",
                "T_liq",
                "V_gas",
                "m_dot_gas_out",
                "m_dot_liq_out",
                "m_gas",
                "m_liq",
            ],
            rows=arng,
            cols=arng,
            method="cs",
        )
        self.declare_partials("P_gas", ["m_gas", "T_gas", "V_gas"], rows=arng, cols=arng)

        # Compute the maximum allowable temperature of the liquid.
        # The maximum allowable temperature of the liquid is the saturation temperature
        # at the minimum pressure. If it is at this temperature, don't let it increase further.
        self.T_liq_max = H2_prop.sat_gh2_T(self.options["P_min"])

        # Use this to check if the compute method has been called already with the same inputs
        self.inputs_cache = None

    def _process_inputs(self, inputs):
        """
        Adds a small perturbation to any input states that have values of zero or less and
        shouldn't (either because it's nonphysical or causes divide by zero errors). Returns
        a new dictionary with modified inputs.

        See OpenMDAO issue #2824 for more details on why this might happen despite putting
        bounds on these outputs.
        """
        adjusted_inputs = deepcopy(dict(inputs))
        new_val = 1e-10
        inputs_to_change = ["m_gas", "m_liq", "T_gas", "T_liq", "V_gas", "A_interface", "L_interface"]

        for var in inputs_to_change:
            adjusted_inputs[var][adjusted_inputs[var] <= new_val] = new_val

        return adjusted_inputs

    def compute(self, inputs_orig, outputs):
        inputs = self._process_inputs(inputs_orig)

        # Unpack the states from the inputs
        m_gas = inputs["m_gas"]
        m_liq = inputs["m_liq"]
        T_gas = inputs["T_gas"]
        T_liq = inputs["T_liq"]
        V_gas = inputs["V_gas"]
        self.inputs_cache = deepcopy(dict(inputs))

        m_dot_gas_out = inputs["m_dot_gas_out"]  # gas released for venting or consumption
        m_dot_liq_out = inputs["m_dot_liq_out"]  # liquid leaving the tank (e.g., for fuel to the engines)

        # Heat flows into bulk liquid and ullage
        Q_gas = inputs["Q_gas"]
        Q_liq = inputs["Q_liq"]
        Q_add = inputs["Q_add"]  # heat added to bulk liquid with heater

        A_int = inputs["A_interface"]  # area of the surface of the bulk liquid (the interface)
        L_int = inputs["L_interface"]  # characteristic length of the interface

        # =============================== Compute physical properties ===============================
        # Use ideal gas law to compute ullage pressure (real gas properties are used elsewhere)
        self.P_gas = P_gas = m_gas * T_gas * UNIVERSAL_GAS_CONST / (V_gas * MOLEC_WEIGHT_H2)

        # Ullage gas properties
        self.h_gas = H2_prop.gh2_h(P_gas, T_gas)  # enthalpy
        self.u_gas = H2_prop.gh2_u(P_gas, T_gas)  # internal energy
        self.cv_gas = H2_prop.gh2_cv(P_gas, T_gas)  # specific heat at constant volume

        # Bulk liquid properties
        self.h_liq = H2_prop.lh2_h(T_liq)  # enthalpy
        self.u_liq = H2_prop.lh2_u(T_liq)  # internal energy
        self.cp_liq = H2_prop.lh2_cp(T_liq)  # specific heat at constant pressure
        self.rho_liq = H2_prop.lh2_rho(T_liq)  # density
        self.P_liq = H2_prop.lh2_P(T_liq)  # pressure

        # Temperature of the interface assumes saturated hydrogen with same pressure as the ullage
        self.T_int = T_int = H2_prop.sat_gh2_T(P_gas)  # use saturated GH2 temperature
        self.h_int = H2_prop.lh2_h(T_int)  # use saturated LH2 enthalpy

        # Saturated gas properties at the mean film temperature
        self.T_mean_film = T_mean_film = 0.5 * (T_gas + T_int)
        self.cp_sat_gas = H2_prop.sat_gh2_cp(T_mean_film)  # specific heat at constant pressure
        self.visc_sat_gas = H2_prop.sat_gh2_viscosity(T_mean_film)  # viscosity
        self.k_sat_gas = H2_prop.sat_gh2_k(T_mean_film)  # thermal conductivity
        self.beta_sat_gas = H2_prop.sat_gh2_beta(T_mean_film)  # coefficient of thermal expansion
        self.rho_sat_gas = H2_prop.sat_gh2_rho(T_mean_film)  # density
        self.h_sat_gas = H2_prop.sat_gh2_h(T_int)  # enthalpy

        # ==================== Compute heat transfer between ullage and interface ====================
        # Compute the heat transfer coefficient for the heat transfer from the ullage to the interface.
        # Evaluate the heat transfer coefficient between the ullage and interface using the saturated gas
        # properties at the mean film temperature (average of ullage and interface temps) because the ullage
        # near the interface is close to saturated due to thermal stratification effects.
        # Use constants associated with the top of a cold horizontal surface
        self.C_const = 0.27
        self.n_const = 0.25

        # Compute the fluid properties for heat transfer
        self.prandtl = self.cp_sat_gas * self.visc_sat_gas / self.k_sat_gas
        self.grashof = (
            GRAV_CONST
            * self.beta_sat_gas
            * self.rho_sat_gas**2
            * np.sqrt((T_gas - T_int) ** 2)  # use sqrt of square as absolute value shorthand so complex-safe
            * L_int**3
            / self.visc_sat_gas**2
        )
        self.prandtl[np.real(self.prandtl) < 0] = 0.0
        self.grashof[np.real(self.grashof) < 0] = 0.0
        self.nusselt = self.C_const * (self.prandtl * self.grashof) ** self.n_const
        self.heat_transfer_coeff_gas_int = self.k_sat_gas / L_int * self.nusselt

        # Heat from the environment that goes to heating the walls is likely be small (only a few percent),
        # so we'll ignore it (see Van Dresar paper).
        self.Q_gas_int = Q_gas_int = self.heat_transfer_coeff_gas_int * A_int * (T_gas - T_int)

        # ============================================ ODEs ============================================
        # Compute the boil-off mass flow rate
        # TODO: double check the denominator
        boil_off_denom = self.cp_liq * (T_int - T_liq) + (self.h_gas - self.h_int) + (self.h_gas - self.h_sat_gas)
        self.m_dot_boil_off = Q_gas_int / boil_off_denom
        m_dot_boil_off_heat_added = Q_add / boil_off_denom
        total_boil_off = self.m_dot_boil_off + m_dot_boil_off_heat_added

        # Mass flows
        self.m_dot_gas = total_boil_off - m_dot_gas_out
        self.m_dot_liq = -total_boil_off - m_dot_liq_out

        # TODO: since liquid density changes, should there be an additional term with a rho_dot?
        self.V_dot_liq = self.m_dot_liq / self.rho_liq
        self.V_dot_gas = -self.V_dot_liq

        self.T_dot_gas = (
            Q_gas
            - Q_gas_int
            - P_gas * self.V_dot_gas
            + self.m_dot_boil_off * (self.h_gas - self.u_gas)
            - m_dot_gas_out * self.h_gas
        ) / (m_gas * self.cv_gas)
        self.T_dot_liq = (
            Q_liq
            - self.P_liq * self.V_dot_liq
            + self.m_dot_boil_off * (self.h_liq - self.u_liq)
            - m_dot_liq_out * self.h_liq
        ) / (m_liq * self.cp_liq)

        # The maximum allowable temperature of the liquid is the saturation temperature
        # at the minimum pressure. If it is at this temperature, don't let it increase further.
        self.T_dot_liq[np.real(T_liq) >= self.T_liq_max] *= 0.0

        # We got em!
        outputs["m_dot_gas"] = self.m_dot_gas
        outputs["m_dot_liq"] = self.m_dot_liq
        outputs["T_dot_gas"] = self.T_dot_gas
        outputs["T_dot_liq"] = self.T_dot_liq
        outputs["V_dot_gas"] = self.V_dot_gas

        # Ullage pressure (useful for other stuff)
        outputs["P_gas"] = P_gas

    # def compute_partials(self, inputs_orig, J):
    #     inputs = self._process_inputs(inputs_orig)

    #     # Check that the compute method has been called with the same inputs
    #     if self.inputs_cache is None:
    #         self.compute(inputs, {})
    #     else:
    #         for name in inputs.keys():
    #             try:
    #                 if np.any(inputs[name] != self.inputs_cache[name]):
    #                     raise ValueError()
    #             except:
    #                 self.compute(inputs, {})
    #                 break

    #     # Unpack the states from the inputs
    #     m_gas = inputs["m_gas"]
    #     m_liq = inputs["m_liq"]
    #     T_gas = inputs["T_gas"]
    #     T_liq = inputs["T_liq"]
    #     V_gas = inputs["V_gas"]

    #     # Heat input
    #     Q = inputs["Q"]

    #     # ============================== Compute geometric quantities ==============================
    #     A_int = inputs["A_interface"]  # area of the surface of the bulk liquid (the interface)
    #     L_int = inputs["L_interface"]  # characteristic length of the interface
    #     A_wet = inputs["A_wet"]
    #     A_dry = inputs["A_dry"]

    #     # ============================== Use reverse AD-style approach ==============================
    #     # ------------------------------ m_dot_gas ------------------------------
    #     # Initial seed with desired output
    #     d_m_dot_gas = 1.0

    #     # Influence of m_dot_boil_off on computing m_dot_gas
    #     d_m_dot_boil_off = 1.0 * d_m_dot_gas

    #     # Influence of terms in m_dot_boil_off
    #     d_Q_gas_int = d_m_dot_boil_off / (
    #         self.cp_liq * (self.T_int - T_liq) + (self.h_gas - self.h_int) + (self.h_gas - self.h_sat_gas)
    #     )
    #     deriv_m_dot_boil_off_denom = (
    #         -self.Q_gas_int
    #         / (self.cp_liq * (self.T_int - T_liq) + 2 * self.h_gas - self.h_int - self.h_sat_gas) ** 2
    #     )  # derivative of terms in denomenator without associated chain rule
    #     d_cp_liq = deriv_m_dot_boil_off_denom * (self.T_int - T_liq) * d_m_dot_boil_off
    #     # do d_T_int in the for loop since it'll get overwritten otherwise, save the variable temporarily
    #     d_T_int_temp = deriv_m_dot_boil_off_denom * self.cp_liq * d_m_dot_boil_off
    #     d_T_liq = deriv_m_dot_boil_off_denom * (-self.cp_liq) * d_m_dot_boil_off
    #     d_h_gas = deriv_m_dot_boil_off_denom * 2 * d_m_dot_boil_off
    #     d_h_int = deriv_m_dot_boil_off_denom * (-1) * d_m_dot_boil_off
    #     d_h_sat_gas = deriv_m_dot_boil_off_denom * (-1) * d_m_dot_boil_off

    #     # Since this portion of the code is almost identical to the reverse AD code
    #     # necessary for T_dos_gas derivatives, use a loop here to do both
    #     for Q_gas_int_seed, output_name in zip(
    #         [-1 / (m_gas * self.cv_gas), d_Q_gas_int], ["T_dot_gas", "m_dot_gas"]
    #     ):
    #         is_m_dot = output_name == "m_dot_gas"

    #         # Influence of terms in Q_gas_int
    #         d_heat_transfer_coeff_gas_int = A_int * (T_gas - self.T_int) * Q_gas_int_seed
    #         d_A_interface = self.Q_gas_int / A_int * Q_gas_int_seed
    #         d_T_gas = self.Q_gas_int / (T_gas - self.T_int) * Q_gas_int_seed
    #         if is_m_dot:
    #             d_T_int = d_T_int_temp
    #             d_T_int -= d_T_gas
    #         else:
    #             d_T_int = -d_T_gas

    #         # Influence of terms in heat_transfer_coeff_gas_int
    #         d_k_sat_gas = self.nusselt / L_int * d_heat_transfer_coeff_gas_int
    #         d_L_interface = -self.heat_transfer_coeff_gas_int / L_int * d_heat_transfer_coeff_gas_int
    #         d_nusselt = self.k_sat_gas / L_int * d_heat_transfer_coeff_gas_int

    #         # Influence of terms in nusselt
    #         NaN_mask = np.logical_and(self.prandtl != 0.0, self.grashof != 0.0)
    #         d_grashof = np.zeros_like(self.grashof)
    #         d_prandtl = np.zeros_like(self.prandtl)
    #         d_grashof[NaN_mask] = (
    #             self.n_const
    #             * self.C_const
    #             * (self.prandtl[NaN_mask] * self.grashof[NaN_mask]) ** (self.n_const - 1)
    #             * self.prandtl[NaN_mask]
    #             * d_nusselt[NaN_mask]
    #         )
    #         d_prandtl[NaN_mask] = (
    #             self.n_const
    #             * self.C_const
    #             * (self.prandtl[NaN_mask] * self.grashof[NaN_mask]) ** (self.n_const - 1)
    #             * self.grashof[NaN_mask]
    #             * d_nusselt[NaN_mask]
    #         )

    #         # Influence of terms in grashof
    #         # These derivatives are zero anywhere the grashof number was zeroed out
    #         mask = np.ones(self.options["num_nodes"])
    #         mask[self.grashof < 0] = 0.0
    #         abs_val_mult = np.ones(self.options["num_nodes"])
    #         abs_val_mult[(T_gas - self.T_int) < 0] = -1.0
    #         d_beta_sat_gas = self.grashof / self.beta_sat_gas * mask * d_grashof
    #         d_rho_sat_gas = 2 * self.grashof / self.rho_sat_gas * mask * d_grashof
    #         d_T_gas += self.grashof / np.abs(T_gas - self.T_int) * abs_val_mult * mask * d_grashof
    #         d_T_int += self.grashof / np.abs(T_gas - self.T_int) * (-abs_val_mult) * mask * d_grashof
    #         d_L_interface += 3 * self.grashof / L_int * mask * d_grashof
    #         d_visc_sat_gas = -2 * self.grashof / self.visc_sat_gas * mask * d_grashof

    #         # Influence of terms in prandtl
    #         # These derivatives are zero anywhere the grashof number was zeroed out
    #         mask = np.ones(self.options["num_nodes"])
    #         mask[self.prandtl < 0] = 0.0
    #         d_cp_sat_gas = self.visc_sat_gas / self.k_sat_gas * mask * d_prandtl
    #         d_visc_sat_gas += self.cp_sat_gas / self.k_sat_gas * mask * d_prandtl
    #         d_k_sat_gas += -self.prandtl / self.k_sat_gas * mask * d_prandtl

    #         # T_mean_film (and P_gas on h_sat_gas) influence on saturated gas properties
    #         if is_m_dot:
    #             d_T_int += H2_prop.sat_gh2_h(self.T_int, deriv=True) * d_h_sat_gas
    #         d_T_mean_film = (
    #             H2_prop.sat_gh2_cp(self.T_mean_film, deriv=True) * d_cp_sat_gas
    #             + H2_prop.sat_gh2_viscosity(self.T_mean_film, deriv=True) * d_visc_sat_gas
    #             + H2_prop.sat_gh2_k(self.T_mean_film, deriv=True) * d_k_sat_gas
    #             + H2_prop.sat_gh2_beta(self.T_mean_film, deriv=True) * d_beta_sat_gas
    #             + H2_prop.sat_gh2_rho(self.T_mean_film, deriv=True) * d_rho_sat_gas
    #         )

    #         # Influence of terms in T_mean_film
    #         d_T_gas += 0.5 * d_T_mean_film
    #         d_T_int += 0.5 * d_T_mean_film

    #         # T_int influence on h_int
    #         if is_m_dot:
    #             d_T_int += H2_prop.lh2_h(self.T_int, deriv=True) * d_h_int

    #         # Influence of P_gas on T_int
    #         d_P_gas = H2_prop.sat_gh2_T(self.P_gas, deriv=True) * d_T_int

    #         if is_m_dot:
    #             # Influence of T_liq on cp_liq
    #             d_T_liq += H2_prop.lh2_cp(T_liq, deriv=True) * d_cp_liq

    #             # Influence of P_gas and T_gas on h_gas
    #             phpP, phpT = H2_prop.gh2_h(self.P_gas, T_gas, deriv=True)
    #             d_P_gas += phpP * d_h_gas
    #             d_T_gas += phpT * d_h_gas

    #         # Influence of terms in P_gas
    #         d_m_gas = self.P_gas / m_gas * d_P_gas
    #         d_T_gas += self.P_gas / T_gas * d_P_gas
    #         d_V_gas = -self.P_gas / V_gas * d_P_gas

    #         J[output_name, "A_interface"] = d_A_interface.copy()
    #         J[output_name, "L_interface"] = d_L_interface.copy()
    #         J[output_name, "T_gas"] = d_T_gas.copy()
    #         J[output_name, "V_gas"] = d_V_gas.copy()
    #         J[output_name, "m_gas"] = d_m_gas.copy()

    #     J["m_dot_gas", "T_liq"] = d_T_liq
    #     J["m_dot_gas", "m_dot_gas_in"] = 1.0
    #     J["m_dot_gas", "m_dot_gas_out"] = -1.0

    #     # ------------ m_dot_liq is mostly just negative of m_dot_gas ------------
    #     J["m_dot_liq", "A_interface"] = -d_A_interface
    #     J["m_dot_liq", "L_interface"] = -d_L_interface
    #     J["m_dot_liq", "T_gas"] = -d_T_gas
    #     J["m_dot_liq", "T_liq"] = -d_T_liq
    #     J["m_dot_liq", "V_gas"] = -d_V_gas
    #     J["m_dot_liq", "m_dot_liq_in"] = 1.0
    #     J["m_dot_liq", "m_dot_liq_out"] = -1.0
    #     J["m_dot_liq", "m_gas"] = -d_m_gas

    #     # ------------------------------ V_dot_liq ------------------------------
    #     J["V_dot_gas", "A_interface"] = -J["m_dot_liq", "A_interface"] / self.rho_liq
    #     J["V_dot_gas", "L_interface"] = -J["m_dot_liq", "L_interface"] / self.rho_liq
    #     J["V_dot_gas", "T_gas"] = -J["m_dot_liq", "T_gas"] / self.rho_liq
    #     J["V_dot_gas", "T_liq"] = -J[
    #         "m_dot_liq", "T_liq"
    #     ] / self.rho_liq + self.m_dot_liq / self.rho_liq**2 * H2_prop.lh2_rho(T_liq, deriv=True)
    #     J["V_dot_gas", "V_gas"] = -J["m_dot_liq", "V_gas"] / self.rho_liq
    #     J["V_dot_gas", "m_dot_liq_in"] = -J["m_dot_liq", "m_dot_liq_in"] / self.rho_liq
    #     J["V_dot_gas", "m_dot_liq_out"] = -J["m_dot_liq", "m_dot_liq_out"] / self.rho_liq
    #     J["V_dot_gas", "m_gas"] = -J["m_dot_liq", "m_gas"] / self.rho_liq

    #     # -------------------------------- P_gas --------------------------------
    #     J["P_gas", "m_gas"] = T_gas * UNIVERSAL_GAS_CONST / (V_gas * MOLEC_WEIGHT_H2)
    #     J["P_gas", "T_gas"] = m_gas * UNIVERSAL_GAS_CONST / (V_gas * MOLEC_WEIGHT_H2)
    #     J["P_gas", "V_gas"] = -m_gas * T_gas * UNIVERSAL_GAS_CONST * MOLEC_WEIGHT_H2 / (V_gas * MOLEC_WEIGHT_H2) ** 2

    #     # ------------------------------ T_dot_gas ------------------------------
    #     # Some of the T_dot_gas partials are already set in the m_dot_gas section
    #     # Q_gas portion
    #     partial_Q_gas = 1 / (m_gas * self.cv_gas)
    #     J["T_dot_gas", "Q"] = partial_Q_gas * A_dry / (A_wet + A_dry)
    #     J["T_dot_gas", "A_wet"] = partial_Q_gas * Q * (-A_dry) / (A_wet + A_dry) ** 2
    #     J["T_dot_gas", "A_dry"] = partial_Q_gas * Q * A_wet / (A_wet + A_dry) ** 2

    #     # P_gas portion
    #     partial_P_gas = -self.V_dot_gas / (m_gas * self.cv_gas)
    #     J["T_dot_gas", "m_gas"] += partial_P_gas * J["P_gas", "m_gas"]
    #     J["T_dot_gas", "T_gas"] += partial_P_gas * J["P_gas", "T_gas"]
    #     J["T_dot_gas", "V_gas"] += partial_P_gas * J["P_gas", "V_gas"]

    #     # V_dot_gas portion
    #     partial_V_dot_gas = -self.P_gas / (m_gas * self.cv_gas)
    #     J["T_dot_gas", "A_interface"] += partial_V_dot_gas * J["V_dot_gas", "A_interface"]
    #     J["T_dot_gas", "L_interface"] += partial_V_dot_gas * J["V_dot_gas", "L_interface"]
    #     J["T_dot_gas", "T_gas"] += partial_V_dot_gas * J["V_dot_gas", "T_gas"]
    #     J["T_dot_gas", "T_liq"] = partial_V_dot_gas * J["V_dot_gas", "T_liq"]
    #     J["T_dot_gas", "V_gas"] += partial_V_dot_gas * J["V_dot_gas", "V_gas"]
    #     J["T_dot_gas", "m_dot_liq_in"] = partial_V_dot_gas * J["V_dot_gas", "m_dot_liq_in"]
    #     J["T_dot_gas", "m_dot_liq_out"] = partial_V_dot_gas * J["V_dot_gas", "m_dot_liq_out"]
    #     J["T_dot_gas", "m_gas"] += partial_V_dot_gas * J["V_dot_gas", "m_gas"]

    #     # m_dot_gas portion
    #     partial_m_dot_gas = (self.h_gas - self.u_gas) / (m_gas * self.cv_gas)
    #     J["T_dot_gas", "A_interface"] += partial_m_dot_gas * J["m_dot_gas", "A_interface"]
    #     J["T_dot_gas", "L_interface"] += partial_m_dot_gas * J["m_dot_gas", "L_interface"]
    #     J["T_dot_gas", "T_gas"] += partial_m_dot_gas * J["m_dot_gas", "T_gas"]
    #     J["T_dot_gas", "T_liq"] += partial_m_dot_gas * J["m_dot_gas", "T_liq"]
    #     J["T_dot_gas", "V_gas"] += partial_m_dot_gas * J["m_dot_gas", "V_gas"]
    #     J["T_dot_gas", "m_dot_gas_in"] = partial_m_dot_gas * J["m_dot_gas", "m_dot_gas_in"]
    #     J["T_dot_gas", "m_dot_gas_out"] = partial_m_dot_gas * J["m_dot_gas", "m_dot_gas_out"]
    #     J["T_dot_gas", "m_gas"] += partial_m_dot_gas * J["m_dot_gas", "m_gas"]

    #     # h_gas and u_gas portions
    #     partial_h_gas = self.m_dot_gas / (m_gas * self.cv_gas)
    #     partial_u_gas = -partial_h_gas
    #     pupP, pupT = H2_prop.gh2_u(self.P_gas, T_gas, deriv=True)
    #     J["T_dot_gas", "m_gas"] += (
    #         partial_h_gas * phpP * J["P_gas", "m_gas"] + partial_u_gas * pupP * J["P_gas", "m_gas"]
    #     )
    #     J["T_dot_gas", "T_gas"] += (
    #         partial_h_gas * phpP * J["P_gas", "T_gas"]
    #         + partial_u_gas * pupP * J["P_gas", "T_gas"]
    #         + partial_h_gas * phpT
    #         + partial_u_gas * pupT
    #     )
    #     J["T_dot_gas", "V_gas"] += (
    #         partial_h_gas * phpP * J["P_gas", "V_gas"] + partial_u_gas * pupP * J["P_gas", "V_gas"]
    #     )

    #     # m_gas portion
    #     J["T_dot_gas", "m_gas"] += -self.T_dot_gas / m_gas

    #     # cv_gas portion
    #     partial_cv_gas = -self.T_dot_gas / self.cv_gas
    #     pcvpP, pcvpT = H2_prop.gh2_cv(self.P_gas, T_gas, deriv=True)
    #     J["T_dot_gas", "m_gas"] += partial_cv_gas * pcvpP * J["P_gas", "m_gas"]
    #     J["T_dot_gas", "T_gas"] += partial_cv_gas * (pcvpP * J["P_gas", "T_gas"] + pcvpT)
    #     J["T_dot_gas", "V_gas"] += partial_cv_gas * pcvpP * J["P_gas", "V_gas"]

    #     # ------------------------------ T_dot_liq ------------------------------
    #     J["T_dot_liq", "Q"] = 1 / (m_liq * self.cp_liq) * A_wet / (A_wet + A_dry)
    #     J["T_dot_liq", "A_wet"] = 1 / (m_liq * self.cp_liq) * Q * A_dry / (A_wet + A_dry) ** 2
    #     J["T_dot_liq", "A_dry"] = 1 / (m_liq * self.cp_liq) * Q * (-A_wet) / (A_wet + A_dry) ** 2
    #     J["T_dot_liq", "A_interface"] = (
    #         self.P_liq * J["V_dot_gas", "A_interface"] + (self.h_liq - self.u_liq) * J["m_dot_liq", "A_interface"]
    #     ) / (m_liq * self.cp_liq)
    #     J["T_dot_liq", "L_interface"] = (
    #         self.P_liq * J["V_dot_gas", "L_interface"] + (self.h_liq - self.u_liq) * J["m_dot_liq", "L_interface"]
    #     ) / (m_liq * self.cp_liq)
    #     J["T_dot_liq", "T_gas"] = (
    #         self.P_liq * J["V_dot_gas", "T_gas"] + (self.h_liq - self.u_liq) * J["m_dot_liq", "T_gas"]
    #     ) / (m_liq * self.cp_liq)
    #     J["T_dot_liq", "T_liq"] = (
    #         (self.P_liq * J["V_dot_gas", "T_liq"] + (self.h_liq - self.u_liq) * J["m_dot_liq", "T_liq"])
    #         / (m_liq * self.cp_liq)
    #         + -self.V_dot_liq / (m_liq * self.cp_liq) * H2_prop.lh2_P(T_liq, deriv=True)
    #         + self.m_dot_liq / (m_liq * self.cp_liq) * H2_prop.lh2_h(T_liq, deriv=True)
    #         + -self.m_dot_liq / (m_liq * self.cp_liq) * H2_prop.lh2_u(T_liq, deriv=True)
    #         + -self.T_dot_liq / self.cp_liq * H2_prop.lh2_cp(T_liq, deriv=True)
    #     )
    #     J["T_dot_liq", "V_gas"] = (
    #         self.P_liq * J["V_dot_gas", "V_gas"] + (self.h_liq - self.u_liq) * J["m_dot_liq", "V_gas"]
    #     ) / (m_liq * self.cp_liq)
    #     J["T_dot_liq", "m_dot_liq_in"] = (
    #         self.P_liq * J["V_dot_gas", "m_dot_liq_in"] + (self.h_liq - self.u_liq) * J["m_dot_liq", "m_dot_liq_in"]
    #     ) / (m_liq * self.cp_liq)
    #     J["T_dot_liq", "m_dot_liq_out"] = (
    #         self.P_liq * J["V_dot_gas", "m_dot_liq_out"] + (self.h_liq - self.u_liq) * J["m_dot_liq", "m_dot_liq_out"]
    #     ) / (m_liq * self.cp_liq)
    #     J["T_dot_liq", "m_gas"] = (
    #         self.P_liq * J["V_dot_gas", "m_gas"] + (self.h_liq - self.u_liq) * J["m_dot_liq", "m_gas"]
    #     ) / (m_liq * self.cp_liq)
    #     J["T_dot_liq", "m_liq"] = -self.T_dot_liq / m_liq

    #     for wrt in [
    #         "Q",
    #         "A_wet",
    #         "A_dry",
    #         "A_interface",
    #         "L_interface",
    #         "T_gas",
    #         "T_liq",
    #         "V_gas",
    #         "m_dot_liq_in",
    #         "m_dot_liq_out",
    #         "m_gas",
    #         "m_liq",
    #     ]:
    #         J["T_dot_liq", wrt][T_liq >= self.T_liq_max] *= 0.0


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
        self.add_output("fill_level", val=0.5, shape=(nn,), lower=0.001, upper=0.999)

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
        J["fill_level", "V_gas"] = -1 / V_tank
        J["fill_level", "radius"] = inputs["V_gas"] / V_tank**2 * (4 * np.pi * r**2 + 2 * np.pi * r * L)
        J["fill_level", "length"] = inputs["V_gas"] / V_tank**2 * (np.pi * r**2)


class InitialTankStateModification(om.ExplicitComponent):
    """
    Inputs
    ------
    delta_m_gas : float
        Change in mass of the gaseous hydrogen in the tank ullage since the beginning of the mission (vector, kg)
    delta_m_liq : float
        Change in mass of liquid hydrogen in the tank since the beginning of the mission (vector, kg)
    delta_T_gas : float
        Change in temperature of the gaseous hydrogen in the ullage since the beginning of the mission (vector, K)
    delta_T_liq : float
        Change in temperature of the bulk liquid hydrogen since the beginning of the mission (vector, K)
    delta_V_gas : float
        Change in volume of the ullage since the beginning of the mission (vector, m^3)

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
    V_gas : float
        Volume of the ullage (vector, m^3)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    fill_level_init : float
        Initial fill level (in range 0-1) of the tank, default 0.95
        to leave space for boil-off gas; 5% adopted from Millis et al. 2009 (scalar, dimensionless)
    ullage_T_init : float
        Initial temperature of gas in ullage, default 21 K (scalar, K)
    ullage_P_init : float
        Initial pressure of gas in ullage, default 150,000 Pa; ullage pressure must be higher than ambient
        to prevent air leaking in and creating a combustible mixture (scalar, Pa)
    liquid_T_init : float
        Initial temperature of bulk liquid hydrogen, default 20 K (scalar, K)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("fill_level_init", default=0.95, desc="Initial fill level")
        self.options.declare("ullage_T_init", default=21.0, desc="Initial ullage temp (K)")
        self.options.declare("ullage_P_init", default=1.5e5, desc="Initial ullage pressure (Pa)")
        self.options.declare("liquid_T_init", default=20.0, desc="Initial bulk liquid temp (K)")

    def setup(self):
        nn = self.options["num_nodes"]

        r_default = 1.0
        L_default = 0.5
        self.add_input("radius", val=r_default, units="m")
        self.add_input("length", val=L_default, units="m")

        self.add_input("delta_m_gas", shape=(nn,), units="kg", val=0.0)
        self.add_input("delta_m_liq", shape=(nn,), units="kg", val=0.0)
        self.add_input("delta_T_gas", shape=(nn,), units="K", val=0.0)
        self.add_input("delta_T_liq", shape=(nn,), units="K", val=0.0)
        self.add_input("delta_V_gas", shape=(nn,), units="m**3", val=0.0)

        # Get reasonable default values for states
        defaults = self._compute_initial_states(r_default, L_default)
        self.add_output("m_gas", shape=(nn,), units="kg", lower=1e-6, val=defaults["m_gas_init"], upper=1e4)
        self.add_output("m_liq", shape=(nn,), units="kg", lower=1e-2, val=defaults["m_liq_init"], upper=1e6)
        self.add_output("T_gas", shape=(nn,), units="K", lower=15, val=defaults["T_gas_init"], upper=60)
        self.add_output("T_liq", shape=(nn,), units="K", lower=10, val=defaults["T_liq_init"], upper=25)
        self.add_output("V_gas", shape=(nn,), units="m**3", lower=1e-3, val=defaults["V_gas_init"], upper=1e4)

        arng = np.arange(nn)
        self.declare_partials("V_gas", "delta_V_gas", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials("m_liq", "delta_m_liq", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials("m_gas", "delta_m_gas", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials("T_gas", "delta_T_gas", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials("T_liq", "delta_T_liq", rows=arng, cols=arng, val=np.ones(nn))
        self.declare_partials(["V_gas", "m_gas", "m_liq"], ["radius", "length"], rows=arng, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        init_states = self._compute_initial_states(inputs["radius"], inputs["length"])

        outputs["V_gas"] = inputs["delta_V_gas"] + init_states["V_gas_init"]
        outputs["m_gas"] = inputs["delta_m_gas"] + init_states["m_gas_init"]
        outputs["m_liq"] = inputs["delta_m_liq"] + init_states["m_liq_init"]
        outputs["T_gas"] = inputs["delta_T_gas"] + init_states["T_gas_init"]
        outputs["T_liq"] = inputs["delta_T_liq"] + init_states["T_liq_init"]

    def compute_partials(self, inputs, partials):
        r = inputs["radius"]
        L = inputs["length"]
        fill_init = self.options["fill_level_init"]
        T_gas_init = self.options["ullage_T_init"]
        T_liq_init = self.options["liquid_T_init"]
        P_init = self.options["ullage_P_init"]

        # Partial derivatives of tank geometry w.r.t. radius and length
        Vtank_r = 4 * np.pi * r**2 + 2 * np.pi * r * L
        Vtank_L = np.pi * r**2

        partials["V_gas", "radius"] = Vtank_r * (1 - fill_init)
        partials["V_gas", "length"] = Vtank_L * (1 - fill_init)

        coeff = P_init / T_gas_init / UNIVERSAL_GAS_CONST * MOLEC_WEIGHT_H2
        partials["m_gas", "radius"] = coeff * partials["V_gas", "radius"]
        partials["m_gas", "length"] = coeff * partials["V_gas", "length"]

        partials["m_liq", "radius"] = (Vtank_r - partials["V_gas", "radius"]) * H2_prop.lh2_rho(T_liq_init)
        partials["m_liq", "length"] = (Vtank_L - partials["V_gas", "length"]) * H2_prop.lh2_rho(T_liq_init)

    def _compute_initial_states(self, radius, length):
        """
        Returns a dictionary with inital state values based on the specified options and tank geometry.
        """
        fill_init = self.options["fill_level_init"]
        T_gas_init = self.options["ullage_T_init"]
        T_liq_init = self.options["liquid_T_init"]
        P_init = self.options["ullage_P_init"]

        V_tank = 4 / 3 * np.pi * radius**3 + np.pi * radius**2 * length

        res = {}
        res["T_liq_init"] = T_liq_init
        res["T_gas_init"] = T_gas_init
        res["V_gas_init"] = V_tank * (1 - fill_init)
        res["m_gas_init"] = P_init / T_gas_init / UNIVERSAL_GAS_CONST * res["V_gas_init"] * MOLEC_WEIGHT_H2
        res["m_liq_init"] = (V_tank - res["V_gas_init"]) * H2_prop.lh2_rho(T_liq_init)

        return res
