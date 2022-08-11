import numpy as np
from openmdao.api import ExplicitComponent, Group, MetaModelStructuredComp, ExecComp
from openconcept.utilities import ElementMultiplyDivideComp
from openconcept.utilities.constants import GRAV_CONST
import warnings


class HeatPipe(Group):
    """
    Model for an ammonia heat pipe. After model has been run, make sure that the
    heat transfer is always less than the q_max output!

    Inputs
    ------
    T_evap : float
        Temperature of connection to evaporator end of heat pipe (vector, degC)
    q : float
        Heat transferred from evaporator side to condenser side by heat pipe (vector, W)
    length : float
        Heat pipe length (scalar, m)
    inner_diam : float
        Inner diameter of heat pipe vapor/wick section (scalar, m)
    n_pipes : float
        Number of heat pipes in parallel; non-integer values are nonphysical but
        maintained for gradient optimization purposes (scalar, dimensionless)
    T_design : float
        Max temperature expected in heat pipe, used to compute weight (scalar, degC)

    Outputs
    -------
    q_max : float
        Maximum heat transfer possible by heat pipes before dry-out (vector, W)
    T_cond : float
        Temperature of connection to condenser end of heat pipe (vector, degC)
    weight : float
        Weight of heat pipe walls, excludes working fluid (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    length_evap : float
        Length of evaporator, default 0.25 m (scalar, m)
    length_cond : float
        Length of condenser, default 0.25 m (scalar, m)
    wall_conduct : float
        Thermal conductivity of wall material, default aluminum 7075 (scalar, W/(m-K))
    wick_thickness : float
        Thickness of internal wick liner in heat pipe, default no wick (scalar, m)
    wick_conduct : float
        Thermal conductivity of wick liner, default 4 (scalar, W/(m-K))
        Note: default is from wick resistance on slide 21 of https://www.youtube.com/watch?v=JnS0ui8Pt64 and backs out
        thermal conductivity using an assumed thickness of ~0.005" (rough estimate based on the x-section picture)
    yield_stress : float
        Yield stress of the heat pipe materia;, default 7075 aluminum (scalar, MPa)
    rho_wall : float
        Density of the wall material, default 7075 aluminum (scalar, kg/m^3)
    stress_safety_factor : float
        Factor of safety for the wall hoop stress, default 4 (scalar, dimensionless)
    theta : float
        Tilt from vertical of heat pipe, default 0 deg; MUST be greater than or equal to 0 less than 90 (scalar, deg)
    q_max_warn : float
        User will be warned if q input exceeds q_max_warn * q_max, default 0.75 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("length_evap", default=0.25, desc="Length of evaporator m")
        self.options.declare("length_cond", default=0.25, desc="Length of condenser m")
        self.options.declare(
            "wall_conduct",
            default=196.0,
            desc="Thermal conductivity of pipe wall material (default aluminum 7075) W/(m-K)",
        )
        self.options.declare("wick_thickness", default=0e-3, desc="Wick thickness in heat pipe m")
        self.options.declare(
            "wick_conduct",
            default=4.0,
            desc="Thermal conductivity of wick material and evaporation/condensation W/(m-K)",
        )
        self.options.declare("yield_stress", default=572.0, desc="Wall yield stress in MPa (default 7075)")
        self.options.declare("rho_wall", default=2810.0, desc="Wall matl density in kg/m3 (default 7075)")
        self.options.declare("stress_safety_factor", default=4.0, desc="FOS on the wall stress")
        self.options.declare("theta", default=0.0, desc="Tilt from vertical deg")
        self.options.declare("q_max_warn", default=0.75, desc="Warning threshold for q exceeding q_max")

    def setup(self):
        nn = self.options["num_nodes"]

        # Scale the heat transfer by the number of pipes
        # Used ExecComp here because multiplying vector and scalar inputs
        self.add_subsystem(
            "heat_divide",
            ExecComp("q_div = q / n_pipes", q_div={"units": "W", "shape": (nn,)}, q={"units": "W", "shape": (nn,)}),
            promotes_inputs=["q", "n_pipes"],
        )

        # Maximum heat transfer and weight
        self.add_subsystem(
            "q_max_calc",
            QMaxHeatPipe(
                num_nodes=nn,
                theta=self.options["theta"],
                yield_stress=self.options["yield_stress"],
                rho_wall=self.options["rho_wall"],
                stress_safety_factor=self.options["stress_safety_factor"],
            ),
            # Assume temp in heat pipe is close to evaporator temp
            promotes_inputs=["inner_diam", "length", ("design_temp", "T_design"), ("temp", "T_evap")],
        )

        # Multiply max heat transfer and weight by number of pipes
        multiply = ElementMultiplyDivideComp()
        multiply.add_equation(
            output_name="weight", input_names=["single_pipe_weight", "n_pipes"], input_units=["kg", None]
        )
        self.add_subsystem("weight_multiplier", multiply, promotes_inputs=["n_pipes"], promotes_outputs=["weight"])
        self.connect("q_max_calc.heat_pipe_weight", "weight_multiplier.single_pipe_weight")
        # Used ExecComp here because multiplying vector and scalar inputs
        self.add_subsystem(
            "q_max_multiplier",
            ExecComp(
                "q_max = single_pipe_q_max * n_pipes",
                q_max={"units": "W", "shape": (nn,)},
                single_pipe_q_max={"units": "W", "shape": (nn,)},
            ),
            promotes_inputs=["n_pipes"],
            promotes_outputs=["q_max"],
        )
        self.connect("q_max_calc.q_max", "q_max_multiplier.single_pipe_q_max")

        # Thermal resistance at current operator condition
        self.add_subsystem(
            "ammonia_surrogate",
            AmmoniaProperties(num_nodes=nn),
            # Assume temp in heat pipe is close to evaporator temp
            promotes_inputs=[("temp", "T_evap")],
        )
        self.add_subsystem(
            "delta_T_calc",
            HeatPipeVaporTempDrop(num_nodes=nn),
            # Assume temp in heat pipe is close to evaporator temp
            promotes_inputs=[("temp", "T_evap"), "inner_diam", "length"],
        )
        self.add_subsystem(
            "resistance",
            HeatPipeThermalResistance(
                num_nodes=nn,
                length_evap=self.options["length_evap"],
                length_cond=self.options["length_cond"],
                wall_conduct=self.options["wall_conduct"],
                wick_thickness=self.options["wick_thickness"],
                wick_conduct=self.options["wick_conduct"],
                vapor_resistance=True,
            ),
            promotes_inputs=["inner_diam"],
        )
        self.connect("ammonia_surrogate.rho_vapor", "delta_T_calc.rho_vapor")
        self.connect("ammonia_surrogate.vapor_pressure", "delta_T_calc.vapor_pressure")
        self.connect("delta_T_calc.delta_T", "resistance.delta_T")
        self.connect(
            "q_max_calc.wall_thickness", "resistance.wall_thickness"
        )  # use wall thickness from hoop stress calc

        # Compute condenser temperature
        self.add_subsystem(
            "cond_temp_calc",
            ExecComp(
                "T_cond = T_evap - q*R",
                T_cond={"units": "degC", "shape": (nn,)},
                T_evap={"units": "degC", "shape": (nn,)},
                q={"units": "W", "shape": (nn,)},
                R={"units": "K/W", "shape": (nn,)},
            ),
            promotes_inputs=["T_evap"],
            promotes_outputs=["T_cond"],
        )
        self.connect("heat_divide.q_div", ["delta_T_calc.q", "resistance.q", "cond_temp_calc.q"])
        self.connect("resistance.thermal_resistance", "cond_temp_calc.R")

        # Warn the user if heat transfer exceeds maximum possible
        self.add_subsystem(
            "q_max_warning",
            QMaxWarning(num_nodes=nn, q_max_warn=self.options["q_max_warn"]),
            promotes_inputs=["q", "q_max"],
        )


class HeatPipeThermalResistance(ExplicitComponent):
    """
    Computes thermal resistance of a heat pipe with metal exterior, wicking liner, and vapor core.

    Inputs
    ------
    inner_diam : float
        Inner diameter of vapor/wick portion of heat pipe (scalar, m)
    wall_thickness : float
        Thickness of outer metallic wall of heat pipe, default 1.25 mm (scalar, m)
    q : float
        Heat transferred through the pipe per unit time, should always be positive;
        must be specified if vapor_resistance = True, otherwise unused (vector, W)
    delta_T : float
        Vapor temperature drop from one end of the heat pipe to the other;
        must be specified if vapor_resistance = True, otherwise unused (vector, degC)

    Outputs
    -------
    thermal_resistance : float
        Effective thermal resistance of heat pipe, takes into account heat entering/exiting through
        pipe and wick boundary radially and traveling axially along the pipe (vector, K/W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    length_evap : float
        Length of evaporator, default 0.25 m (scalar, m)
    length_cond : float
        Length of condenser, default 0.25 m (scalar, m)
    wall_conduct : float
        Thermal conductivity of wall material, default aluminum 7075 (scalar, W/(m-K))
    wick_thickness : float
        Thickness of internal wick liner in heat pipe, default no wick (scalar, m)
    wick_conduct : float
        Thermal conductivity of wick liner, default 4 (scalar, W/(m-K))
        Note: default is from wick resistance on slide 21 of https://www.youtube.com/watch?v=JnS0ui8Pt64 and backs out
        thermal conductivity using an assumed thickness of ~0.005" (rough estimate based on the x-section picture)
    vapor_resistance : bool
        Set to true to include vapor resistance (usually negligible) in calculation, default false. If set to true,
        the q and temp inputs MUST be connected
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("length_evap", default=0.25, desc="Length of evaporator m")
        self.options.declare("length_cond", default=0.25, desc="Length of condenser m")
        self.options.declare(
            "wall_conduct",
            default=196.0,
            desc="Thermal conductivity of pipe wall material (default aluminum 7075) W/(m-K)",
        )
        self.options.declare("wick_thickness", default=0e-3, desc="Wick thickness in heat pipe m")
        self.options.declare(
            "wick_conduct",
            default=4.0,
            desc="Thermal conductivity of wick material and evaporation/condensation W/(m-K)",
        )
        self.options.declare("vapor_resistance", default=False, desc="Include vapor resistance in calculation")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("inner_diam", units="m", val=0.02)
        self.add_input("wall_thickness", val=1.25e-3, units="m")
        self.add_input("q", shape=(nn,), units="W", val=800)
        self.add_input("delta_T", shape=(nn,), units="K", val=0)

        self.add_output("thermal_resistance", shape=(nn,), units="K/W")

        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        # Heat enters radially in through end of pipe (evaporator), along length of pipe, and radially out (condenser)
        # Resistance of evaporator
        log_o_i_mesh = np.log(inputs["inner_diam"] / (inputs["inner_diam"] - 2 * self.options["wick_thickness"]))
        R_mesh_evap = log_o_i_mesh / (2 * np.pi * self.options["length_evap"] * self.options["wall_conduct"])
        log_o_i_pipe = np.log((inputs["inner_diam"] + 2 * inputs["wall_thickness"]) / inputs["inner_diam"])
        R_pipe_evap = log_o_i_pipe / (2 * np.pi * self.options["length_evap"] * self.options["wall_conduct"])
        R_evap = R_mesh_evap + R_pipe_evap

        # Resistance of condenser
        log_o_i_mesh = np.log(inputs["inner_diam"] / (inputs["inner_diam"] - 2 * self.options["wick_thickness"]))
        R_mesh_cond = log_o_i_mesh / (2 * np.pi * self.options["length_cond"] * self.options["wall_conduct"])
        log_o_i_pipe = np.log((inputs["inner_diam"] + 2 * inputs["wall_thickness"]) / inputs["inner_diam"])
        R_pipe_cond = log_o_i_pipe / (2 * np.pi * self.options["length_cond"] * self.options["wall_conduct"])
        R_cond = R_mesh_cond + R_pipe_cond

        # Resistance of axial component in vapor
        R_vapor_axial = 0
        if self.options["vapor_resistance"]:
            R_vapor_axial = inputs["delta_T"] / inputs["q"]

        # Combine
        outputs["thermal_resistance"] = R_evap + R_cond + R_vapor_axial


class HeatPipeVaporTempDrop(ExplicitComponent):
    """
    Inputs the vapor space temp drop due to pressure drop

    This component is hard-coded on ammonia and uses a curve
    fit of the slope of the temp-pressure curve applicable from -10C to 100C
    operating temperatures

    Inputs
    ------
    q : float
        Heat transfer in the heat pipe (vector, W)
    temp : float
        Mean temp of the heat pipe (vector, degC)
    rho_vapor : float
        Vapor density (vector, kg/m3)
    vapor_pressure : float
        Vapor pressure (vector, Pa)
    inner_diam : float
        Inner diameter of the heat pipe (scalar, m)
    length : float
        Length of the heat pipe (scalar, m)

    Outputs
    -------
    delta_T : float
        Temperature differential across the vapor phase (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run, default 1 (scalar, dimensionless)
    Other options shouldn't be adjusted since they're for ammonia and there is a
    hardcoded curve fit also for ammonia in the compute method
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("latent_heat", default=1371.2e3, desc="Latent heat of vaporization J/kg")
        self.options.declare("visc_base", default=9e-6, desc="Dynamic visc at 0C Pa s")
        self.options.declare("visc_inc", default=4e-8, desc="Dynamic visc slope Pa s / deg")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("q", val=3000 * np.ones((nn,)), units="W")
        self.add_input("temp", val=80 * np.ones((nn,)), units="degC")
        self.add_input("rho_vapor", val=np.ones((nn,)), units="kg/m**3")
        self.add_input("vapor_pressure", val=np.ones((nn,)) * 1e5, units="Pa")
        self.add_input("inner_diam", val=0.01, units="m")
        self.add_input("length", val=6.6, units="m")
        self.add_output("delta_T", val=0.1 * np.ones((nn,)), units="K")
        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        # If the input temperature is out of the range of the fitting data, raise a warning
        if np.any(inputs["temp"] < -10) or np.any(inputs["temp"] > 100):
            warnings.warn(
                self.msginfo
                + f" Heat pipe input temperature of {np.max(inputs['temp'])} deg C is outside of the -10 to 100 deg C "
                "temperature range of the data used for the temp-pressure curve fit. Results may be invalid.",
                stacklevel=2,
            )

        mdot = inputs["q"] / self.options["latent_heat"]
        area = np.pi * inputs["inner_diam"] ** 2 / 4
        mean_vel = mdot / inputs["rho_vapor"] / area
        visc = inputs["temp"] * self.options["visc_inc"] + self.options["visc_base"]
        redh = inputs["rho_vapor"] * mean_vel * inputs["inner_diam"] / visc
        darcy_f = 0.3164 * redh ** (-1 / 4)  # blasius correlation for smooth pipes
        delta_p = inputs["length"] * darcy_f * inputs["rho_vapor"] * mean_vel**2 / 2 / inputs["inner_diam"]
        pressure_kPa = inputs["vapor_pressure"] / 1000

        dt_dp = (
            35.976 / pressure_kPa
        ) / 1000  # based on log curve fit of T vs P curve of ammonia vapor from -10 to 100 C
        outputs["delta_T"] = dt_dp * delta_p


class HeatPipeWeight(ExplicitComponent):
    """
    Computes the weight of a heat pipe neglecting liquid/vapor weight.
    Uses a simple expression for hoop stress times a factor of safety.

    Inputs
    ------
    design_pressure : float
        The maximum design vapor pressure (scalar, MPa)
    inner_diam : float
        Inner diameter of the heat pipe (scalar, m)
    length : float
        Length of the heat pipe (scalar, m)

    Outputs
    -------
    heat_pipe_weight : float
        The material weight of the heat pipe tube (scalar, kg)
    wall_thickness : float
        Thickness of heat pipe walls (scalar, m)

    Options
    -------
    yield_stress : float
        Yield stress of the heat pipe material in MPa
    rho_wall : float
        Density of the wall material in kg/m3
    stress_safety_factor : float
        Factor of safety for the wall hoop stress
    """

    def initialize(self):
        self.options.declare("yield_stress", default=572.0, desc="Wall yield stress in MPa (default 7075)")
        self.options.declare("rho_wall", default=2810.0, desc="Wall matl density in kg/m3 (default 7075)")
        self.options.declare("stress_safety_factor", default=4.0, desc="FOS on the wall stress")

    def setup(self):
        self.add_input("design_pressure", units="MPa", val=1.0)
        self.add_input("inner_diam", units="m", val=0.01)
        self.add_input("length", units="m", val=6.6)
        self.add_output("heat_pipe_weight", units="kg")
        self.add_output("wall_thickness", units="m")

        # it's just as fast to CS this simple comp
        self.declare_partials(["heat_pipe_weight"], ["length", "design_pressure", "inner_diam"])
        self.declare_partials(["wall_thickness"], ["design_pressure", "inner_diam"])

    def compute(self, inputs, outputs):
        sigma_y = self.options["yield_stress"]
        FOS = self.options["stress_safety_factor"]
        rho = self.options["rho_wall"]
        outputs["wall_thickness"] = inputs["design_pressure"] * inputs["inner_diam"] * FOS / sigma_y
        outputs["heat_pipe_weight"] = (
            inputs["length"] * np.pi * inputs["inner_diam"] ** 2 * inputs["design_pressure"] * FOS * rho / sigma_y
        )

    def compute_partials(self, inputs, J):
        sigma_y = self.options["yield_stress"]
        FOS = self.options["stress_safety_factor"]
        rho = self.options["rho_wall"]
        J["wall_thickness", "design_pressure"] = inputs["inner_diam"] * FOS / sigma_y
        J["wall_thickness", "inner_diam"] = inputs["design_pressure"] * FOS / sigma_y
        J["heat_pipe_weight", "design_pressure"] = (
            inputs["length"] * np.pi * inputs["inner_diam"] ** 2 * FOS * rho / sigma_y
        )
        J["heat_pipe_weight", "inner_diam"] = (
            inputs["length"] * np.pi * 2 * inputs["inner_diam"] * inputs["design_pressure"] * FOS * rho / sigma_y
        )
        J["heat_pipe_weight", "length"] = (
            np.pi * inputs["inner_diam"] ** 2 * inputs["design_pressure"] * FOS * rho / sigma_y
        )


class AmmoniaProperties(Group):
    """
    Computes properties of ammonia at liquid-vapor equilibrium as a function of temperature
    using a cubic interpolation of data here: https://en.wikipedia.org/wiki/Ammonia_(data_page)#Vapor%E2%80%93liquid_equilibrium_data

    NOTE: Data is from -75 to 100 deg C, any temps outside this range may be inaccurate

    Inputs
    ------
    temp : float
        Temperature of ammonia liquid/vapor (vector, degC)

    Outputs
    -------
    rho_liquid : float
        Ammonia liquid density (vector, kg/m^3)
    rho_vapor : float
        Ammonia vapor density (vector, kg/m^3)
    vapor_pressure : float
        Ammonia vapor pressure (vector, kPa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run, default 1 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]

        # Surrogate model data from https://en.wikipedia.org/wiki/Ammonia_(data_page)#Vapor%E2%80%93liquid_equilibrium_data
        temp_avg = np.array(
            [
                -75,
                -70,
                -65,
                -60,
                -55,
                -50,
                -45,
                -40,
                -35,
                -30,
                -25,
                -20,
                -15,
                -10,
                -5,
                0,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                60,
                70,
                80,
                90,
                100,
            ]
        )  # deg C
        rho_liquid = np.array(
            [
                730.94,
                725.27,
                719.53,
                713.78,
                707.91,
                702,
                696.04,
                689.99,
                683.85,
                677.64,
                671.37,
                665.03,
                658.54,
                651.98,
                645.33,
                638.57,
                631.67,
                624.69,
                617.55,
                610.28,
                602.85,
                595.24,
                588.16,
                579.48,
                571.3,
                562.87,
                545.23,
                526.32,
                505.71,
                482.9,
                456.93,
            ]
        )  # kg/m^3
        rho_vapor = np.array(
            [
                0.078241,
                0.11141,
                0.15552,
                0.21321,
                0.28596,
                0.38158,
                0.4994,
                0.64508,
                0.82318,
                1.0386,
                1.2969,
                1.6039,
                1.9659,
                2.3874,
                2.8827,
                3.4528,
                4.1086,
                4.8593,
                5.7153,
                6.6876,
                7.7882,
                9.031,
                10.431,
                12.006,
                13.775,
                15.761,
                20.5,
                26.5,
                34.1,
                43.9,
                56.8,
            ]
        )  # kg/m^3
        vapor_pressure = np.array(
            [
                7.93,
                10.92,
                15.61,
                21.90,
                30.16,
                40.87,
                54.54,
                71.77,
                93.19,
                119.6,
                151.6,
                190.2,
                236.3,
                290.8,
                354.8,
                429.4,
                515.7,
                614.9,
                728.3,
                857.1,
                1003.0,
                1166.0,
                1350.0,
                1554.0,
                1781.0,
                2032.0,
                2613.0,
                3312.0,
                4144.0,
                5123.0,
                6264.0,
            ]
        )  # kPa

        # Set up the surrogate model to determine densities
        interp = MetaModelStructuredComp(method="cubic", extrapolate=True, vec_size=nn)
        interp.add_input("temp", val=40.0, shape=(nn,), units="degC", training_data=temp_avg)
        interp.add_output("rho_liquid", val=579.48, shape=(nn,), units="kg/m**3", training_data=rho_liquid)
        interp.add_output("rho_vapor", val=12.006, shape=(nn,), units="kg/m**3", training_data=rho_vapor)
        interp.add_output("vapor_pressure", val=6000, shape=(nn,), units="kPa", training_data=vapor_pressure)
        self.add_subsystem("surrogate", interp, promotes=["*"])


class QMaxHeatPipe(Group):
    """
    Computes the maximum possible heat transfer rate of an ammonia heat pipe.
    As a rule of thumb, the heat pipe should stay below 75% of this value.

    NOTE: This model uses experimental data to compute the ammonia surface tension
          and liquid/vapor density, so it is invalid for any other working fluid.
          The walls are assumed to be 7075 Al with a factor of safety of 4

    Inputs
    ------
    inner_diam : float
        Inner diameter of heat pipe (scalar, m)
    length : float
        Length of the heat pipe (scalar, m)
    temp : float
        Average temperature in heat pipe (vector, degC)
    design_temp : float
        Max design temperature of the heat pipe (scalar, degC)

    Outputs
    -------
    q_max : float
        Maximum heat transfer possible in heat pipe (vector, W)
    heat_pipe_weight : float
        Weight of heat pipe walls (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run, default 1 (scalar, dimensionless)
    theta : float
        Tilt from vertical, default 0 deg (scalar, deg)
    yield_stress : float
        Yield stress of the heat pipe material in MPa
    rho_wall : float
        Density of the wall material in kg/m3
    stress_safety_factor : float
        Factor of safety for the wall hoop stress
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")
        self.options.declare("theta", default=0.0, desc="Tilt from vertical degrees")
        self.options.declare("yield_stress", default=572.0, desc="Wall yield stress in MPa (default 7075)")
        self.options.declare("rho_wall", default=2810.0, desc="Wall matl density in kg/m3 (default 7075)")
        self.options.declare("stress_safety_factor", default=4.0, desc="FOS on the wall stress")

    def setup(self):
        nn = self.options["num_nodes"]

        # Ammonia properties from surrogate model
        self.add_subsystem(
            "ammonia_current",
            AmmoniaProperties(num_nodes=nn),
            promotes_inputs=["temp"],
            promotes_outputs=["vapor_pressure"],
        )
        self.add_subsystem(
            "ammonia_design",
            AmmoniaProperties(),
            promotes_inputs=[("temp", "design_temp")],
            promotes_outputs=[("vapor_pressure", "design_pressure")],
        )

        # Take in the densities from the surrogate model and use analytical expressions to get Q max
        self.add_subsystem(
            "q_max_calc",
            QMaxAnalyticalPart(num_nodes=nn, theta=self.options["theta"]),
            promotes_inputs=["inner_diam", "temp"],
            promotes_outputs=["q_max"],
        )

        self.add_subsystem(
            "weight_calc",
            HeatPipeWeight(
                yield_stress=self.options["yield_stress"],
                rho_wall=self.options["rho_wall"],
                stress_safety_factor=self.options["stress_safety_factor"],
            ),
            promotes=["*"],
        )

        # Connect surrogate to analytical expressions
        self.connect("ammonia_current.rho_liquid", "q_max_calc.rho_liquid")
        self.connect("ammonia_current.rho_vapor", "q_max_calc.rho_vapor")


class QMaxAnalyticalPart(ExplicitComponent):
    """
    Computes the analytical part of the Q max calculation. For the overall
    Q max calculation, use the QMaxHeatPipe group, not this component.

    Equations from https://www.1-act.com/resources/heat-pipe-performance/.
    Surface tension data from page 16 of http://web.iiar.org/membersonly/PDF/CO/databook_ch2.pdf.
    Both accessed on Aug 9, 2022.

    Inputs
    ------
    inner_diam : float
        Inner diameter of heat pipe (scalar, m)
    temp : float
        Average temperature in heat pipe (vector, K)
    rho_liquid : float
        Density of working fluid in liquid form (vector, m)
    rho_vapor : float
        Density of working fluid in vapor form (vector, m)

    Outputs
    -------
    q_max : float
        Maximum heat transfer possible in heat pipe (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run, default 1 (scalar, dimensionless)
    theta : float
        Tilt from vertical, default 0 deg (scalar, deg)
    latent_heat : float
        Latent heat of vaporization, default ammonia 1,371,200 J/kg (scalar, J/kg)
    surface_tension_base : float
        Surface tension at 0 deg C, default ammonia (in 0-50 deg C range) 0.026 N/m (scalar, N/m)
    surface_tension_incr : float
        Surface tension sensitivity w.r.t. temperature (used for linear estimate),
        default ammonia (in 0-50 deg C range) -2.3e-4 N/m/degC (scalar, N/m/degC)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")
        self.options.declare("theta", default=0.0, desc="Tilt from vertical degrees")
        self.options.declare("latent_heat", default=1371.2e3, desc="Latent heat of vaporization J/kg")
        self.options.declare("surface_tension_base", default=0.026, desc="Surface tension at 0 deg C N/m")
        self.options.declare(
            "surface_tension_incr", default=-2.3e-4, desc="Surface tension derivative w.r.t. temp N/m/degC"
        )

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("inner_diam", units="m", val=0.01)
        self.add_input("temp", units="degC", val=40.0, shape=(nn,))
        self.add_input("rho_liquid", val=579.48, shape=(nn,), units="kg/m**3")
        self.add_input("rho_vapor", val=12.006, shape=(nn,), units="kg/m**3")

        self.add_output("q_max", shape=(nn,), units="W")

        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        rho_L = inputs["rho_liquid"]
        rho_V = inputs["rho_vapor"]
        A_vapor = np.pi / 4 * inputs["inner_diam"] ** 2  # heat pipe cross sectional area
        latent = self.options["latent_heat"]
        th_rad = self.options["theta"] * np.pi / 180  # rad

        # Use linear estimate for surface tension, see p. 16 of http://web.iiar.org/membersonly/PDF/CO/databook_ch2.pdf (accessed Aug 9 2022)
        surface_tension = self.options["surface_tension_base"] + self.options["surface_tension_incr"] * inputs["temp"]

        # Compute Q max using equations from https://www.1-act.com/resources/heat-pipe-performance/ (accessed Aug 9 2022)
        bond_number = inputs["inner_diam"] * np.sqrt(GRAV_CONST / surface_tension * (rho_L - rho_V))
        k_flooding = (rho_L / rho_V) ** 0.14 * np.tanh(bond_number**0.25) ** 2
        q_max_numer = (
            k_flooding
            * A_vapor
            * latent
            * (GRAV_CONST * np.sin(np.pi / 2 - th_rad) * surface_tension * (rho_L - rho_V)) ** 0.25
        )
        q_max_denom = (rho_L**-0.25 + rho_V**-0.25) ** 2
        outputs["q_max"] = q_max_numer / q_max_denom


class QMaxWarning(ExplicitComponent):
    """
    Component to warn user if the heat transfer ever exceeds a specified fraction of Q max.

    Inputs
    ------
    q : float
        Heat transferred from evaporator side to condenser side by heat pipe (vector, W)
    q_max : float
        Maximum heat transfer possible by heat pipes before dry-out (vector, W)

    Outputs
    -------
    None

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    q_max_warn : float
        User will be warned if q input exceeds q_max_warn * q_max, default 0.75 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("q_max_warn", default=0.75, desc="Warning threshold for q exceeding q_max")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("q", shape=(nn,), val=0.0, units="W")
        self.add_input("q_max", shape=(nn,), units="W")

    def compute(self, inputs, outputs):
        q = inputs["q"]
        q_max = inputs["q_max"]
        q_warn = self.options["q_max_warn"]
        if np.any(q > q_warn * q_max):
            warnings.warn(
                self.msginfo + f" Heat pipe is being asked to transfer "
                f"{np.max(q/q_max)*100:2.1f}% of its maximum heat transfer capability. This is {(np.max(q/q_max) - q_warn)*100:2.1f}% over "
                f"the warning threshold of {q_warn*100:2.1f}%.",
                stacklevel=2,
            )
