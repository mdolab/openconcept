import openmdao.api as om
import numpy as np
from openconcept.utilities import Integrator


class LiquidCooledMotor(om.Group):
    """A component (heat producing) with thermal mass
    cooled by a cold plate.

    Inputs
    ------
    q_in : float
        Heat produced by the operating component (vector, W)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    T_in : float
        Instantaneous coolant inflow temperature (vector, K)
    motor_weight : float
        Object mass (only required in thermal mass mode) (scalar, kg)
    T_initial : float
        Initial temperature of the cold plate (only required in thermal mass mode) / object (scalar, K)
    duration : float
        Duration of mission segment, only required in unsteady mode
    power_rating : float
        Rated power of the motor (scalar, kW)

    Outputs
    -------
    T_out : float
        Instantaneous coolant outlet temperature (vector, K)
    T: float
        Windings temperature (vector, K)

    Options
    -------
    motor_specific_heat : float
        Specific heat capacity of the object in J / kg / K (default 921 = aluminum)
    coolant_specific_heat : float
        Specific heat capacity of the coolant in J / kg / K (default 3801, glycol/water)
    num_nodes : int
        Number of analysis points to run
    quasi_steady : bool
        Whether or not to treat the component as having thermal mass
    case_cooling_coefficient : float
        Watts of heat transfer per square meter of case surface area per K
        temperature differential (default 1100 W/m^2/K)
    """

    def initialize(self):
        self.options.declare("motor_specific_heat", default=921.0, desc="Specific heat in J/kg/K")
        self.options.declare("coolant_specific_heat", default=3801, desc="Specific heat in J/kg/K")
        self.options.declare(
            "quasi_steady", default=False, desc="Treat the component as quasi-steady or with thermal mass"
        )
        self.options.declare("num_nodes", default=1, desc="Number of quasi-steady points to runs")
        self.options.declare("case_cooling_coefficient", default=1100.0)

    def setup(self):
        nn = self.options["num_nodes"]
        quasi_steady = self.options["quasi_steady"]
        self.add_subsystem(
            "hex",
            MotorCoolingJacket(
                num_nodes=nn,
                coolant_specific_heat=self.options["coolant_specific_heat"],
                motor_specific_heat=self.options["motor_specific_heat"],
                case_cooling_coefficient=self.options["case_cooling_coefficient"],
            ),
            promotes_inputs=["q_in", "T_in", "T", "power_rating", "mdot_coolant", "motor_weight"],
            promotes_outputs=["T_out", "dTdt"],
        )
        if not quasi_steady:
            ode_integ = self.add_subsystem(
                "ode_integ",
                Integrator(num_nodes=nn, diff_units="s", method="simpson", time_setup="duration"),
                promotes_outputs=["*"],
                promotes_inputs=["*"],
            )
            ode_integ.add_integrand("T", rate_name="dTdt", units="K", lower=1e-10)
        else:
            self.add_subsystem(
                "thermal_bal",
                om.BalanceComp(
                    "T", eq_units="K/s", lhs_name="dTdt", rhs_val=0.0, units="K", lower=1.0, val=299.0 * np.ones((nn,))
                ),
                promotes_inputs=["dTdt"],
                promotes_outputs=["T"],
            )


class MotorCoolingJacket(om.ExplicitComponent):
    """
    Computes motor winding temperature assuming
    well-designed, high-power-density aerospace motor.
    This component is based on the following assumptions:
    - 2020 technology level
    - 200kW-1MW class inrunner PM motor
    - Liquid cooling of the stators
    - "Reasonable" coolant flow rates (component will validate this)
    - Thermal performance similiar to the Siemens SP200D motor

    The component assumes a constant heat transfer coefficient based
    on the surface area of the motor casing (not counting front and rear faces)
    The MagniX Magni 250/500 and Siemens SP200D motors were measured
    using rough photogrammetry.

    Magni250: 280kW rated power, ~0.559m OD, 0.2m case "depth" (along thrust axis)
    Magni500: 560kW rated power, ~0.652m OD, 0.4m case "depth"
    Siemens SP200D: 200kW rated power, ~0.63m OD, ~0.16 case "depth"

    Based on these dimensions I assume 650kW per square meter
    of casing surface area. This includes only the cylindrical portion,
    not the front and rear motor faces.

    Using a thermal FEM image of the SP200D, I estimate
    a temperature rise of 23K from coolant inlet temperature (~85C)
    to winding max temp (~108C) at the steady state operating point.
    With 95% efficiency at 200kW, this is about 1373 W / m^2 casing area / K.
    We'll reduce that somewhat since this is a direct oil cooling system,
    and assume 1100 W/m^2/K instead.

    Dividing 1.1 kW/m^2/K by 650kWrated/m^2 gives: 1.69e-3 kW / kWrated / K
    At full rated power and 95% efficiency, this is 29.5C steady state temp rise
    which the right order of magnitude.

    .. note::
        See the ``LiquidCooledMotor`` for a group that already integrates
        this component with an electric motor.

    Inputs
    ------
    q_in : float
        Heat production rate in the motor (vector, W)
    T_in : float
        Coolant inlet temperature (vector, K)
    T : float
        Temperature of the motor windings (vector, K)
    mdot_coolant : float
        Mass flow rate of the coolant (vector, kg/s)
    power_rating : float
        Rated steady state power of the motor (scalar, W)
    motor_weight : float
        Weight of electric motor (scalar, kg)

    Outputs
    -------
    dTdt : float
        Time derivative dT/dt (vector, K/s)
    q : float
        Heat transfer rate from the motor to the fluid (vector, W)
    T_out : float
        Outlet fluid temperature (vector, K)


    Options
    -------
    num_nodes : float
        The number of analysis points to run
    coolant_specific_heat : float
        Specific heat of the coolant (J/kg/K) (default 3801, glycol/water)
    case_cooling_coefficient : float
        Watts of heat transfer per square meter of case surface area per K
        temperature differential (default 1100 W/m^2/K)
    case_area_coefficient : float
        rated motor power per square meter of case surface area
        (default 650,000 W / m^2)
    motor_specific_heat : float
        Specific heat of the motor casing (J/kg/K) (default 921, alu)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")
        self.options.declare("coolant_specific_heat", default=3801, desc="Specific heat in J/kg/K")
        self.options.declare("case_cooling_coefficient", default=1100.0)
        self.options.declare("case_area_coefficient", default=650000.0)
        self.options.declare(
            "motor_specific_heat", default=921, desc="Specific heat in J/kg/K - default 921 for aluminum"
        )

    def setup(self):
        nn = self.options["num_nodes"]
        arange = np.arange(nn)
        self.add_input("q_in", shape=(nn,), units="W", val=0.0)
        self.add_input("T_in", shape=(nn,), units="K", val=330)
        self.add_input("T", shape=(nn,), units="K", val=359.546)
        self.add_input("mdot_coolant", shape=(nn,), units="kg/s", val=1.0)
        self.add_input("power_rating", units="W", val=2e5)
        self.add_input("motor_weight", units="kg", val=100)
        self.add_output("q", shape=(nn,), units="W")
        self.add_output("T_out", shape=(nn,), units="K", val=300, lower=1e-10)
        self.add_output(
            "dTdt",
            shape=(nn,),
            units="K/s",
            tags=["integrate", "state_name:T_motor", "state_units:K", "state_val:300.0", "state_promotes:True"],
        )

        self.declare_partials(["T_out", "q", "dTdt"], ["power_rating"], rows=arange, cols=np.zeros((nn,)))
        self.declare_partials(["dTdt"], ["motor_weight"], rows=arange, cols=np.zeros((nn,)))

        self.declare_partials(["T_out", "q", "dTdt"], ["T_in", "T", "mdot_coolant"], rows=arange, cols=arange)
        self.declare_partials(["dTdt"], ["q_in"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        const = self.options["case_cooling_coefficient"] / self.options["case_area_coefficient"]

        NTU = const * inputs["power_rating"] / inputs["mdot_coolant"] / self.options["coolant_specific_heat"]
        effectiveness = 1 - np.exp(-NTU)
        heat_transfer = (
            (inputs["T"] - inputs["T_in"])
            * effectiveness
            * inputs["mdot_coolant"]
            * self.options["coolant_specific_heat"]
        )
        outputs["q"] = heat_transfer
        outputs["T_out"] = (
            inputs["T_in"] + heat_transfer / inputs["mdot_coolant"] / self.options["coolant_specific_heat"]
        )
        outputs["dTdt"] = (inputs["q_in"] - outputs["q"]) / inputs["motor_weight"] / self.options["motor_specific_heat"]

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        cp = self.options["coolant_specific_heat"]
        mdot = inputs["mdot_coolant"]
        const = self.options["case_cooling_coefficient"] / self.options["case_area_coefficient"]

        NTU = const * inputs["power_rating"] / mdot / cp
        dNTU_dP = const / mdot / cp
        dNTU_dmdot = -const * inputs["power_rating"] / mdot**2 / cp
        effectiveness = 1 - np.exp(-NTU)
        deff_dP = np.exp(-NTU) * dNTU_dP
        deff_dmdot = np.exp(-NTU) * dNTU_dmdot

        heat_transfer = (
            (inputs["T"] - inputs["T_in"])
            * effectiveness
            * inputs["mdot_coolant"]
            * self.options["coolant_specific_heat"]
        )

        J["q", "T"] = effectiveness * mdot * cp
        J["q", "T_in"] = -effectiveness * mdot * cp
        J["q", "power_rating"] = (inputs["T"] - inputs["T_in"]) * deff_dP * mdot * cp
        J["q", "mdot_coolant"] = (inputs["T"] - inputs["T_in"]) * cp * (effectiveness + deff_dmdot * mdot)

        J["T_out", "T"] = J["q", "T"] / mdot / cp
        J["T_out", "T_in"] = np.ones(nn) + J["q", "T_in"] / mdot / cp
        J["T_out", "power_rating"] = J["q", "power_rating"] / mdot / cp
        J["T_out", "mdot_coolant"] = (J["q", "mdot_coolant"] * mdot - heat_transfer) / cp / mdot**2

        J["dTdt", "q_in"] = 1 / inputs["motor_weight"] / self.options["motor_specific_heat"]
        J["dTdt", "T"] = -J["q", "T"] / inputs["motor_weight"] / self.options["motor_specific_heat"]
        J["dTdt", "T_in"] = -J["q", "T_in"] / inputs["motor_weight"] / self.options["motor_specific_heat"]
        J["dTdt", "power_rating"] = (
            -J["q", "power_rating"] / inputs["motor_weight"] / self.options["motor_specific_heat"]
        )
        J["dTdt", "mdot_coolant"] = (
            -J["q", "mdot_coolant"] / inputs["motor_weight"] / self.options["motor_specific_heat"]
        )
        J["dTdt", "motor_weight"] = (
            -(inputs["q_in"] - heat_transfer) / inputs["motor_weight"] ** 2 / self.options["motor_specific_heat"]
        )
