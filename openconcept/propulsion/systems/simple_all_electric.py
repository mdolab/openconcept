from openconcept.propulsion import SimpleMotor, SimplePropeller
from openconcept.energy_storage import SOCBattery
from openconcept.utilities import DVLabel
from openmdao.api import Group, IndepVarComp
from openconcept.thermal import (
    LiquidCooledComp,
    CoolantReservoir,
    ImplicitCompressibleDuct,
    ExplicitIncompressibleDuct,
    HXGroup,
)

import numpy as np


class AllElectricSinglePropulsionSystemWithThermal_Compressible(Group):
    """This is an example model of the an electric propulsion system
    consisting of a constant-speed prop, motor, and battery.
    Thermal management is provided using a compressible 1D duct
    with heat exchanger.

    Inputs
    ------
    ac|propulsion|motor|rating : float
        The maximum rated continuous shaft power of the motor
    ac|propulsion|propeller|diameter : float
        Diameter of the propeller
    ac|weights|W_battery : float
        Battery weight

    Options
    -------
    num_nodes : float
        Number of analysis points to run (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("specific_energy", default=300, desc="Battery spec energy in Wh/kg")

    def setup(self):
        nn = self.options["num_nodes"]
        e_b = self.options["specific_energy"]
        # rename incoming design variables
        dvlist = [
            ["ac|propulsion|motor|rating", "motor1_rating", 850, "hp"],
            ["ac|propulsion|propeller|diameter", "prop1_diameter", 2.3, "m"],
            ["ac|weights|W_battery", "battery1_weight", 300, "kg"],
        ]
        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])

        # introduce model components
        self.add_subsystem(
            "motor1",
            SimpleMotor(num_nodes=nn, weight_inc=1 / 5000, weight_base=0, efficiency=0.97),
            promotes_inputs=["throttle"],
        )
        self.add_subsystem(
            "prop1",
            SimplePropeller(num_nodes=nn, num_blades=4, design_J=2.2, design_cp=0.55),
            promotes_inputs=["fltcond|*"],
            promotes_outputs=["thrust"],
        )
        self.add_subsystem(
            "batt1", SOCBattery(num_nodes=nn, specific_energy=e_b, efficiency=0.97), promotes_inputs=["duration"]
        )
        # connect design variables to model component inputs
        self.connect("motor1_rating", "motor1.elec_power_rating")
        self.connect("motor1_rating", "prop1.power_rating")
        self.connect("prop1_diameter", "prop1.diameter")
        self.connect("battery1_weight", "batt1.battery_weight")

        # connect components to each other
        self.connect("motor1.shaft_power_out", "prop1.shaft_power_in")
        self.connect("motor1.elec_load", "batt1.elec_load")

        iv = self.add_subsystem("iv", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("mdot_coolant", val=0.1 * np.ones((nn,)), units="kg/s")
        iv.add_output("rho_coolant", val=997 * np.ones((nn,)), units="kg/m**3")
        iv.add_output("coolant_mass", val=10.0, units="kg")

        iv.add_output("channel_width", val=1, units="mm")
        iv.add_output("channel_height", val=20, units="mm")
        iv.add_output("channel_length", val=0.2, units="m")
        iv.add_output("n_parallel", val=50)

        lc_promotes = ["duration", "channel_*", "n_parallel"]
        self.add_subsystem(
            "motorheatsink", LiquidCooledComp(num_nodes=nn, quasi_steady=False), promotes_inputs=lc_promotes
        )
        self.connect("motor1.heat_out", "motorheatsink.q_in")
        self.connect("motor1.component_weight", "motorheatsink.mass")

        self.add_subsystem(
            "duct",
            ImplicitCompressibleDuct(num_nodes=nn),
            promotes_inputs=[("p_inf", "fltcond|p"), ("T_inf", "fltcond|T"), ("Utrue", "fltcond|Utrue")],
        )

        self.connect("motorheatsink.T_out", "duct.T_in_hot")
        self.connect("rho_coolant", "duct.rho_hot")

        self.add_subsystem(
            "reservoir", CoolantReservoir(num_nodes=nn), promotes_inputs=["duration", ("mass", "coolant_mass")]
        )
        self.connect("duct.T_out_hot", "reservoir.T_in")
        self.connect("reservoir.T_out", "motorheatsink.T_in")
        self.connect("mdot_coolant", ["motorheatsink.mdot_coolant", "duct.mdot_hot", "reservoir.mdot_coolant"])


class AllElectricSinglePropulsionSystemWithThermal_Incompressible(Group):
    """This is an example model of the an electric propulsion system
    consisting of a constant-speed prop, motor, and battery.
    Thermal management is provided using a incompressible
    approximation of a 1D duct with heat exchanger.

    Inputs
    ------
    ac|propulsion|motor|rating : float
        The maximum rated continuous shaft power of the motor
    ac|propulsion|propeller|diameter : float
        Diameter of the propeller
    ac|weights|W_battery : float
        Battery weight

    Options
    -------
    num_nodes : float
        Number of analysis points to run (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("specific_energy", default=300, desc="Battery spec energy in Wh/kg")

    def setup(self):
        nn = self.options["num_nodes"]
        e_b = self.options["specific_energy"]
        # rename incoming design variables
        dvlist = [
            ["ac|propulsion|motor|rating", "motor1_rating", 850, "hp"],
            ["ac|propulsion|propeller|diameter", "prop1_diameter", 2.3, "m"],
            ["ac|weights|W_battery", "battery1_weight", 300, "kg"],
        ]
        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])

        # introduce model components
        self.add_subsystem(
            "motor1",
            SimpleMotor(num_nodes=nn, weight_inc=1 / 5000, weight_base=0, efficiency=0.97),
            promotes_inputs=["throttle"],
        )
        self.add_subsystem(
            "prop1",
            SimplePropeller(num_nodes=nn, num_blades=4, design_J=2.2, design_cp=0.55),
            promotes_inputs=["fltcond|*"],
            promotes_outputs=["thrust"],
        )
        self.add_subsystem(
            "batt1", SOCBattery(num_nodes=nn, specific_energy=e_b, efficiency=0.97), promotes_inputs=["duration"]
        )
        # connect design variables to model component inputs
        self.connect("motor1_rating", "motor1.elec_power_rating")
        self.connect("motor1_rating", "prop1.power_rating")
        self.connect("prop1_diameter", "prop1.diameter")
        self.connect("battery1_weight", "batt1.battery_weight")

        # connect components to each other
        self.connect("motor1.shaft_power_out", "prop1.shaft_power_in")
        self.connect("motor1.elec_load", "batt1.elec_load")

        iv = self.add_subsystem("iv", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("mdot_coolant", val=0.1 * np.ones((nn,)), units="kg/s")
        iv.add_output("rho_coolant", val=997 * np.ones((nn,)), units="kg/m**3")
        iv.add_output("coolant_mass", val=10.0, units="kg")

        iv.add_output("channel_width", val=1, units="mm")
        iv.add_output("channel_height", val=20, units="mm")
        iv.add_output("channel_length", val=0.2, units="m")
        iv.add_output("n_parallel", val=50)
        iv.add_output("area_nozzle", val=58 * np.ones((nn,)), units="inch**2")

        lc_promotes = ["duration", "channel_*", "n_parallel"]
        self.add_subsystem(
            "motorheatsink", LiquidCooledComp(num_nodes=nn, quasi_steady=False), promotes_inputs=lc_promotes
        )
        self.connect("motor1.heat_out", "motorheatsink.q_in")
        self.connect("motor1.component_weight", "motorheatsink.mass")
        self.add_subsystem("duct", ExplicitIncompressibleDuct(num_nodes=nn), promotes_inputs=["fltcond|*"])
        self.connect("area_nozzle", "duct.area_nozzle")
        self.add_subsystem(
            "hx", HXGroup(num_nodes=nn), promotes_inputs=[("T_in_cold", "fltcond|T"), ("rho_cold", "fltcond|rho")]
        )
        self.connect("duct.mdot", "hx.mdot_cold")
        self.connect("hx.delta_p_cold", "duct.delta_p_hex")

        self.connect("motorheatsink.T_out", "hx.T_in_hot")
        self.connect("rho_coolant", "hx.rho_hot")

        self.add_subsystem(
            "reservoir", CoolantReservoir(num_nodes=nn), promotes_inputs=["duration", ("mass", "coolant_mass")]
        )
        self.connect("hx.T_out_hot", "reservoir.T_in")
        self.connect("reservoir.T_out", "motorheatsink.T_in")
        self.connect("mdot_coolant", ["motorheatsink.mdot_coolant", "hx.mdot_hot", "reservoir.mdot_coolant"])
