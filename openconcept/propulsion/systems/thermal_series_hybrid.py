from openconcept.propulsion import SimpleMotor, PowerSplit, SimpleGenerator, SimpleTurboshaft, SimplePropeller

# I had to move specific energy into a design variable to get this outer loop to work correctly
from openconcept.energy_storage import SOCBattery
from openconcept.utilities import DVLabel, AddSubtractComp, ElementMultiplyDivideComp
from openconcept.thermal import (
    LiquidCooledComp,
    CoolantReservoir,
    HeatPumpWithIntegratedCoolantLoop,
    ExplicitIncompressibleDuct,
    HXGroup,
)

from openmdao.api import Group, IndepVarComp, BalanceComp
import numpy as np


class TwinSeriesHybridElectricThermalPropulsionSystem(Group):
    """
    This is an example model of a series-hybrid propulsion system. One motor
    draws electrical load from two sources in a fractional split| a battery pack,
    and a turbogenerator setup. The control inputs are the power split fraction and the
    motor throttle setting; the turboshaft throttle matches the power level necessary
    to drive the generator at the required power level.

    Fuel flows and prop thrust should be fairly accurate. Heat constraints haven't yet been incorporated.

    The "pilot" controls thrust by varying the motor throttles from 0 to 100+% of rated power. She may also vary the percentage of battery versus fuel being used
    by varying the power_split_fraction

    This module alone cannot produce accurate fuel flows, battery loads, etc. You must do the following, either with an implicit solver or with the optimizer:
    - Set eng1.throttle such that gen1.elec_power_out = hybrid_split.power_out_A

    The battery does not track its own state of charge (SOC); it is connected to elec_load simply so that the discharge rate can be compared to the discharge rate capability of the battery.
    SOC and fuel flows should be time-integrated at a higher level (in the mission analysis codes)

    Arrows show flow of information. In openConcept, mechanical power operates on a 'push' basis, while electrical load operates on a 'pull' basis. We reconcile these flows across an implicit gap by driving a residual to 0 using a solver.

    .. code::

        eng1.throttle                                                           hybrid_split.power_split_fraction           motor1.throttle
            ||                                                                                   ||                             ||
        eng1 --shaft_power_out--> gen1 --elec_power_out--> {IMPLICIT GAP} <--power_out_B         ||           <--elec_load-- motor1 --shaft_power_out --> prop1 -->thrust
            ||                                                                             hybrid_split <--elec_load  ++
            ||                                            batt1.elec_load <--power_out_A                       <--elec_load-- motor2 --shaft_power_out --> prop2 -->thrust
            V                                                                   V                                              ||
        fuel_flow (integrate over time)                                   elec_load (integrate over time to obtain SOC)       motor2.throttle

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]

        print(nn)

        # define design variables that are independent of flight condition or control states
        dvlist = [
            ["ac|propulsion|engine|rating", "eng_rating", 260.0, "kW"],
            ["ac|propulsion|propeller|diameter", "prop_diameter", 2.5, "m"],
            ["ac|propulsion|motor|rating", "motor_rating", 240.0, "kW"],
            ["ac|propulsion|generator|rating", "gen_rating", 250.0, "kW"],
            ["ac|weights|W_battery", "batt_weight", 2000, "kg"],
            ["ac|propulsion|thermal|hx|mdot_coolant", "mdot_coolant", 0.1 * np.ones((nn,)), "kg/s"],
            ["ac|propulsion|thermal|hx|coolant_mass", "coolant_mass", 10.0, "kg"],
            ["ac|propulsion|thermal|hx|channel_width", "channel_width", 1.0, "mm"],
            ["ac|propulsion|thermal|hx|channel_height", "channel_height", 20.0, "mm"],
            ["ac|propulsion|thermal|hx|channel_length", "channel_length", 0.2, "m"],
            ["ac|propulsion|thermal|hx|n_parallel", "n_parallel", 50, None],
            # ['ac|propulsion|thermal|duct|area_nozzle','area_nozzle',58.*np.ones((nn,)),'inch**2'],
            ["ac|propulsion|battery|specific_energy", "specific_energy", 300, "W*h/kg"],
        ]

        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
        # introduce model components
        self.add_subsystem("motor1", SimpleMotor(efficiency=0.97, num_nodes=nn), promotes_inputs=["throttle"])
        self.add_subsystem("prop1", SimplePropeller(num_nodes=nn), promotes_inputs=["fltcond|*"])
        self.connect("motor1.shaft_power_out", "prop1.shaft_power_in")

        # propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation("motor2throttle", input_names=["throttle", "propulsor_active"], vec_size=nn)
        self.add_subsystem("failedmotor", failedengine, promotes_inputs=["throttle", "propulsor_active"])

        self.add_subsystem("motor2", SimpleMotor(efficiency=0.97, num_nodes=nn))
        self.add_subsystem("prop2", SimplePropeller(num_nodes=nn), promotes_inputs=["fltcond|*"])
        self.connect("motor2.shaft_power_out", "prop2.shaft_power_in")
        self.connect("failedmotor.motor2throttle", "motor2.throttle")

        addpower = AddSubtractComp(
            output_name="motors_elec_load",
            input_names=["motor1_elec_load", "motor2_elec_load"],
            units="kW",
            vec_size=nn,
        )
        addpower.add_equation(
            output_name="thrust", input_names=["prop1_thrust", "prop2_thrust"], units="N", vec_size=nn
        )
        self.add_subsystem("add_power", subsys=addpower, promotes_outputs=["*"])
        self.connect("motor1.elec_load", "add_power.motor1_elec_load")
        self.connect("motor2.elec_load", "add_power.motor2_elec_load")
        self.connect("prop1.thrust", "add_power.prop1_thrust")
        self.connect("prop2.thrust", "add_power.prop2_thrust")

        self.add_subsystem("hybrid_split", PowerSplit(rule="fraction", num_nodes=nn))
        self.connect("motors_elec_load", "hybrid_split.power_in")

        self.add_subsystem(
            "eng1",
            SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
            promotes_outputs=["fuel_flow"],
        )
        self.add_subsystem("gen1", SimpleGenerator(efficiency=0.97, num_nodes=nn))

        self.connect("eng1.shaft_power_out", "gen1.shaft_power_in")

        self.add_subsystem(
            "batt1", SOCBattery(num_nodes=nn, efficiency=0.97), promotes_inputs=["duration", "specific_energy"]
        )
        self.connect("hybrid_split.power_out_A", "batt1.elec_load")
        self.add_subsystem(
            "eng_throttle_set",
            BalanceComp(
                name="eng_throttle",
                val=np.ones((nn,)) * 0.5,
                units=None,
                eq_units="kW",
                rhs_name="gen_power_required",
                lhs_name="gen_power_available",
            ),
        )
        self.connect("hybrid_split.power_out_B", "eng_throttle_set.gen_power_required")
        self.connect("gen1.elec_power_out", "eng_throttle_set.gen_power_available")
        self.connect("eng_throttle_set.eng_throttle", "eng1.throttle")

        adder = AddSubtractComp(output_name="motors_weight", input_names=["motor1_weight", "motor2_weight"], units="kg")
        adder.add_equation(output_name="propellers_weight", input_names=["prop1_weight", "prop2_weight"], units="kg")
        adder.add_equation(
            output_name="motors_heat", input_names=["motor1_heat", "motor2_heat"], vec_size=nn, units="W"
        )
        self.add_subsystem("adder", subsys=adder, promotes_inputs=["*"], promotes_outputs=["*"])
        relabel = [["hybrid_split_A_in", "battery_load", np.ones(nn) * 260.0, "kW"]]
        self.add_subsystem("relabel", DVLabel(relabel), promotes_outputs=["battery_load"])
        self.connect("hybrid_split.power_out_A", "relabel.hybrid_split_A_in")

        self.connect("motor1.component_weight", "motor1_weight")
        self.connect("motor2.component_weight", "motor2_weight")
        self.connect("prop1.component_weight", "prop1_weight")
        self.connect("prop2.component_weight", "prop2_weight")
        self.connect("motor1.heat_out", "motor1_heat")
        self.connect("motor2.heat_out", "motor2_heat")

        # connect design variables to model component inputs
        self.connect("eng_rating", "eng1.shaft_power_rating")
        self.connect("prop_diameter", ["prop1.diameter", "prop2.diameter"])
        self.connect("motor_rating", ["motor1.elec_power_rating", "motor2.elec_power_rating"])
        self.connect("motor_rating", ["prop1.power_rating", "prop2.power_rating"])
        self.connect("gen_rating", "gen1.elec_power_rating")
        self.connect("batt_weight", "batt1.battery_weight")
        iv = self.add_subsystem("iv", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("rho_coolant", val=997 * np.ones((nn,)), units="kg/m**3")
        lc_promotes = ["duration", "channel_*", "n_parallel"]

        self.add_subsystem(
            "batteryheatsink", LiquidCooledComp(num_nodes=nn, quasi_steady=False), promotes_inputs=lc_promotes
        )
        self.connect("batt1.heat_out", "batteryheatsink.q_in")
        self.connect("batt_weight", "batteryheatsink.mass")

        self.add_subsystem(
            "motorheatsink", LiquidCooledComp(num_nodes=nn, quasi_steady=False), promotes_inputs=lc_promotes
        )
        self.connect("motors_heat", "motorheatsink.q_in")
        self.connect("motors_weight", "motorheatsink.mass")

        self.add_subsystem("duct", ExplicitIncompressibleDuct(num_nodes=nn), promotes_inputs=["fltcond|*"])
        iv.add_output("ac|propulsion|thermal|duct|area_nozzle", val=58.0 * np.ones((nn,)), units="inch**2")
        self.connect("ac|propulsion|thermal|duct|area_nozzle", "duct.area_nozzle")
        self.add_subsystem(
            "hx",
            HXGroup(num_nodes=nn),
            promotes_inputs=["ac|*", ("T_in_cold", "fltcond|T"), ("rho_cold", "fltcond|rho")],
        )
        self.connect("duct.mdot", "hx.mdot_cold")
        self.connect("hx.delta_p_cold", "duct.delta_p_hex")

        self.connect("motorheatsink.T_out", "hx.T_in_hot")
        self.connect("rho_coolant", "hx.rho_hot")

        self.add_subsystem(
            "reservoir", CoolantReservoir(num_nodes=nn), promotes_inputs=["duration", ("mass", "coolant_mass")]
        )
        self.connect("hx.T_out_hot", "reservoir.T_in")
        self.connect("reservoir.T_out", "batteryheatsink.T_in")
        self.connect("batteryheatsink.T_out", "motorheatsink.T_in")

        self.connect(
            "mdot_coolant",
            ["batteryheatsink.mdot_coolant", "motorheatsink.mdot_coolant", "hx.mdot_hot", "reservoir.mdot_coolant"],
        )


class TwinSeriesHybridElectricThermalPropulsionRefrigerated(Group):
    """
    This is an example model of a series-hybrid propulsion system that uses active
    refrigeration to cool the electrical components. Other than the addition of
    a refrigerator in the coolant loop, this model is identical to
    TwinSeriesHybridElectricPropulsionSystem. One motor draws electrical
    load from two sources in a fractional split| a battery pack, and a
    turbogenerator setup. The control inputs are the power split fraction and the
    motor throttle setting; the turboshaft throttle matches the power level necessary
    to drive the generator at the required power level.

    Fuel flows and prop thrust should be fairly accurate. Heat constraints haven't yet been incorporated.

    The "pilot" controls thrust by varying the motor throttles from 0 to 100+% of rated power. She may also vary the percentage of battery versus fuel being used
    by varying the power_split_fraction

    This module alone cannot produce accurate fuel flows, battery loads, etc. You must do the following, either with an implicit solver or with the optimizer:
    - Set eng1.throttle such that gen1.elec_power_out = hybrid_split.power_out_A

    The battery does not track its own state of charge (SOC); it is connected to elec_load simply so that the discharge rate can be compared to the discharge rate capability of the battery.
    SOC and fuel flows should be time-integrated at a higher level (in the mission analysis codes)

    Arrows show flow of information. In openConcept, mechanical power operates on a 'push' basis, while electrical load operates on a 'pull' basis. We reconcile these flows across an implicit gap by driving a residual to 0 using a solver.

    .. code::

        eng1.throttle                                                           hybrid_split.power_split_fraction           motor1.throttle
            ||                                                                                   ||                             ||
        eng1 --shaft_power_out--> gen1 --elec_power_out--> {IMPLICIT GAP} <--power_out_B         ||           <--elec_load-- motor1 --shaft_power_out --> prop1 -->thrust
            ||                                                                             hybrid_split <--elec_load  ++
            ||                                            batt1.elec_load <--power_out_A                       <--elec_load-- motor2 --shaft_power_out --> prop2 -->thrust
            V                                                                   V                                              ||
        fuel_flow (integrate over time)                                   elec_load (integrate over time to obtain SOC)       motor2.throttle

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]

        # define design variables that are independent of flight condition or control states
        dvlist = [
            ["ac|propulsion|engine|rating", "eng_rating", 260.0, "kW"],
            ["ac|propulsion|propeller|diameter", "prop_diameter", 2.5, "m"],
            ["ac|propulsion|motor|rating", "motor_rating", 240.0, "kW"],
            ["ac|propulsion|generator|rating", "gen_rating", 250.0, "kW"],
            ["ac|weights|W_battery", "batt_weight", 2000, "kg"],
            ["ac|propulsion|thermal|refrig|rating", "refrig_rating", 1.0, "kW"],
            ["ac|propulsion|thermal|refrig|specific_power", "refrig_spec_pow", 200.0, "W/kg"],
            ["ac|propulsion|thermal|refrig|eff", "refrig_eff_factor", 0.4, None],
            ["ac|propulsion|thermal|hx|mdot_coolant", "mdot_coolant", 0.1 * np.ones((nn,)), "kg/s"],
            ["ac|propulsion|thermal|hx|coolant_mass", "coolant_mass", 10.0, "kg"],
            ["ac|propulsion|thermal|hx|channel_width", "channel_width", 1.0, "mm"],
            ["ac|propulsion|thermal|hx|channel_height", "channel_height", 20.0, "mm"],
            ["ac|propulsion|thermal|hx|channel_length", "channel_length", 0.2, "m"],
            ["ac|propulsion|thermal|hx|n_parallel", "n_parallel", 50, None],
            ["ac|propulsion|thermal|duct|area_nozzle", "area_nozzle", 58.0 * np.ones((nn,)), "inch**2"],
            ["ac|propulsion|battery|specific_energy", "specific_energy", 300, "W*h/kg"],
        ]

        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
        # introduce model components
        self.add_subsystem("motor1", SimpleMotor(efficiency=0.97, num_nodes=nn), promotes_inputs=["throttle"])
        self.add_subsystem("prop1", SimplePropeller(num_nodes=nn), promotes_inputs=["fltcond|*"])
        self.connect("motor1.shaft_power_out", "prop1.shaft_power_in")

        # propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation("motor2throttle", input_names=["throttle", "propulsor_active"], vec_size=nn)
        self.add_subsystem("failedmotor", failedengine, promotes_inputs=["throttle", "propulsor_active"])

        self.add_subsystem("motor2", SimpleMotor(efficiency=0.97, num_nodes=nn))
        self.add_subsystem("prop2", SimplePropeller(num_nodes=nn), promotes_inputs=["fltcond|*"])
        self.connect("motor2.shaft_power_out", "prop2.shaft_power_in")
        self.connect("failedmotor.motor2throttle", "motor2.throttle")

        addpower = AddSubtractComp(
            output_name="total_elec_load",
            input_names=["motor1_elec_load", "motor2_elec_load", "refrig_elec_load"],
            units="kW",
            vec_size=nn,
        )
        addpower.add_equation(
            output_name="thrust", input_names=["prop1_thrust", "prop2_thrust"], units="N", vec_size=nn
        )
        self.add_subsystem("add_power", subsys=addpower, promotes_outputs=["*"])
        self.connect("motor1.elec_load", "add_power.motor1_elec_load")
        self.connect("motor2.elec_load", "add_power.motor2_elec_load")
        self.connect("prop1.thrust", "add_power.prop1_thrust")
        self.connect("prop2.thrust", "add_power.prop2_thrust")

        self.add_subsystem("hybrid_split", PowerSplit(rule="fraction", num_nodes=nn))
        self.connect("total_elec_load", "hybrid_split.power_in")

        self.add_subsystem(
            "eng1",
            SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
            promotes_outputs=["fuel_flow"],
        )
        self.add_subsystem("gen1", SimpleGenerator(efficiency=0.97, num_nodes=nn))

        self.connect("eng1.shaft_power_out", "gen1.shaft_power_in")

        self.add_subsystem(
            "batt1", SOCBattery(num_nodes=nn, efficiency=0.97), promotes_inputs=["duration", "specific_energy"]
        )
        self.connect("hybrid_split.power_out_A", "batt1.elec_load")
        # TODO set val= right number of nn
        self.add_subsystem(
            "eng_throttle_set",
            BalanceComp(
                name="eng_throttle",
                val=np.ones((nn,)) * 0.5,
                units=None,
                eq_units="kW",
                rhs_name="gen_power_required",
                lhs_name="gen_power_available",
            ),
        )
        # need to use the optimizer to drive hybrid_split.power_out_B to the same value as gen1.elec_power_out
        self.connect("hybrid_split.power_out_B", "eng_throttle_set.gen_power_required")
        self.connect("gen1.elec_power_out", "eng_throttle_set.gen_power_available")
        self.connect("eng_throttle_set.eng_throttle", "eng1.throttle")

        adder = AddSubtractComp(output_name="motors_weight", input_names=["motor1_weight", "motor2_weight"], units="kg")
        adder.add_equation(output_name="propellers_weight", input_names=["prop1_weight", "prop2_weight"], units="kg")
        adder.add_equation(
            output_name="motors_heat", input_names=["motor1_heat", "motor2_heat"], vec_size=nn, units="W"
        )
        self.add_subsystem("adder", subsys=adder, promotes_inputs=["*"], promotes_outputs=["*"])
        relabel = [["hybrid_split_A_in", "battery_load", np.ones(nn) * 260.0, "kW"]]
        self.add_subsystem("relabel", DVLabel(relabel), promotes_outputs=["battery_load"])
        self.connect("hybrid_split.power_out_A", "relabel.hybrid_split_A_in")

        self.connect("motor1.component_weight", "motor1_weight")
        self.connect("motor2.component_weight", "motor2_weight")
        self.connect("prop1.component_weight", "prop1_weight")
        self.connect("prop2.component_weight", "prop2_weight")
        self.connect("motor1.heat_out", "motor1_heat")
        self.connect("motor2.heat_out", "motor2_heat")

        # connect design variables to model component inputs
        self.connect("eng_rating", "eng1.shaft_power_rating")
        self.connect("prop_diameter", ["prop1.diameter", "prop2.diameter"])
        self.connect("motor_rating", ["motor1.elec_power_rating", "motor2.elec_power_rating"])
        self.connect("motor_rating", ["prop1.power_rating", "prop2.power_rating"])
        self.connect("gen_rating", "gen1.elec_power_rating")
        self.connect("batt_weight", "batt1.battery_weight")

        iv = self.add_subsystem("iv", IndepVarComp(), promotes_outputs=["*"])

        rho_coolant = 997.0  # kg/m^3
        iv.add_output("rho_coolant", val=rho_coolant * np.ones((nn,)), units="kg/m**3")
        lc_promotes = ["duration", "channel_*", "n_parallel"]

        # Add the refrigerator's electrical load to the splitter with the two motors
        # so it pulls power from both the battery and turboshaft at the hybridization ratio.
        # Bypass the refrigeration with refrig.control.bypass_start and refrig.control.bypass_end
        self.add_subsystem("refrig", HeatPumpWithIntegratedCoolantLoop(num_nodes=nn))
        self.connect("refrig.elec_load", "add_power.refrig_elec_load")
        self.connect("refrig_eff_factor", "refrig.eff_factor")
        self.connect("refrig_rating", "refrig.power_rating")
        self.connect("refrig_spec_pow", "refrig.specific_power")

        # Coolant loop on electrical component side (cooling side of refrigerator)
        # ,---> battery ---> motor ---,
        # |                           |
        # '---- refrig cold side <----'
        self.add_subsystem(
            "batteryheatsink", LiquidCooledComp(num_nodes=nn, quasi_steady=False), promotes_inputs=lc_promotes
        )
        self.connect("batt1.heat_out", "batteryheatsink.q_in")
        self.connect("batt_weight", "batteryheatsink.mass")
        self.connect("refrig.T_out_cold", "batteryheatsink.T_in")

        self.add_subsystem(
            "motorheatsink", LiquidCooledComp(num_nodes=nn, quasi_steady=False), promotes_inputs=lc_promotes
        )
        self.connect("motors_heat", "motorheatsink.q_in")
        self.connect("motors_weight", "motorheatsink.mass")
        self.connect("motorheatsink.T_out", "refrig.T_in_cold")
        self.connect("batteryheatsink.T_out", "motorheatsink.T_in")

        self.connect(
            "mdot_coolant", ["batteryheatsink.mdot_coolant", "motorheatsink.mdot_coolant", "refrig.mdot_coolant"]
        )

        # Coolant loop on hot side of refrigerator to reject heat
        # ,----> refrigerator hot side -----,
        # |                                 |
        # '----- heat exchanger/duct <------'
        self.add_subsystem("duct", ExplicitIncompressibleDuct(num_nodes=nn), promotes_inputs=["fltcond|*"])
        self.add_subsystem(
            "hx",
            HXGroup(num_nodes=nn),
            promotes_inputs=["ac|*", ("rho_cold", "fltcond|rho"), ("T_in_cold", "fltcond|T")],
        )
        self.connect("duct.mdot", "hx.mdot_cold")
        self.connect("hx.delta_p_cold", "duct.delta_p_hex")

        self.connect("rho_coolant", "hx.rho_hot")
        self.connect("refrig.T_out_hot", "hx.T_in_hot")
        self.connect("hx.T_out_hot", "refrig.T_in_hot")

        self.connect("mdot_coolant", "hx.mdot_hot")
