import numpy as np

import openmdao.api as om
from openconcept.utilities import AddSubtractComp, LinearInterpolator, DictIndepVarComp, plot_trajectory

# imports for the airplane model itself
from openconcept.aerodynamics import PolarDrag
from openconcept.examples.aircraft_data.HybridSingleAisle import data as acdata
from openconcept.examples.aircraft_data.HybridSingleAisle import MotorFaultProtection
from openconcept.mission import BasicMission, IntegratorGroup
from openconcept.propulsion import N3Hybrid, SimpleMotor
from openconcept.energy_storage import SOCBattery
from openconcept.thermal import (
    HeatPumpWithIntegratedCoolantLoop,
    SimpleHose,
    SimplePump,
    ImplicitCompressibleDuct_ExternalHX,
    HXGroup,
    LiquidCooledBattery,
    LiquidCooledMotor,
)


class HybridSingleAisleModel(IntegratorGroup):
    """
    Model for NASA N+3 twin hybrid single aisle study

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]

        # =============AERODYNAMICS======================
        # use a different drag coefficient for takeoff versus cruise
        if flight_phase not in ["v0v1", "v1v0", "v1vr", "rotate"]:
            cd0_source = "ac|aero|polar|CD0_cruise"
        else:
            cd0_source = "ac|aero|polar|CD0_TO"

        self.add_subsystem(
            "airframe_drag",
            PolarDrag(num_nodes=nn),
            promotes_inputs=["fltcond|CL", "ac|geom|*", ("CD0", cd0_source), "fltcond|q", ("e", "ac|aero|polar|e")],
        )
        self.promote_add(
            sources=["airframe_drag.drag", "variable_duct.force.F_net", "motor_duct.force.F_net"],
            prom_name="drag",
            factors=[1.0, -2.0, -2.0],
            vec_size=nn,
            units="N",
        )

        # =============PROPULSION=======================
        # Hybrid propulsion motor (model one side only)
        self.add_subsystem(
            "hybrid_throttle",
            LinearInterpolator(num_nodes=nn, units=None),
            promotes_inputs=[("start_val", "hybrid_throttle_start"), ("end_val", "hybrid_throttle_end")],
        )

        self.add_subsystem(
            "hybrid_motor",
            SimpleMotor(num_nodes=nn, efficiency=0.95),
            promotes_inputs=[("elec_power_rating", "ac|propulsion|motor|rating")],
        )
        self.connect("hybrid_motor.shaft_power_out", "engine.hybrid_power")
        self.connect("hybrid_throttle.vec", "hybrid_motor.throttle")
        # Add a surrogate model for the engine. Inputs are Mach, Alt, Throttle, Hybrid power
        self.add_subsystem("engine", N3Hybrid(num_nodes=nn, plot=False), promotes_inputs=["fltcond|*", "throttle"])

        # double the thrust and fuel flow of the engine and integrate fuel flow
        self.promote_mult("engine.thrust", prom_name="thrust", factor=2.0, vec_size=nn, units="kN")
        self.promote_mult(
            "engine.fuel_flow",
            prom_name="fuel_flow",
            factor=2.0,
            vec_size=nn,
            units="kg/s",
            tags=["integrate", "state_name:fuel_used", "state_units:kg", "state_val:1.0", "state_promotes:True"],
        )
        # Hybrid propulsion battery
        self.add_subsystem(
            "battery",
            SOCBattery(num_nodes=nn, efficiency=0.95, specific_energy=400),
            promotes_inputs=[("battery_weight", "ac|propulsion|battery|weight")],
        )
        self.promote_add(
            sources=[
                "hybrid_motor.elec_load",
                "refrig.elec_load",
                "motorfaultprot.elec_load",
                "battery_coolant_pump.elec_load",
                "motor_coolant_pump.elec_load",
            ],
            prom_name="elec_load",
            factors=[1.0, 1.0, 1.0, 1.0, 1.0],
            vec_size=nn,
            val=1.0,
            units="kW",
        )
        self.connect("elec_load", "battery.elec_load")

        # =============THERMAL======================
        thermal_params = self.add_subsystem("thermal_params", om.IndepVarComp(), promotes_outputs=["*"])
        # properties
        thermal_params.add_output("rho_coolant", val=1020 * np.ones((nn,)), units="kg/m**3")
        # controls
        thermal_params.add_output("mdot_coolant_battery", val=4.8 * np.ones((nn,)), units="kg/s")
        thermal_params.add_output("mdot_coolant_motor", val=1.2 * np.ones((nn,)), units="kg/s")
        # fault protection needs separate cooler because it needs 40C inflow temp at 3gpm
        thermal_params.add_output("mdot_coolant_fault_prot", val=0.19 * np.ones((nn,)), units="kg/s")

        thermal_params.add_output("bypass_heat_pump", val=np.ones((nn,)))
        thermal_params.add_output("variable_duct_nozzle_area_start", val=20, units="inch**2")
        thermal_params.add_output("variable_duct_nozzle_area_end", val=20, units="inch**2")
        thermal_params.add_output("heat_pump_specific_power", val=200.0, units="W/kg")
        thermal_params.add_output("heat_pump_eff_factor", val=0.4, units=None)

        self.add_subsystem(
            "li_battery",
            LinearInterpolator(num_nodes=nn, units="inch**2"),
            promotes_outputs=[("vec", "variable_duct_nozzle_area")],
        )
        self.connect("variable_duct_nozzle_area_start", "li_battery.start_val")
        self.connect("variable_duct_nozzle_area_end", "li_battery.end_val")
        self.add_subsystem(
            "li_motor",
            LinearInterpolator(num_nodes=nn, units="inch**2"),
            promotes_inputs=[
                ("start_val", "ac|propulsion|thermal|hx_motor|nozzle_area"),
                ("end_val", "ac|propulsion|thermal|hx_motor|nozzle_area"),
            ],
            promotes_outputs=[("vec", "motor_duct_area_nozzle_in")],
        )

        hx_design_vars = [
            "ac|propulsion|thermal|hx|n_wide_cold",
            "ac|propulsion|thermal|hx|n_long_cold",
            "ac|propulsion|thermal|hx|n_tall",
        ]

        # ===========MOTOR LOOP=======================

        self.add_subsystem(
            "motorheatsink",
            LiquidCooledMotor(num_nodes=nn, case_cooling_coefficient=2100.0, quasi_steady=False),
            promotes_inputs=[("power_rating", "ac|propulsion|motor|rating")],
        )
        self.connect("hybrid_motor.heat_out", "motorheatsink.q_in")
        self.connect("hybrid_motor.component_weight", "motorheatsink.motor_weight")

        self.add_subsystem("motorfaultprot", MotorFaultProtection(num_nodes=nn))
        self.connect("hybrid_motor.elec_load", "motorfaultprot.motor_power")

        self.add_subsystem(
            "hx_motor",
            HXGroup(num_nodes=nn),
            promotes_inputs=[(a, a.replace("hx", "hx_motor")) for a in hx_design_vars],
        )
        self.connect("rho_coolant", "hx_motor.rho_hot")
        fault_prot_promotes = [
            ("ac|propulsion|thermal|hx|n_wide_cold", "ac|propulsion|thermal|hx_motor|n_wide_cold"),
            ("ac|propulsion|thermal|hx|n_long_cold", "ac|propulsion|thermal|hx_fault_prot|n_long_cold"),
            ("ac|propulsion|thermal|hx|n_tall", "ac|propulsion|thermal|hx_motor|n_tall"),
        ]
        self.add_subsystem("hx_fault_prot", HXGroup(num_nodes=nn), promotes_inputs=fault_prot_promotes)
        self.connect(
            "rho_coolant", ["hx_fault_prot.rho_hot", "motor_hose.rho_coolant", "motor_coolant_pump.rho_coolant"]
        )

        self.add_subsystem(
            "motor_duct",
            ImplicitCompressibleDuct_ExternalHX(num_nodes=nn, cfg=0.90),
            promotes_inputs=[
                ("p_inf", "fltcond|p"),
                ("T_inf", "fltcond|T"),
                ("Utrue", "fltcond|Utrue"),
                ("area_nozzle_in", "motor_duct_area_nozzle_in"),
            ],
        )

        self.add_subsystem(
            "motor_hose",
            SimpleHose(num_nodes=nn),
            promotes_inputs=[
                ("hose_length", "ac|geom|thermal|hx_to_motor_length"),
                ("hose_diameter", "ac|geom|thermal|hx_to_motor_diameter"),
            ],
        )
        self.add_subsystem(
            "motor_coolant_pump",
            SimplePump(num_nodes=nn),
            promotes_inputs=[("power_rating", "ac|propulsion|thermal|hx_motor|pump_power_rating")],
        )
        self.promote_add(
            sources=["motor_hose.delta_p", "hx_motor.delta_p_hot"],
            prom_name="pressure_drop_motor_loop",
            factors=[1.0, -1.0],
            vec_size=nn,
            units="Pa",
        )
        self.connect("pressure_drop_motor_loop", "motor_coolant_pump.delta_p")

        # in to HXGroup:
        self.connect("motor_duct.sta2.T", "hx_fault_prot.T_in_cold")
        self.connect("hx_fault_prot.T_out_cold", "hx_motor.T_in_cold")

        self.connect("motor_duct.sta2.rho", ["hx_motor.rho_cold", "hx_fault_prot.rho_cold"])
        self.connect("motor_duct.mdot", ["hx_motor.mdot_cold", "hx_fault_prot.mdot_cold"])
        self.connect("motorheatsink.T_out", "hx_motor.T_in_hot")
        self.connect("motorfaultprot.T_out", "hx_fault_prot.T_in_hot")
        self.connect(
            "mdot_coolant_motor",
            [
                "motorheatsink.mdot_coolant",
                "hx_motor.mdot_hot",
                "motor_hose.mdot_coolant",
                "motor_coolant_pump.mdot_coolant",
            ],
        )
        self.connect("mdot_coolant_fault_prot", ["motorfaultprot.mdot_coolant", "hx_fault_prot.mdot_hot"])

        # out from HXGroup
        self.connect("hx_motor.frontal_area", ["motor_duct.area_2", "motor_duct.area_3"])
        self.connect("hx_motor.delta_p_cold", "motorfaultprot.delta_p_motor_hx")
        self.connect("hx_fault_prot.delta_p_cold", "motorfaultprot.delta_p_fault_prot_hx")
        self.connect("motorfaultprot.delta_p_stack", "motor_duct.sta3.delta_p")

        self.connect("hx_motor.heat_transfer", "motorfaultprot.heat_transfer_motor_hx")
        self.connect("hx_fault_prot.heat_transfer", "motorfaultprot.heat_transfer_fault_prot_hx")
        self.connect("motorfaultprot.heat_transfer", "motor_duct.sta3.heat_in")

        self.connect("hx_motor.T_out_hot", "motorheatsink.T_in")
        self.connect("hx_fault_prot.T_out_hot", "motorfaultprot.T_in")

        # =========BATTERY LOOP=====================
        self.add_subsystem(
            "batteryheatsink",
            LiquidCooledBattery(num_nodes=nn, quasi_steady=False),
            promotes_inputs=[("battery_weight", "ac|propulsion|battery|weight")],
        )
        self.connect("battery.heat_out", "batteryheatsink.q_in")

        # self.connect('mdot_coolant_battery', ['batteryheatsink.mdot_coolant', 'hx_battery.mdot_hot', 'battery_hose.mdot_coolant', 'battery_coolant_pump.mdot_coolant'])
        self.connect(
            "mdot_coolant_battery",
            [
                "batteryheatsink.mdot_coolant",
                "refrig.mdot_coolant",
                "hx_battery.mdot_hot",
                "battery_hose.mdot_coolant",
                "battery_coolant_pump.mdot_coolant",
            ],
        )

        self.connect("batteryheatsink.T_out", "refrig.T_in_cold")
        self.connect("refrig.T_out_cold", "batteryheatsink.T_in")

        self.add_subsystem("hx_battery", HXGroup(num_nodes=nn), promotes_inputs=hx_design_vars)
        self.connect(
            "rho_coolant", ["hx_battery.rho_hot", "battery_hose.rho_coolant", "battery_coolant_pump.rho_coolant"]
        )

        # Hot side balance param will be set to the cooling duct nozzle area
        self.add_subsystem(
            "refrig",
            HeatPumpWithIntegratedCoolantLoop(num_nodes=nn),
            promotes_inputs=[("power_rating", "ac|propulsion|thermal|heatpump|power_rating")],
        )
        self.connect("heat_pump_eff_factor", "refrig.eff_factor")
        self.connect("heat_pump_specific_power", "refrig.specific_power")

        self.add_subsystem(
            "variable_duct",
            ImplicitCompressibleDuct_ExternalHX(num_nodes=nn, cfg=0.95),
            promotes_inputs=[("p_inf", "fltcond|p"), ("T_inf", "fltcond|T"), ("Utrue", "fltcond|Utrue")],
        )
        self.connect("variable_duct_nozzle_area", "variable_duct.area_nozzle_in")

        self.add_subsystem(
            "battery_hose",
            SimpleHose(num_nodes=nn),
            promotes_inputs=[
                ("hose_length", "ac|geom|thermal|hx_to_battery_length"),
                ("hose_diameter", "ac|geom|thermal|hx_to_battery_diameter"),
            ],
        )
        self.add_subsystem(
            "battery_coolant_pump",
            SimplePump(num_nodes=nn),
            promotes_inputs=[("power_rating", "ac|propulsion|thermal|hx|pump_power_rating")],
        )
        self.promote_add(
            sources=["battery_hose.delta_p", "hx_battery.delta_p_hot"],
            prom_name="pressure_drop_battery_loop",
            factors=[1.0, -1.0],
            vec_size=nn,
            units="Pa",
        )
        self.connect("pressure_drop_battery_loop", "battery_coolant_pump.delta_p")

        # in to HXGroup:
        self.connect("variable_duct.sta2.T", "hx_battery.T_in_cold")
        self.connect("variable_duct.sta2.rho", "hx_battery.rho_cold")
        self.connect("variable_duct.mdot", "hx_battery.mdot_cold")

        # out from HXGroup
        self.connect("hx_battery.frontal_area", ["variable_duct.area_2", "variable_duct.area_3"])
        self.connect("hx_battery.delta_p_cold", "variable_duct.sta3.delta_p")
        self.connect("hx_battery.heat_transfer", "variable_duct.sta3.heat_in")
        self.connect("hx_battery.T_out_hot", "refrig.T_in_hot")
        self.connect("refrig.T_out_hot", "hx_battery.T_in_hot")

        # self.connect('hx_battery.T_out_hot', 'batteryheatsink.T_in')
        # self.connect('batteryheatsink.T_out', 'hx_battery.T_in_hot')

        # =============WEIGHTS======================

        # generally the weights module will be custom to each airplane

        # Motor, Battery, TMS, N+3 weight delta
        self.promote_add(
            sources=[
                "refrig.component_weight",
                "hx_motor.component_weight",
                "hx_battery.component_weight",
                "battery_hose.component_weight",
                "battery_coolant_pump.component_weight",
                "motor_hose.component_weight",
                "motor_coolant_pump.component_weight",
                "hx_fault_prot.component_weight",
                "hybrid_motor.component_weight",
            ],
            promoted_sources=["ac|weights|OEW"],
            prom_name="OEW",
            factors=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],
            vec_size=1,
            units="kg",
        )

        self.add_subsystem(
            "weight",
            AddSubtractComp(
                output_name="weight",
                input_names=["ac|design_mission|TOW", "fuel_used"],
                units="kg",
                vec_size=[1, nn],
                scaling_factors=[1, -1],
            ),
            promotes_inputs=["*"],
            promotes_outputs=["weight"],
        )

        self.promote_add(
            sources=[],
            promoted_sources=["ac|design_mission|TOW", "OEW", "fuel_used_final", "ac|propulsion|battery|weight"],
            prom_name="margin",
            factors=[1.0, -1.0, -1.0, -2.0],
            vec_size=1,
            units="kg",
        )


class HybridSingleAisleAnalysisGroup(om.Group):
    """
    Mission analysis group for N+3 hybrid single aisle
    """

    def setup(self):
        # Define number of analysis points to run pers mission segment
        nn = 21

        # Define a bunch of design varaiables and airplane-specific parameters
        dv_comp = self.add_subsystem("dv_comp", DictIndepVarComp(acdata), promotes_outputs=["*"])
        dv_comp.add_output_from_dict("ac|aero|CLmax_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|e")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_cruise")

        dv_comp.add_output_from_dict("ac|geom|wing|S_ref")
        dv_comp.add_output_from_dict("ac|geom|wing|AR")
        dv_comp.add_output_from_dict("ac|geom|wing|c4sweep")
        dv_comp.add_output_from_dict("ac|geom|wing|taper")
        dv_comp.add_output_from_dict("ac|geom|wing|toverc")
        dv_comp.add_output_from_dict("ac|geom|hstab|S_ref")
        dv_comp.add_output_from_dict("ac|geom|hstab|c4_to_wing_c4")
        dv_comp.add_output_from_dict("ac|geom|vstab|S_ref")
        dv_comp.add_output_from_dict("ac|geom|thermal|hx_to_battery_length")
        dv_comp.add_output_from_dict("ac|geom|thermal|hx_to_battery_diameter")
        dv_comp.add_output_from_dict("ac|geom|thermal|hx_to_motor_length")
        dv_comp.add_output_from_dict("ac|geom|thermal|hx_to_motor_diameter")

        dv_comp.add_output_from_dict("ac|geom|nosegear|length")
        dv_comp.add_output_from_dict("ac|geom|maingear|length")

        dv_comp.add_output_from_dict("ac|weights|MTOW")
        dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
        dv_comp.add_output_from_dict("ac|weights|MLW")
        dv_comp.add_output_from_dict("ac|weights|OEW")

        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output_from_dict("ac|propulsion|motor|rating")
        dv_comp.add_output_from_dict("ac|propulsion|battery|weight")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx|n_wide_cold")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx|n_long_cold")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx|n_tall")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx|pump_power_rating")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx_motor|pump_power_rating")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx_fault_prot|n_long_cold")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx_motor|n_wide_cold")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx_motor|n_long_cold")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx_motor|n_tall")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx_motor|nozzle_area")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|heatpump|power_rating")

        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")
        dv_comp.add_output_from_dict("ac|design_mission|TOW")

        # Run a full mission analysis including takeoff, reserve_, cruise,reserve_ and descereserve_nt
        self.add_subsystem(
            "analysis",
            BasicMission(num_nodes=nn, aircraft_model=HybridSingleAisleModel, include_ground_roll=True),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )


def configure_problem():
    prob = om.Problem()
    prob.model = HybridSingleAisleAnalysisGroup()
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options["maxiter"] = 15
    prob.model.nonlinear_solver.options["atol"] = 5e-8
    prob.model.nonlinear_solver.options["rtol"] = 5e-8
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar", print_bound_enforce=False)
    return prob


def set_values(prob, num_nodes):
    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val("climb.fltcond|vs", np.linspace(2300.0, 600.0, num_nodes), units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.linspace(230, 220, num_nodes), units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 4.0, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.linspace(265, 258, num_nodes), units="kn")
    prob.set_val("descent.fltcond|vs", np.linspace(-600, -150, num_nodes), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.ones((num_nodes,)) * 250, units="kn")
    prob.set_val("cruise|h0", 35000.0, units="ft")
    prob.set_val("mission_range", 800, units="NM")
    prob.set_val("takeoff|v2", 160.0, units="kn")
    phases_list = ["groundroll", "climb", "cruise", "descent"]
    for phase in phases_list:
        prob.set_val(phase + ".hybrid_throttle_start", 0.00)
        prob.set_val(phase + ".hybrid_throttle_end", 0.00)
        prob.set_val(phase + ".fltcond|TempIncrement", 20, units="degC")
        prob.set_val(phase + ".refrig.control.bypass_start", 1.0)
        prob.set_val(phase + ".refrig.control.bypass_end", 1.0)

        for duct_name in ["variable_duct", "motor_duct"]:
            prob.set_val(phase + "." + duct_name + ".area_1", 150, units="inch**2")
            prob.set_val(phase + "." + duct_name + ".sta1.M", 0.8)
            prob.set_val(phase + "." + duct_name + ".sta2.M", 0.05)
            prob.set_val(phase + "." + duct_name + ".sta3.M", 0.05)
            prob.set_val(phase + "." + duct_name + ".nozzle.nozzle_pressure_ratio", 0.95)
            prob.set_val(phase + "." + duct_name + ".convergence_hack", -1.0, units="Pa")
        prob.set_val(phase + ".hx_battery.channel_height_hot", 3, units="mm")
        prob.set_val(phase + ".hx_motor.channel_height_hot", 3, units="mm")
        prob.set_val(phase + ".hx_battery.cp_cold", 1002.93, units="J/kg/K")
        prob.set_val(phase + ".hx_motor.cp_cold", 1002.93, units="J/kg/K")
    for duct_name in ["variable_duct", "motor_duct"]:
        prob.set_val("groundroll." + duct_name + ".sta1.M", 0.2)
        prob.set_val("groundroll." + duct_name + ".nozzle.nozzle_pressure_ratio", 0.85)
        prob.set_val("groundroll." + duct_name + ".convergence_hack", -500, units="Pa")
    # prob.set_val('groundroll.bypass_heat_pump', np.zeros((num_nodes,)))
    # prob.set_val('climb.bypass_heat_pump', np.zeros((num_nodes,)))

    prob.set_val("groundroll.variable_duct_nozzle_area_start", 150, units="inch**2")
    prob.set_val("groundroll.variable_duct_nozzle_area_end", 150, units="inch**2")
    prob.set_val("descent.variable_duct_nozzle_area_start", 20, units="inch**2")
    prob.set_val("descent.variable_duct_nozzle_area_end", 20, units="inch**2")

    prob.set_val("groundroll.hybrid_throttle_start", 1.0)
    prob.set_val("groundroll.hybrid_throttle_end", 1.0)
    prob.set_val("climb.hybrid_throttle_start", 1.0)
    prob.set_val("climb.hybrid_throttle_end", 1.0)
    prob.set_val("cruise.hybrid_throttle_start", 1.0)
    prob.set_val("cruise.hybrid_throttle_end", 1.0)

    prob.set_val("groundroll.motorheatsink.T_initial", 30.0, "degC")
    prob.set_val("groundroll.batteryheatsink.T_initial", 30.0, "degC")
    prob.set_val("groundroll.fltcond|Utrue", np.ones((num_nodes)) * 50, units="kn")


def show_outputs(prob):
    # print some outputs
    vars_list = ["descent.fuel_used_final", "descent.hx_battery.xs_area_cold", "descent.hx_battery.frontal_area"]
    units = ["lb", "inch**2", "inch**2"]
    nice_print_names = ["Block fuel", "Duct HX XS area", "Duct HX Frontal Area"]
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i] + ": " + str(prob.get_val(thing, units=units[i])[0]) + " " + units[i])

    # plot some stuff
    plots = True
    if plots:
        x_var = "range"
        x_unit = "NM"
        y_vars = [
            "fltcond|h",
            "fltcond|Ueas",
            "fuel_used",
            "throttle",
            "fltcond|vs",
            "fltcond|M",
            "fltcond|CL",
            "battery.SOC",
            "motorheatsink.T",
            "batteryheatsink.T",
            "batteryheatsink.T_in",
            "variable_duct.force.F_net",
            "motorheatsink.T_in",
            "motor_duct.force.F_net",
            "hx_fault_prot.T_out_hot",
        ]
        y_units = [
            "ft",
            "kn",
            "lbm",
            None,
            "ft/min",
            None,
            None,
            None,
            "degC",
            "degC",
            "degC",
            "lbf",
            "degC",
            "lbf",
            "degC",
        ]
        x_label = "Range (nmi)"
        y_labels = [
            "Altitude (ft)",
            "Veas airspeed (knots)",
            "Fuel used (lb)",
            "Throttle setting",
            "Vertical speed (ft/min)",
            "Mach number",
            "CL",
            "Batt SOC",
            "Motor Temp",
            "Battery Temp (C)",
            "Battery Coolant Inflow Temp",
            "Batt duct cooling Net Force (lb)",
            "Motor Coolant Inflow Temp",
            "Motor duct cooling Net Force (lb)",
            "Motor fault prot inflow temp (C)",
        ]
        phases = ["groundroll", "climb", "cruise", "descent"]
        plot_trajectory(
            prob,
            x_var,
            x_unit,
            y_vars,
            y_units,
            phases,
            x_label=x_label,
            y_labels=y_labels,
            marker="-",
            plot_title="Hybrid Single Aisle Mission",
        )
    # prob.model.list_outputs()


def run_hybrid_sa_analysis(plots=True):
    num_nodes = 21
    prob = configure_problem()
    prob.setup(check=True, mode="fwd", force_alloc_complex=True)
    set_values(prob, num_nodes)
    phases_list = ["groundroll", "climb", "cruise", "descent"]
    print("=======================================")
    for phase in phases_list:
        if phase != "groundroll":
            # loss factor set per https://apps.dtic.mil/dtic/tr/fulltext/u2/002804.pdf for large area ratio diffuser
            prob.set_val(phase + ".motor_duct.loss_factor_1", 0.20)
            prob.set_val(phase + ".variable_duct.loss_factor_1", 0.20)
    prob.set_val("cruise|h0", 31000.0, units="ft")
    for phase in ["climb", "cruise", "descent"]:
        prob.set_val(phase + ".refrig.control.bypass_start", 0.5)
        prob.set_val(phase + ".refrig.control.bypass_end", 0.5)
    prob.run_model()  # set values and run the model in between to get it to converge
    for phase in ["climb", "cruise", "descent"]:
        prob.set_val(phase + ".refrig.control.bypass_start", 0.0)  # full refrigeration (1 is bypass refrigerator)
        prob.set_val(phase + ".refrig.control.bypass_end", 0.0)
    prob.run_model()

    if plots:
        show_outputs(prob)
    prob.cleanup()
    return prob


def run_hybrid_sa_optimization(plots=True):
    """
    Optimize the sizing of the entire thermal management system to minimize block fuel burn
    """
    num_nodes = 21
    prob = configure_problem()
    prob.model.add_design_var("ac|design_mission|TOW", 50000, 79002, ref0=70000, ref=80000, units="kg")
    prob.model.add_design_var("ac|propulsion|thermal|hx|n_wide_cold", 2, 1500, ref0=750, ref=1500, units=None)
    prob.model.add_design_var("ac|propulsion|thermal|hx|n_long_cold", lower=3.0, upper=75.0, ref0=7, ref=75)
    prob.model.add_design_var("ac|propulsion|thermal|hx_motor|n_wide_cold", 50, 1500, ref0=750, ref=1500, units=None)
    prob.model.add_design_var("ac|propulsion|thermal|hx_motor|n_long_cold", lower=3.0, upper=75.0, ref0=7, ref=75)
    prob.model.add_design_var("ac|propulsion|thermal|hx_motor|nozzle_area", lower=5.0, upper=60.0, ref0=5, ref=60)
    prob.model.add_design_var("ac|propulsion|thermal|hx_motor|n_tall", lower=10.0, upper=25.0, ref0=5, ref=60)
    prob.model.add_design_var("ac|propulsion|thermal|hx_fault_prot|n_long_cold", lower=1.0, upper=4.0, ref0=1, ref=4)
    prob.model.add_design_var("climb.hybrid_throttle_start", lower=0.02, upper=1.0, ref0=0, ref=1)
    prob.model.add_design_var("climb.hybrid_throttle_end", lower=0.02, upper=1.0, ref0=0, ref=1)
    prob.model.add_design_var("cruise.hybrid_throttle_start", lower=0.02, upper=1.0, ref0=0, ref=1)
    prob.model.add_design_var("cruise.hybrid_throttle_end", lower=0.02, upper=1.0, ref0=0, ref=1)
    prob.model.add_design_var("descent.hybrid_throttle_start", lower=0.02, upper=0.3, ref0=0, ref=1)
    prob.model.add_design_var("descent.hybrid_throttle_end", lower=0.02, upper=0.3, ref0=0, ref=1)
    prob.model.add_design_var(
        "ac|propulsion|battery|weight", lower=5000 / 2, upper=25000 / 2, ref0=2000 / 2, ref=15000 / 2
    )
    prob.model.add_constraint("descent.battery.SOC_final", lower=0.05, ref0=0.05, ref=0.07)
    prob.model.add_constraint("descent.hx_battery.width_overall", upper=1.2, ref=1.0)
    prob.model.add_constraint(
        "descent.hx_battery.xs_area_cold", lower=70, upper=300.0, units="inch**2", ref0=70, ref=100
    )
    prob.model.add_constraint("descent.hx_motor.width_overall", upper=0.6, ref=1.0)
    prob.model.add_constraint("descent.hx_motor.height_overall", upper=0.3, ref=1.0)
    prob.model.add_constraint("descent.hx_motor.xs_area_cold", lower=70, upper=300.0, units="inch**2", ref0=70, ref=100)
    prob.model.add_constraint("descent.battery_coolant_pump.component_sizing_margin", indices=[0], upper=1.0)
    prob.model.add_constraint("descent.motor_coolant_pump.component_sizing_margin", indices=[0], upper=1.0)
    prob.model.add_objective("descent.fuel_used_final", ref0=3800.0, ref=4200.0)
    prob.model.add_constraint("descent.margin", lower=20000, ref0=10000, ref=30000)
    prob.model.add_design_var(
        "ac|propulsion|thermal|heatpump|power_rating", lower=0.1, upper=50.0, units="kW", ref0=15.0, ref=50.0
    )
    prob.model.add_design_var(
        "ac|propulsion|thermal|hx|pump_power_rating", lower=0.1, upper=5.0, units="kW", ref0=0.0, ref=5.0
    )
    prob.model.add_design_var(
        "ac|geom|thermal|hx_to_battery_diameter", lower=0.5, upper=2.0, units="inch", ref0=0.0, ref=2.0
    )
    prob.model.add_design_var(
        "ac|propulsion|thermal|hx_motor|pump_power_rating", lower=0.1, upper=5.0, units="kW", ref0=0.0, ref=5.0
    )
    prob.model.add_design_var(
        "ac|geom|thermal|hx_to_motor_diameter", lower=0.5, upper=2.0, units="inch", ref0=0.0, ref=2.0
    )

    for phase in ["climb", "cruise", "descent"]:
        prob.model.add_design_var(phase + ".refrig.control.bypass_start", lower=0.0, upper=1.0, units=None, ref=1.0)
        prob.model.add_design_var(phase + ".refrig.control.bypass_end", lower=0.0, upper=1.0, units=None, ref=1.0)

    for phase in ["groundroll"]:
        prob.model.add_design_var(
            phase + ".variable_duct_nozzle_area_start", lower=5.0, upper=150.0, ref0=148, ref=150, units="inch**2"
        )
        prob.model.add_design_var(
            phase + ".variable_duct_nozzle_area_end", lower=5.0, upper=150.0, ref0=148, ref=150, units="inch**2"
        )
    phases_list = ["climb", "cruise"]
    for phase in phases_list:
        prob.model.add_design_var(
            phase + ".variable_duct_nozzle_area_start", lower=5.0, upper=150.0, ref0=75, ref=150, units="inch**2"
        )
        prob.model.add_design_var(
            phase + ".variable_duct_nozzle_area_end", lower=5.0, upper=150.0, ref0=75, ref=150, units="inch**2"
        )
        prob.model.add_constraint(phase + ".batteryheatsink.T", upper=45, ref0=45, ref=50, units="degC")
        prob.model.add_constraint(phase + ".motorheatsink.T", upper=90, ref0=45, ref=90, units="degC")
        prob.model.add_constraint(phase + ".hx_fault_prot.T_out_hot", upper=50, ref0=45, ref=90, units="degC")

    phases_list = ["descent"]
    for phase in phases_list:
        prob.model.add_design_var(
            phase + ".variable_duct_nozzle_area_start", lower=5.0, upper=150.0, ref0=75, ref=150, units="inch**2"
        )
        prob.model.add_design_var(
            phase + ".variable_duct_nozzle_area_end", lower=5.0, upper=150.0, ref0=75, ref=150, units="inch**2"
        )
        constraintvals = np.ones((num_nodes,)) * 45
        constraintvals[-1] = 35
        prob.model.add_constraint(phase + ".batteryheatsink.T", upper=constraintvals, ref0=35, ref=40, units="degC")

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.opt_settings["limited_memory_max_history"] = 1000
    prob.driver.opt_settings["print_level"] = 1
    prob.driver.options["debug_print"] = ["objs"]  # ,'desvars','nl_cons']

    recorder = om.SqliteRecorder("HSA_Refrig_31kft.sql")
    prob.add_recorder(recorder)
    prob.driver.add_recorder(recorder)

    prob.setup(check=True, mode="fwd", force_alloc_complex=True)
    set_values(prob, num_nodes)
    phases_list = ["groundroll", "climb", "cruise", "descent"]
    print("=======================================")
    for phase in phases_list:
        if phase != "groundroll":
            # loss factor set per https://apps.dtic.mil/dtic/tr/fulltext/u2/002804.pdf for large area ratio diffuser
            prob.set_val(phase + ".motor_duct.loss_factor_1", 0.20)
            prob.set_val(phase + ".variable_duct.loss_factor_1", 0.20)
    prob.set_val("cruise|h0", 31000.0, units="ft")
    for phase in ["climb", "cruise", "descent"]:
        prob.set_val(phase + ".refrig.control.bypass_start", 0.5)
        prob.set_val(phase + ".refrig.control.bypass_end", 0.5)
    prob.run_model()  # set values and run the model in between to get it to converge
    for phase in ["climb", "cruise", "descent"]:
        prob.set_val(phase + ".refrig.control.bypass_start", 0.0)
        prob.set_val(phase + ".refrig.control.bypass_end", 0.0)
    prob.run_driver()

    if plots:
        show_outputs(prob)
    prob.cleanup()
    return prob


if __name__ == "__main__":
    # run_hybrid_sa_analysis(plots=True)
    run_hybrid_sa_optimization(plots=True)
