"""
Experimental compressible-flow model of the hydrogen-powered PC-12 NGX.

This prototype attempts to model heat-exchanger cooling air with the
ImplicitCompressibleDuct_ExternalHX component defined in
openconcept/thermal/ducts.py. The experimental compressible propulsion-system
assembly is also present as FuelCellPropulsionSystemCompressible in
openconcept/propulsion/systems/simple_fuel_cell_hybrid.py.

The compressible duct and its coupling to the heat exchanger proved highly
sensitive to input values and solver initialization. The approach has not been
used effectively, and its results and correct operation have not been
guaranteed. This example is retained for experimentation and should be treated
as non-operational for reliable analysis or design decisions. Use the
incompressible hydrogen model in PC12NGXHydrogen.py for the established
hydrogen-mission workflow.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import argparse

import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om
from openconcept.utilities import AddSubtractComp, DictIndepVarComp, plot_trajectory
from openconcept.aerodynamics import PolarDrag
from openconcept.propulsion.systems.simple_fuel_cell_hybrid import (
    AdvancedFuelCell,
    BOPAirSystem,
    LH2Evaporator,
)
from openconcept.propulsion import SimpleMotor, SimplePropeller
from openconcept.thermal import CoolantReservoir, HXGroup, LiquidCooledComp, SimpleHose, SimplePump
from openconcept.thermal.ducts import ImplicitCompressibleDuct_ExternalHX
from openconcept.examples.aircraft_data.PC12NGXHydrogen import data as acdata
from openconcept.mission import BasicMission, IntegratorGroup
from openconcept.weights import SingleTurboPropEmptyWeight


class PC12NGXHydrogenModel(IntegratorGroup):
    """Assemble one mission phase of the experimental compressible hydrogen model.

    Couples aircraft aerodynamics and hydrogen propulsion to a liquid-coolant
    loop, heat exchanger, and compressible external cooling duct. The duct/HX
    coupling is experimental and has not been validated for dependable use.
    """

    def initialize(self):
        # Set the number of analysis nodes and identify the current mission phase.
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]


        # Select the takeoff or cruise drag polar for this mission phase.
        if flight_phase not in ["v0v1", "v1v0", "v1vr", "rotate"]:
            cd0_source = "ac|aero|polar|CD0_cruise"
        else:
            cd0_source = "ac|aero|polar|CD0_TO"

        self.add_subsystem(
            "airframe_drag",
            PolarDrag(num_nodes=nn),
            promotes_inputs=[
                "fltcond|CL", 
                "ac|geom|wing|S_ref", 
                "ac|geom|wing|AR", 
                ("CD0", cd0_source), 
                "fltcond|q", 
                ("e", "ac|aero|polar|e")
            ],
        )

        # Combine airframe drag with the opposing force reported by the duct.
        self.promote_add(
            sources=["airframe_drag.drag", "cooling_duct.force.F_net"],
            prom_name="drag",
            factors=[1.0, -1.0],
            vec_size=nn,
            units="N",
        )

        # ============= PROPULSION AND THERMAL SYSTEMS ========================
        # Convert throttle to motor output, calculate propeller thrust, and
        # couple fuel-cell power and cathode airflow to the BOP compressor/turbine.
        self.add_subsystem(
            "motor",
            SimpleMotor(num_nodes=nn, weight_inc=1/5000, efficiency=0.97),
            promotes_inputs=["throttle", ("elec_power_rating", "ac|propulsion|motor|rating")],
        )
        self.add_subsystem(
            "propeller",
            SimplePropeller(num_nodes=nn, num_blades=5),
            promotes_inputs=[
                "fltcond|*", 
                ("power_rating", "ac|propulsion|motor|rating"), 
                ("diameter", "ac|propulsion|propeller|diameter")
            ],
            promotes_outputs=["thrust"],
        )
        self.add_subsystem("fuelcell", AdvancedFuelCell(num_nodes=nn))
        self.add_subsystem("bop", BOPAirSystem(num_nodes=nn), promotes_inputs=["fltcond|*"])

        self.connect("motor.shaft_power_out", "propeller.shaft_power_in")
        self.connect("motor.elec_load", "fuelcell.elec_power_in")
        self.connect("fuelcell.air_flow", "bop.air_flow")
        self.connect("bop.compressor_power_demand", "fuelcell.compressor_power_demand")
        self.connect("bop.turbine_power_recovery", "fuelcell.turbine_power_recovery")

        # Set coolant properties, flow rate, and heat-exchanger channel geometry
        # used by the thermal models in this experimental configuration.
        thermal_params = self.add_subsystem("thermal_params", om.IndepVarComp(), promotes_outputs=["*"])
        thermal_params.add_output("rho_coolant", val=1020.0 * np.ones((nn,)), units="kg/m**3")
        thermal_params.add_output("mdot_coolant_fc", val=80 * np.ones((nn,)), units="kg/s")
        thermal_params.add_output("channel_width", val=1.5, units="mm")
        thermal_params.add_output("channel_height", val=1.5, units="mm")
        thermal_params.add_output("channel_length", val=0.5, units="m")
        thermal_params.add_output("n_parallel", val=2000)

        # Use the hydrogen flow to absorb part of the fuel-cell waste heat, then
        # send the remaining heat load to the transient liquid-cooled stack model.
        self.add_subsystem("evaporator", LH2Evaporator(num_nodes=nn))
        self.connect("fuelcell.heat_output", "evaporator.heat_output")
        self.connect("fuel_flow", "evaporator.fuel_flow")

        self.add_subsystem(
            "fc_thermal", 
            LiquidCooledComp(num_nodes=nn, quasi_steady=False), 
            promotes_inputs=["duration", "channel_*", "n_parallel"]
        )
        self.connect("evaporator.effective_heat_load", "fc_thermal.q_in")
        self.connect("fuelcell.component_weight", "fc_thermal.mass")

        hx_design_vars = [
            "ac|propulsion|thermal|hx|n_wide_cold",
            "ac|propulsion|thermal|hx|n_long_cold",
            "ac|propulsion|thermal|hx|n_tall",
        ]
        self.add_subsystem("hx_fc", HXGroup(num_nodes=nn), promotes_inputs=[(name, name) for name in hx_design_vars])
        self.connect("rho_coolant", "hx_fc.rho_hot")

        # Experimental compressible cooling-air path. The duct state and the HX
        # are tightly coupled; this configuration is sensitive to initialization
        # and has not been validated for reliable analysis.
        self.add_subsystem(
            "cooling_duct",
            ImplicitCompressibleDuct_ExternalHX(num_nodes=nn, cfg=0.95),
            promotes_inputs=[("p_inf", "fltcond|p"), ("T_inf", "fltcond|T"), ("Utrue", "fltcond|Utrue")],
        )
        self.add_subsystem(
            "duct_area_from_hx",
            om.ExecComp(
                [
                    "duct_area_in = 0.4 * hx_frontal_area",
                    "duct_area_out = 0.4 * hx_frontal_area",
                ],
                hx_frontal_area={"val": 1.0, "units": "m**2"},
                duct_area_in={"val": 0.4, "units": "m**2"},
                duct_area_out={"val": 0.4 * np.ones((nn,)), "units": "m**2"},
            ),
        )
        self.connect("hx_fc.frontal_area", "duct_area_from_hx.hx_frontal_area")
        self.connect("duct_area_from_hx.duct_area_in", "cooling_duct.area_1")
        self.connect("duct_area_from_hx.duct_area_out", "cooling_duct.area_nozzle_in")

        # ================= LIQUID-COOLANT LOOP ========================
        # Model coolant pressure losses and pump behavior in the stack/HX loop.
        # These connections form part of the experimental configuration and
        # require further validation.
        self.add_subsystem("fc_hose", SimpleHose(num_nodes=nn))
        self.add_subsystem(
            "fc_coolant_pump",
            SimplePump(num_nodes=nn),
            promotes_inputs=[("power_rating", "ac|propulsion|thermal|hx_fc|pump_power_rating")],
        )

        self.promote_add(
            sources=["fc_hose.delta_p", "hx_fc.delta_p_hot"],
            prom_name="pressure_drop_fc_loop",
            factors=[1.0, -1.0],
            vec_size=nn,
            units="Pa",
        )
        self.connect("pressure_drop_fc_loop", "fc_coolant_pump.delta_p")
        self.connect("rho_coolant", ["fc_hose.rho_coolant", "fc_coolant_pump.rho_coolant"])

        # Feed compressible-duct station-2 temperature, density, and mass flow
        # to the heat-exchanger cold side.
        self.connect("cooling_duct.sta2.T", "hx_fc.T_in_cold")
        self.connect("cooling_duct.sta2.rho", "hx_fc.rho_cold")
        self.connect("cooling_duct.mdot", "hx_fc.mdot_cold")
        # Feed coolant flow to the stack, heat exchanger, hose, and pump.
        self.connect(
            "mdot_coolant_fc", 
            ["fc_thermal.mdot_coolant", "hx_fc.mdot_hot", "fc_hose.mdot_coolant", "fc_coolant_pump.mdot_coolant"]
        )
        # Return HX pressure loss and heat transfer to the duct, and close the
        # hot-side temperature loop between the stack and heat exchanger.
        self.connect("hx_fc.delta_p_cold", "cooling_duct.sta3.delta_p")
        self.connect("hx_fc.heat_transfer", "cooling_duct.sta3.heat_in")
        self.connect("fc_thermal.T_out", "hx_fc.T_in_hot")
        self.connect("hx_fc.T_out_hot", "fc_thermal.T_in")

        # Integrate fuel flow over the phase, sum installed component masses
        # into OEW, calculate mission weight from takeoff weight minus fuel used,
        # and report the remaining design-mission weight margin.
        self.promote_mult(
            "fuelcell.fuel_flow",
            prom_name="fuel_flow",
            factor=1.0,
            vec_size=nn,
            units="kg/s",
            tags=["integrate", "state_name:fuel_used", "state_units:kg", "state_val:1.0", "state_promotes:True"],
        )

        self.promote_add(
            sources=[
                "hx_fc.component_weight",
                "fc_hose.component_weight",
                "fc_coolant_pump.component_weight",
                "motor.component_weight",
                "fuelcell.component_weight",
                "propeller.component_weight",
            ],
            prom_name="OEW",
            factors=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
            promoted_sources=["ac|design_mission|TOW", "OEW", "fuel_used_final"],
            prom_name="margin",
            factors=[1.0, -1.0, -1.0],
            vec_size=1,
            units="kg",
        )



class PC12NGXHydrogenAnalysisGroup(om.Group):
    """Build the experimental hydrogen aircraft model across all mission phases."""

    def initialize(self):
        # Use the requested number of nodes for each phase in the mission.
        self.options.declare("num_nodes", default=31)

    def setup(self):
        nn = self.options["num_nodes"]

        # Load shared PC-12 NGX aerodynamic, geometric, and weight data, then
        # add propulsion, HX, coolant-pump, and mission-specific inputs.
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
        dv_comp.add_output_from_dict("ac|geom|nosegear|length")
        dv_comp.add_output_from_dict("ac|geom|maingear|length")

        dv_comp.add_output_from_dict("ac|weights|MTOW")
        dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
        dv_comp.add_output_from_dict("ac|weights|MLW")
        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output("ac|propulsion|motor|rating", val=1200.0, units="hp")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx|n_wide_cold")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx|n_long_cold")
        dv_comp.add_output_from_dict("ac|propulsion|thermal|hx|n_tall")
        dv_comp.add_output("ac|propulsion|thermal|hx_fc|pump_power_rating", val=5.0, units="kW")

        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")
        dv_comp.add_output("ac|design_mission|TOW", val=4740.0, units="kg")

        # Construct the ground-roll, climb, cruise, and descent phases using
        # the compressible hydrogen aircraft model.
        self.add_subsystem(
            "analysis",
            BasicMission(num_nodes=nn, aircraft_model=PC12NGXHydrogenModel, include_ground_roll=True),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.set_input_defaults("ac|propulsion|propeller|diameter", val=2.67, units="m")

        # Carry the fuel-cell thermal state continuously from one phase to the next.
        self.connect("groundroll.fc_thermal.T_final", "climb.fc_thermal.T_initial")
        self.connect("climb.fc_thermal.T_final", "cruise.fc_thermal.T_initial")
        self.connect("cruise.fc_thermal.T_final", "descent.fc_thermal.T_initial")


def configure_problem():
    """Create the mission problem and configure its nonlinear solution method.

    Solver convergence for this compressible duct/HX configuration is sensitive
    to initialization; the settings here are experimental, not a guarantee of
    convergence or physically valid results.
    """
    prob = om.Problem()
    prob.model = PC12NGXHydrogenAnalysisGroup()
    prob.model.nonlinear_solver = om.NewtonSolver(
        iprint=2,
        solve_subsystems=True,
        maxiter=20,
    )
    prob.model.nonlinear_solver.options["atol"] = 1e-6
    prob.model.nonlinear_solver.options["rtol"] = 1e-6
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(
        bound_enforcement="scalar",
        print_bound_enforce=False,
    )
    return prob


def set_values(prob, num_nodes):
    """Set flight profiles and initial guesses for the experimental mission.

    The duct station states, nozzle pressure ratio, loss factors, and HX
    properties below are solver starting values. They are particularly
    influential in this sensitive compressible-flow model and require validation.
    """
    # Specify climb, cruise, descent, cruise altitude, range, and takeoff speed.
    prob.set_val("climb.fltcond|vs", np.linspace(1500.0, 100.0, num_nodes), units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.linspace(160, 150, num_nodes), units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 0.01, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.linspace(150, 150, num_nodes), units="kn")
    prob.set_val("descent.fltcond|vs", np.linspace(-750, -100, num_nodes), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.linspace(150, 100, num_nodes), units="kn")
    prob.set_val("cruise|h0", 25000.0, units="ft")
    prob.set_val("mission_range", 500, units="NM")
    prob.set_val("takeoff|v2", 100.0, units="kn")

    phases_list = ["groundroll", "climb", "cruise", "descent"]
    for phase in phases_list:
        prob.set_val(phase + ".fltcond|TempIncrement", 0, units="degC")

        # Provide initial duct station Mach numbers, nozzle pressure ratio, and
        # empirical loss values to help initialize the implicit duct/HX solve.
        # These are numerical starting guesses, not evidence that the selected
        # operating state or compressible-flow solution is physically valid.
        prob.set_val(phase + ".cooling_duct.sta1.M", 0.3)
        prob.set_val(phase + ".cooling_duct.sta2.M", 0.05)
        prob.set_val(phase + ".cooling_duct.sta3.M", 0.05)
        prob.set_val(phase + ".cooling_duct.nozzle.nozzle_pressure_ratio", 0.85)
        prob.set_val(phase + ".cooling_duct.convergence_hack", -10.0, units="Pa")
        prob.set_val(phase + ".cooling_duct.loss_factor_1", 0.20)
        prob.set_val(phase + ".hx_fc.channel_height_hot", 1, units="mm")
        prob.set_val(phase + ".hx_fc.cp_cold", 1002.93, units="J/kg/K")

    prob.set_val("groundroll.cooling_duct.sta1.M", 0.3)
    prob.set_val("groundroll.cooling_duct.nozzle.nozzle_pressure_ratio", 0.85)
    prob.set_val("groundroll.cooling_duct.convergence_hack", -50.0, units="Pa")

    prob.set_val("groundroll.fc_thermal.T_initial", 15.0, units="degC")
    prob.set_val("groundroll.fltcond|Utrue", np.ones((num_nodes,)) * 50, units="kn")
    prob.set_val("ac|propulsion|thermal|hx|n_wide_cold", 700.0)
    prob.set_val("ac|propulsion|thermal|hx|n_long_cold", 30.0)
    prob.set_val("ac|propulsion|thermal|hx|n_tall", 30.0)


def show_outputs(prob):
    """Print fuel, thermal, heat-exchanger, and aircraft-weight results.

    These values are diagnostic outputs from an unvalidated experimental model
    and should not be treated as verified performance predictions.
    """
    phases = ["groundroll", "climb", "cruise", "descent"]
    print("\n==================== PC-12 NGX HYDROGEN COMPRESSIBLE SUMMARY ====================")

    for phase in phases:
        try:
            fuel = prob.get_val(f"{phase}.fuel_used_final", units="lb")
            print(f"{phase.capitalize()} fuel used: {float(np.asarray(fuel).ravel()[-1]):.3f} lb")
        except Exception:
            pass

    for phase in phases:
        try:
            temp = prob.get_val(f"{phase}.fc_thermal.T_final", units="degC")
            print(f"{phase.capitalize()} FC final temp: {float(np.asarray(temp).ravel()[-1]):.2f} degC")
        except Exception:
            pass

        for variable, label in [
            ("fuelcell.heat_output", "FC heat produced"),
            ("fuelcell.elec_power_req", "FC power required"),
        ]:
            try:
                value = prob.get_val(f"{phase}.{variable}", units="kW")
                print(f"{phase.capitalize()} {label}: {float(np.asarray(value).ravel()[-1]):.2f} kW")
            except Exception:
                pass

    print("\n==================== KEY MISSION RESULTS ====================")
    summary_vars = [
        ("descent.fuel_used_final", "Mission fuel used", "lb"),
        ("descent.drag", "Mission final drag", "N"),
        ("descent.hx_fc.frontal_area", "HX frontal area", "m**2"),
        ("descent.cooling_duct.sta3.T", "Cooling duct outlet temp", "degC"),
    ]
    for key, label, units in summary_vars:
        try:
            val = prob.get_val(key, units=units)
            print(f"{label}: {float(np.asarray(val).ravel()[-1]):.4f} {units}")
        except Exception:
            continue

    print("\n==================== THERMAL / HX SUMMARY ====================")
    thermal_vars = [
        ("cruise.fc_thermal.T", "Fuel Cell temp (cruise)", "degC"),
        ("cruise.fuelcell.heat_output", "FC heat produced (cruise)", "kW"),
        ("cruise.fuelcell.elec_power_req", "FC power required (cruise)", "kW"),
        ("cruise.hx_fc.heat_transfer", "HX heat transfer (cruise)", "kW"),
        ("cruise.hx_fc.component_weight", "HX component weight", "kg"),
        ("cruise.hx_fc.height_overall", "HX height", "m"),
        ("cruise.hx_fc.width_overall", "HX width", "m"),
        ("cruise.hx_fc.length_overall", "HX length", "m"),
        ("cruise.cooling_duct.area_1", "Cooling duct inlet area", "m**2"),
        ("cruise.cooling_duct.area_nozzle", "Cooling duct outlet area", "m**2"),
        ("cruise.cooling_duct.mdot", "Cooling air mass flow (cruise)", "kg/s"),
        ("cruise.hx_fc.delta_p_cold", "HX cold-side pressure drop", "Pa"),
    ]
    for key, label, units in thermal_vars:
        try:
            val = prob.get_val(key, units=units)
            print(f"{label}: {float(np.asarray(val).ravel()[-1]):.4f} {units}")
        except Exception:
            continue

    print("\n==================== AIRCRAFT WEIGHT BREAKDOWN ====================")
    weight_vars = [
        ("descent.ac|design_mission|TOW", "Design takeoff weight"),
        ("descent.OEW", "Operating empty weight"),
        ("descent.hx_fc.component_weight", "Heat exchanger weight"),
        ("descent.fc_hose.component_weight", "Coolant hose weight"),
        ("descent.fc_coolant_pump.component_weight", "Coolant pump weight"),
        ("descent.motor.component_weight", "Motor weight"),
        ("descent.fuelcell.component_weight", "Fuel cell weight"),
        ("descent.propeller.component_weight", "Propeller weight"),
        ("descent.fuel_used_final", "Fuel used"),
        ("descent.weight", "Final aircraft weight"),
    ]
    for key, label in weight_vars:
        try:
            value = prob.get_val(key, units="kg")
            print(f"{label}: {float(np.asarray(value).ravel()[-1]):.4f} kg")
        except Exception:
            continue


def get_range_series(prob, phase, var, units=None):
    """Return finite range/value samples for one phase, or None if unavailable."""
    try:
        x = np.asarray(prob.get_val(f"{phase}.range", units="nmi")).ravel()
        if units is None:
            y = np.asarray(prob.get_val(f"{phase}.{var}")).ravel()
        else:
            y = np.asarray(prob.get_val(f"{phase}.{var}", units=units)).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]
    except Exception:
        return None, None


def get_time_series(prob, phase, var, units=None):
    """Return finite time/value samples for one phase using its duration."""
    try:
        t = np.asarray(prob.get_val(f"{phase}.duration", units="min")).ravel()
        if units is None:
            y = np.asarray(prob.get_val(f"{phase}.{var}")).ravel()
        else:
            y = np.asarray(prob.get_val(f"{phase}.{var}", units=units)).ravel()
        if t.size == 1 and y.size > 1:
            t = np.linspace(0.0, float(t[0]), y.size)
        mask = np.isfinite(t) & np.isfinite(y)
        return t[mask], y[mask]
    except Exception:
        return None, None


def get_continuous_time_series(prob, phases, var, units=None):
    """Join phase time histories on a continuous mission-time axis."""
    x_segments = []
    y_segments = []
    elapsed_min = 0.0

    for phase in phases:
        try:
            if units is None:
                y = np.asarray(prob.get_val(f"{phase}.{var}")).ravel()
            else:
                y = np.asarray(prob.get_val(f"{phase}.{var}", units=units)).ravel()

            duration = np.asarray(prob.get_val(f"{phase}.duration", units="min")).ravel()
            if duration.size == 1 and y.size > 1:
                x = np.linspace(0.0, float(duration[0]), y.size)
            else:
                x = np.asarray(duration).ravel()
                if x.size == 1 and y.size > 1:
                    x = np.linspace(0.0, float(x[0]), y.size)

            if x.size != y.size:
                x = np.linspace(0.0, float(np.asarray(duration).ravel()[-1]) if duration.size else 0.0, y.size)

            x = x + elapsed_min
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                x_segments.append(x[mask])
                y_segments.append(y[mask])

            elapsed_min += float(np.asarray(prob.get_val(f"{phase}.duration", units="min")).ravel()[-1])
        except Exception:
            continue

    if not x_segments:
        return None, None

    return np.concatenate(x_segments), np.concatenate(y_segments)


def plot_mission_summary(prob):
    """Create mission, thermal, drag, and aircraft-weight diagnostic figures."""
    phases = ["groundroll", "climb", "cruise", "descent"]

    mission_fig, mission_axs = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    mission_axs = mission_axs.flatten()

    plots = [
        ("fltcond|h", "Altitude", "ft"),
        ("fltcond|Utrue", "True Airspeed", "kn"),
        ("fltcond|vs", "Vertical Speed", "ft/min"),
        ("throttle", "Throttle", None),
        ("fuel_flow", "Fuel Flow", "kg/s"),
        ("drag", "Drag", "N"),
    ]

    for ax, (var, label, units) in zip(mission_axs, plots):
        for phase in phases:
            x, y = get_range_series(prob, phase, var, units)
            if x is not None and y is not None:
                ax.plot(x, y, "-o", markersize=2.0, alpha=0.85)
        ax.set_xlabel("Range (nmi)")
        ax.set_ylabel(label if units is None else f"{label} ({units})")
        ax.grid(True, linestyle="--", alpha=0.5)

    mission_fig.suptitle("Mission Profile (PC-12 NGX H2 Compressible)")
    mission_fig.savefig("pc12ngx_hydrogen_compressible_mission.svg", transparent=True)

    thermal_fig, thermal_axs = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    thermal_axs = thermal_axs.flatten()

    thermal_plots = [
        ("fc_thermal.T", "Fuel Cell Temp", "degC"),
        ("fuelcell.heat_output", "FC Heat Produced", "kW"),
        ("fuelcell.elec_power_req", "FC Power Required", "kW"),
        ("hx_fc.heat_transfer", "HX Heat Transfer", "kW"),
        ("cooling_duct.sta3.T", "Duct Outlet Temp", "degC"),
        ("cooling_duct.mdot", "Cooling Air Mass Flow", "kg/s"),
    ]

    for ax, (var, label, units) in zip(thermal_axs, thermal_plots):
        x_all, y_all = get_continuous_time_series(prob, phases, var, units)
        if x_all is not None and y_all is not None:
            ax.plot(x_all, y_all, "-o", markersize=2.0, alpha=0.85)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{label} ({units})")
        ax.grid(True, linestyle="--", alpha=0.5)

    thermal_fig.suptitle("Thermal Response (PC-12 NGX H2 Compressible)")
    thermal_fig.savefig("pc12ngx_hydrogen_compressible_thermal.svg", transparent=True)

    drag_fig, drag_axs = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, (var, label, units) in zip(drag_axs, [("drag", "Total drag", "N"), ("cooling_duct.force.F_net", "Cooling duct net force", "N")]):
        for phase in phases:
            x, y = get_range_series(prob, phase, var, units)
            if x is not None and y is not None:
                ax.plot(x, y, "-o", markersize=2.0, alpha=0.85)
        ax.set_xlabel("Range (nmi)")
        ax.set_ylabel(f"{label} ({units})")
        ax.grid(True, linestyle="--", alpha=0.5)
    drag_fig.suptitle("Drag Distribution (PC-12 NGX H2 Compressible)")
    drag_fig.savefig("pc12ngx_hydrogen_compressible_drag.svg", transparent=True)

    weight_fig, weight_ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x_all, y_all = get_continuous_time_series(prob, phases, "weight", "kg")
    if x_all is not None and y_all is not None:
        weight_ax.plot(x_all, y_all, "-o", markersize=2.0, alpha=0.85)
    weight_ax.set_xlabel("Time (min)")
    weight_ax.set_ylabel("Aircraft weight (kg)")
    weight_ax.grid(True, linestyle="--", alpha=0.5)
    weight_fig.suptitle("Aircraft Weight During Mission (PC-12 NGX H2 Compressible)")
    weight_fig.savefig("pc12ngx_hydrogen_compressible_weight.svg", transparent=True)


def run_pc_compressible_analisis(plots=True):
    """Run one experimental compressible mission evaluation and optional plots.

    This entry point uses fixed solver initial guesses and is not a validated
    production workflow; successful execution alone does not verify the model.
    """
    num_nodes = 21
    prob = configure_problem()
    prob.setup(check=True, mode="fwd", force_alloc_complex=True)
    set_values(prob, num_nodes)
    phases_list = ["groundroll", "climb", "cruise", "descent"]
    print("=======================================")
    for phase in phases_list:
        if phase != "groundroll":
            # Apply the selected diffuser loss factor to climb, cruise, and descent.
            # This empirical choice is experimental and must be validated for this geometry.
            prob.set_val(phase + ".cooling_duct.loss_factor_1", 0.20)
    prob.run_model()

    if plots:
        show_outputs(prob)
        try:
            plot_mission_summary(prob)
        except Exception as exc:
            print(f"Plot generation warning: {exc}")
    prob.cleanup()
    return prob


# ==============================================================================
# OPTIMIZATION CONFIGURATION AND RESULT HELPERS
# ==============================================================================
def configure_problem(num_nodes: int = 31, temp_limit_c: float = 85.0, maxiter: int = 60, tol: float = 1e-3) -> om.Problem:
    """Configure the mission model, optimizer, and nonlinear/linear solvers."""
    prob = om.Problem()
    prob.model = PC12NGXHydrogenAnalysisGroup(num_nodes=num_nodes)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["maxiter"] = maxiter
    prob.driver.options["tol"] = tol
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["objs", "nl_cons"]

    # Configure Newton's method, an assembled direct linear solve, and a bounded
    # line search. These settings are experimental for the sensitive duct/HX coupling.
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2)
    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

    # Set subsystem solving, iteration count, and residual tolerances.
    prob.model.nonlinear_solver.options["solve_subsystems"] = True
    prob.model.nonlinear_solver.options["maxiter"] = 10
    prob.model.nonlinear_solver.options["atol"] = 1e-3
    prob.model.nonlinear_solver.options["rtol"] = 1e-3

    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar")

    return prob


def configure_drag_minimization_optimization(prob: om.Problem, num_nodes: int, temp_limit_c: float = 85.0) -> None:
    """Configure the experimental HX-geometry drag-minimization problem."""
    # Optimize the three HX geometry variables within these bounds. This
    # optimization is experimental and depends on convergence of the compressible model.
    prob.model.add_design_var("ac|propulsion|thermal|hx|n_wide_cold", lower=300.0, upper=1000.0, scaler=1e-2)
    prob.model.add_design_var("ac|propulsion|thermal|hx|n_long_cold", lower=10.0, upper=500.0)
    prob.model.add_design_var("ac|propulsion|thermal|hx|n_tall", lower=30.0, upper=500.0)

    # Define the objective as mean total drag over climb, cruise, and descent.
    prob.model.add_subsystem(
        "avg_drag_metric",
        om.ExecComp(
            f"drag_metric = (sum(climb_drag) + sum(cruise_drag) + sum(descent_drag)) / {3 * num_nodes}",
            climb_drag={"val": np.ones((num_nodes,)), "units": "N"},
            cruise_drag={"val": np.ones((num_nodes,)), "units": "N"},
            descent_drag={"val": np.ones((num_nodes,)), "units": "N"},
            drag_metric={"val": 1.0, "units": "N"},
        ),
    )
    prob.model.connect("climb.drag", "avg_drag_metric.climb_drag")
    prob.model.connect("cruise.drag", "avg_drag_metric.cruise_drag")
    prob.model.connect("descent.drag", "avg_drag_metric.descent_drag")
    prob.model.add_objective("avg_drag_metric.drag_metric", scaler=1e-3)

    # Constrain fuel-cell temperature at every node in ground roll, climb,
    # cruise, and descent.
    max_temp_k = temp_limit_c + 273.15
    mission_vector_phases = ["groundroll", "climb", "cruise", "descent"]
    for phase_name in mission_vector_phases:
        prob.model.add_constraint(
            f"{phase_name}.fc_thermal.T",
            upper=max_temp_k * np.ones((num_nodes,)),
        )


def get_max_fuel_cell_temperature(prob: om.Problem, num_nodes: int) -> float:
    """Return the highest finite fuel-cell temperature across mission phases."""
    phases = ["groundroll", "climb", "cruise", "descent"]
    tmax = -np.inf
    for phase_name in phases:
        try:
            t_phase_k = np.asarray(prob.get_val(f"{phase_name}.fc_thermal.T")).ravel()
            t_phase = t_phase_k - 273.15
            t_phase = t_phase[np.isfinite(t_phase)]
            if t_phase.size:
                tmax = max(tmax, float(np.max(t_phase)))
        except KeyError:
            continue
    return float(tmax)


def run_optimization(
    num_nodes: int = 31,
    maxiter: int = 60,
    tol: float = 1e-3,
    temp_limit_c: float = 85.0,
    plots: bool = False,
):
    """Run the experimental HX sizing optimization and report diagnostic outputs.

    Optimization completion does not establish that the compressible duct/HX
    model is valid; results must be independently reviewed and validated.
    """
    prob = configure_problem(num_nodes=num_nodes, temp_limit_c=temp_limit_c, maxiter=maxiter, tol=tol)
    configure_drag_minimization_optimization(prob, num_nodes=num_nodes, temp_limit_c=temp_limit_c)
    prob.setup(check=False, mode="fwd")
    set_values(prob, num_nodes)
    prob.run_model()
    prob.run_driver()
    prob.run_model()

    print("\n==================== OPTIMIZATION SUMMARY ====================")
    print(f"Maximum fuel-cell temperature: {get_max_fuel_cell_temperature(prob, num_nodes):.3f} degC")
    drag_metric = np.asarray(prob.get_val("avg_drag_metric.drag_metric")).ravel()[-1]
    print(f"Mission-average drag: {float(drag_metric):.3f} N")

    if plots:
        show_outputs(prob)
        plot_mission_summary(prob)

    return prob


if __name__ == "__main__":
    # Command-line entry point for the experimental optimization. This script
    # remains available for investigation but is not a validated analysis tool.
    parser = argparse.ArgumentParser()
    parser.add_argument("--hide_visuals", action="store_true", help="Do not display plots at the end.")
    parser.add_argument("--num_nodes", type=int, default=31)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--temp_limit_c", type=float, default=85.0)
    args = parser.parse_args()
    run_optimization(
        num_nodes=args.num_nodes,
        maxiter=args.maxiter,
        temp_limit_c=args.temp_limit_c,
        plots=not args.hide_visuals,
    )
    if not args.hide_visuals:
        plt.show()