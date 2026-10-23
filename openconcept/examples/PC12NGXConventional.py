"""
Conventional turboprop model of the real Pilatus PC-12 NGX.

This example represents the aircraft in its conventional, kerosene-powered
configuration and simulates its ground roll, climb, cruise, and descent. It
provides a baseline for comparing aircraft performance, fuel use, and weight
with the hydrogen-powered configuration modeled in PC12NGXHydrogen.py.
"""

# rst Imports (beg)
import numpy as np
import openmdao.api as om

# Components used to assemble the conventional aircraft and mission model.
from openconcept.propulsion.systems.simple_turboprop import PC12NGXPropulsionSystem
from openconcept.aerodynamics import PolarDrag
from openconcept.weights import SingleTurboPropEmptyWeight
from openconcept.mission import BasicMission
from openconcept.examples.aircraft_data.PC12NGXConventional import data as acdata
from openconcept.utilities import AddSubtractComp, Integrator, DictIndepVarComp

# rst Imports (end)


# rst Aircraft (beg)
class PC12NGXConventionalAirplaneModel(om.Group):
    """
    Assemble the conventional PC-12 NGX propulsion, aerodynamic, and weight models.

    BasicMission uses this group for each mission phase. The selected flight
    phase determines which zero-lift drag coefficient is used.
    """

    # rst Options
    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)

    # rst Setup
    def setup(self):
        nn = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]

        # ======================================== Propulsion ========================================
        # rst Propulsion (beg)
        # Add the conventional turboprop model and expose the flight conditions,
        # engine rating, propeller diameter, and throttle needed to calculate thrust.
        self.add_subsystem(
            "propmodel",
            PC12NGXPropulsionSystem(num_nodes=nn),
            promotes_inputs=[
                "fltcond|rho",
                "fltcond|Utrue",
                "ac|propulsion|engine|rating",
                "ac|propulsion|propeller|diameter",
                "throttle",
            ],
            promotes_outputs=["thrust"],
        )
        # Set the Hartzell five-blade propeller speed to the PC-12 NGX maximum
        # of 1,700 rpm for this conventional-aircraft analysis.
        self.set_input_defaults("propmodel.prop1.rpm", val=np.full(nn, 1700.0), units="rpm") 
        # rst Propulsion (end)

        # ======================================== Aerodynamics ========================================
        # rst Aero (beg)
        # Use the takeoff zero-lift drag coefficient during ground roll and
        # takeoff transition phases; use the cruise value during the other phases.
        if flight_phase not in ["v0v1", "v1v0", "v1vr", "rotate"]:
            cd0_source = "ac|aero|polar|CD0_cruise"
        else:
            cd0_source = "ac|aero|polar|CD0_TO"

        self.add_subsystem(
            "drag",
            PolarDrag(num_nodes=nn),
            promotes_inputs=[
                "fltcond|CL",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                ("CD0", cd0_source),
                "fltcond|q",
                ("e", "ac|aero|polar|e"),
            ],
            promotes_outputs=["drag"],
        )
        # rst Aero (end)

        # ======================================== Weights ========================================
        # rst Weight (beg)
        # Estimate conventional operating empty weight (OEW) from the aircraft
        # data and component weights. This is reported as a useful design output;
        # mission weight below is calculated separately from MTOW and fuel burn.
        self.add_subsystem(
            "OEW",
            SingleTurboPropEmptyWeight(),
            promotes_inputs=["*", ("P_TO", "ac|propulsion|engine|rating")],
            promotes_outputs=["OEW"],
        )
        self.connect("propmodel.prop1.component_weight", "W_propeller")
        self.connect("propmodel.eng1.component_weight", "W_engine")

        # Integrate fuel flow over this phase to obtain cumulative fuel consumed.
        intfuel = self.add_subsystem(
            "intfuel",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_outputs=["fuel_used_final"],
        )
        intfuel.add_integrand("fuel_used", rate_name="fuel_flow", val=1.0, units="kg")
        self.connect("propmodel.fuel_flow", "intfuel.fuel_flow")

        # Calculate phase weight by subtracting cumulative fuel burn from MTOW,
        # assuming the aircraft starts the mission at maximum takeoff weight.
        self.add_subsystem(
            "weight",
            AddSubtractComp(
                output_name="weight",
                input_names=["ac|weights|MTOW", "fuel_used"],
                units="kg",
                vec_size=[1, nn],
                scaling_factors=[1, -1],
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["weight"],
        )
        self.connect("intfuel.fuel_used", "weight.fuel_used")
        # rst Weight (end)


# rst Mission (beg)
class PC12NGXConventionalAnalysisGroup(om.Group):
    """
    Define the conventional PC-12 NGX aircraft data and complete mission.

    The mission model includes ground roll, climb, cruise, and descent, and
    instantiates PC12NGXConventionalAirplaneModel for each phase.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=11)

    def setup(self):
        nn = self.options["num_nodes"]

        # Load the conventional PC-12 NGX reference data and promote its outputs
        # so they can be used throughout the aircraft and mission groups.
        dv_comp = self.add_subsystem("dv_comp", DictIndepVarComp(acdata), promotes_outputs=["*"])

        # Aerodynamic limits and drag-polar coefficients.
        dv_comp.add_output_from_dict("ac|aero|CLmax_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|e")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_cruise")

        # Wing, tail, fuselage, and landing-gear dimensions used by the model.
        dv_comp.add_output_from_dict("ac|geom|wing|S_ref")
        dv_comp.add_output_from_dict("ac|geom|wing|AR")
        dv_comp.add_output_from_dict("ac|geom|wing|c4sweep")
        dv_comp.add_output_from_dict("ac|geom|wing|taper")
        dv_comp.add_output_from_dict("ac|geom|wing|toverc")
        dv_comp.add_output_from_dict("ac|geom|hstab|S_ref")
        dv_comp.add_output_from_dict("ac|geom|hstab|c4_to_wing_c4")
        dv_comp.add_output_from_dict("ac|geom|vstab|S_ref")
        dv_comp.add_output_from_dict("ac|geom|fuselage|S_wet")
        dv_comp.add_output_from_dict("ac|geom|fuselage|width")
        dv_comp.add_output_from_dict("ac|geom|fuselage|length")
        dv_comp.add_output_from_dict("ac|geom|fuselage|height")
        dv_comp.add_output_from_dict("ac|geom|nosegear|length")
        dv_comp.add_output_from_dict("ac|geom|maingear|length")

        # Maximum takeoff, fuel, and landing weights.
        dv_comp.add_output_from_dict("ac|weights|MTOW")
        dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
        dv_comp.add_output_from_dict("ac|weights|MLW")

        # Engine rating and propeller diameter for the conventional turboprop.
        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output_from_dict("ac|propulsion|propeller|diameter")

        # Passenger capacity and cruise dynamic pressure.
        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")

        # Build and expose the ground-roll, climb, cruise, and descent phases.
        self.add_subsystem(
            "analysis",
            BasicMission(
                num_nodes=nn,
                aircraft_model=PC12NGXConventionalAirplaneModel,
                include_ground_roll=True,
                include_descent=True,
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        # rst Mission (end)


# rst Setup problem (beg)
def run_pc12ngx_conventional_analysis():
    # Create the OpenMDAO problem, attach the aircraft mission, and configure
    # nonlinear and linear solvers for the coupled phase calculations.
    nn = 11
    prob = om.Problem()
    prob.model = PC12NGXConventionalAnalysisGroup(num_nodes=nn)
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.setup()

    # Set initial flight-condition profiles for each mission phase. The ground
    # roll speed is nonzero so the takeoff model can evaluate its flight dynamics.
    prob.set_val("groundroll.fltcond|Utrue", np.ones((nn,)) * 50, units="kn")
    prob.set_val("climb.fltcond|vs", np.linspace(1500, 100, nn), units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.linspace(150, 150, nn), units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((nn,)) * 0.01, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.linspace(150, 150, nn), units="kn")
    prob.set_val("descent.fltcond|vs", np.linspace(-750, -100, nn), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.linspace(150, 100, nn), units="kn")
    prob.set_val("cruise|h0", 7620, units="m")  # Cruise altitude: 25,000 ft.
    prob.set_val("mission_range", 500, units="NM")  # Target mission range.

    return prob
    # rst Setup problem (end)


# rst Run (beg)
if __name__ == "__main__":
    # Parse the option to suppress interactive figures and the OpenMDAO N2 diagram.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hide_visuals",
        default=False,
        action="store_true",
        help="Do not show matplotlib figure or open N2 diagram in browser",
    )
    hide_viz = parser.parse_args().hide_visuals

    # Build the model and evaluate the complete conventional mission.
    prob = run_pc12ngx_conventional_analysis()
    prob.run_model()

    # Save the model-connection diagram and open it unless visuals are disabled.
    om.n2(prob, outfile="pc12ngx_conventional_n2.html", show_browser=not hide_viz)

    # Select key aircraft, fuel-consumption, and mission outputs for the summary.
    print_vars = [
        {"var": "ac|weights|MTOW", "name": "MTOW", "units": "lb"},
        {"var": "climb.OEW", "name": "OEW", "units": "lb"},
        {"var": "groundroll.fuel_used_final", "name": "Ground roll fuel", "units": "lb"},
        {"var": "climb.fuel_used_final", "name": "Climb fuel", "units": "lb"},
        {"var": "cruise.fuel_used_final", "name": "Cruise fuel", "units": "lb"},
        {"var": "descent.fuel_used_final", "name": "Fuel used", "units": "lb"},
    ]

    flight_duration = (
        prob.get_val("groundroll.duration", units="min").item()
        + prob.get_val("climb.duration", units="min").item()
        + prob.get_val("cruise.duration", units="min").item()
        + prob.get_val("descent.duration", units="min").item()
    )

    # Find the first available phase-level value for each requested weight.
    # Names are checked from the end of the mission backward because OpenMDAO
    # can expose the same mission quantity under different phase paths.
    def get_weight_value(candidates):
        for candidate in candidates:
            try:
                value = np.asarray(prob.get_val(candidate, units="kg")).ravel()
                if value.size:
                    return float(value[-1])
            except Exception:
                continue
        return np.nan

    # Prefer the latest phase that provides each cumulative mission quantity.
    phase_candidates = ["descent", "cruise", "climb", "groundroll"]
    oew_kg = get_weight_value([f"{phase}.OEW" for phase in phase_candidates])
    mission_fuel_kg = get_weight_value([f"{phase}.fuel_used_final" for phase in phase_candidates])
    engine_kg = get_weight_value(
        [
            f"analysis.{phase}.acmodel.propmodel.eng1.component_weight"
            for phase in phase_candidates
        ]
    )
    propeller_kg = get_weight_value(
        [
            f"analysis.{phase}.acmodel.propmodel.prop1.propweights.component_weight"
            for phase in phase_candidates
        ]
    )
    operating_weight_with_fuel_kg = oew_kg + mission_fuel_kg
    # Diagnostic sum includes engine and propeller in addition to OEW + fuel.
    # These component weights are already represented in OEW, so this is not
    # the aircraft operating weight and must not be used as such.
    literal_sum_kg = operating_weight_with_fuel_kg + engine_kg + propeller_kg
    mtow_kg = get_weight_value(["ac|weights|MTOW"])
    available_payload_kg = mtow_kg - operating_weight_with_fuel_kg

    print("\n=======================================================================\n")
    for var in print_vars:
        print(f"{var['name']}: {prob.get_val(var['var'], units=var['units']).item()} {var['units']}")
    print(f"Flight duration: {flight_duration:.2f} min")

    print("\n==================== WEIGHT BREAKDOWN ====================\n")
    print(f"{'Operating empty weight':<32} {oew_kg:>10.2f} kg")
    print(f"{'Mission fuel used':<32} {mission_fuel_kg:>10.2f} kg")
    print(f"{'Engine':<32} {engine_kg:>10.2f} kg")
    print(f"{'Propeller':<32} {propeller_kg:>10.2f} kg")
    print(f"{'Operating weight + fuel':<32} {operating_weight_with_fuel_kg:>10.2f} kg")
    print(f"{'MTOW':<32} {mtow_kg:>10.2f} kg")
    print(f"{'Available payload':<32} {available_payload_kg:>10.2f} kg")
    print(f"{'Literal sum of all four lines':<32} {literal_sum_kg:>10.2f} kg")

    # =============== Ground-roll plots ================
    import matplotlib.pyplot as plt

    takeoff_fig, takeoff_axs = plt.subplots(1, 3, figsize=[9, 2.7], constrained_layout=True)
    takeoff_axs = takeoff_axs.flatten()  # Convert the 1-by-3 axes array to a one-dimensional array.

    # Choose the flight quantities shown against ground-roll distance.
    takeoff_vars = [
        {"var": "fltcond|h", "name": "Altitude", "units": "ft"},
        {"var": "fltcond|Utrue", "name": "True airspeed", "units": "kn"},
        {"var": "throttle", "name": "Throttle", "units": None},
    ]

    for idx_fig, var in enumerate(takeoff_vars):
        takeoff_axs[idx_fig].set_xlabel("Range (ft)")
        takeoff_axs[idx_fig].set_ylabel(f"{var['name']}" if var["units"] is None else f"{var['name']} ({var['units']})")

        # Plot the selected quantity for the ground-roll phase.
        colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
        for i, phase in enumerate(["groundroll"]):
            takeoff_axs[idx_fig].plot(
                prob.get_val(f"{phase}.range", units="ft"),
                prob.get_val(f"{phase}.{var['var']}", units=var["units"]),
                "-o",
                c=colors[i],
                markersize=2.0,
            )

    takeoff_fig.legend(
        ["Ground roll"],
        loc=(0.067, 0.6),
        fontsize="small",
    )
    takeoff_fig.suptitle("Ground roll")
    takeoff_fig.savefig("turboprop_takeoff_results.svg", transparent=True)

    # =============== Mission flight-condition plots ================
    mission_phases = ["groundroll", "climb", "cruise", "descent"]
    phase_colors = {
        "groundroll": "tab:orange",
        "climb": "tab:blue",
        "cruise": "tab:green",
        "descent": "tab:red",
    }
    mission_handles = [
        plt.Line2D(
            [0],
            [0],
            color=phase_colors[phase],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=phase.capitalize(),
        )
        for phase in mission_phases
    ]

    # Plot the principal flight conditions in separate figures for easier comparison.
    flight_condition_vars = [
        {"var": "fltcond|h", "name": "Altitude", "units": "ft"},
        {"var": "fltcond|vs", "name": "Vertical speed", "units": "ft/min"},
        {"var": "fltcond|Utrue", "name": "True airspeed", "units": "kn"},
        {"var": "throttle", "name": "Throttle", "units": None},
    ]

    plot_filenames = {
        "Altitude": "turboprop_altitude_vs_range.svg",
        "Vertical speed": "turboprop_vertical_speed_vs_range.svg",
        "True airspeed": "turboprop_true_airspeed_vs_range.svg",
        "Throttle": "turboprop_throttle_vs_range.svg",
    }

    for var in flight_condition_vars:
        mission_fig, mission_ax = plt.subplots(figsize=[9, 5.5], constrained_layout=True)
        mission_ax.set_title(f"{var['name']} vs mission range", fontsize=16)
        mission_ax.set_xlabel("Range (nmi)", fontsize=14)
        mission_ax.set_ylabel(
            f"{var['name']}" if var["units"] is None else f"{var['name']} ({var['units']})",
            fontsize=14,
        )
        mission_ax.tick_params(axis="both", labelsize=12)
        mission_ax.grid(True, alpha=0.3)

        for phase in mission_phases:
            mission_ax.plot(
                prob.get_val(f"{phase}.range", units="nmi"),
                prob.get_val(f"{phase}.{var['var']}", units=var["units"]),
                "-o",
                c=phase_colors[phase],
                markersize=2.0,
            )

        mission_ax.legend(
            handles=mission_handles,
            loc="best",
            frameon=True,
            fontsize=12,
            markerscale=1.2,
        )
        mission_fig.savefig(plot_filenames[var["name"]], transparent=True)
        if var["name"] in ("Altitude", "True airspeed"):
            manager = mission_fig.canvas.manager
            if hasattr(manager, "set_window_title"):
                manager.set_window_title(var["name"])
            if not hide_viz:
                mission_fig.show()

    # Plot fuel flow and aircraft weight together in a separate mission-systems figure.
    mission_system_vars = [
        {"var": "propmodel.fuel_flow", "name": "Fuel flow", "units": "g/s"},
        {"var": "weight", "name": "Weight", "units": "kg"},
    ]

    mission_system_fig, mission_system_axs = plt.subplots(1, 2, figsize=[14, 5.5], constrained_layout=True)

    for mission_ax, var in zip(mission_system_axs, mission_system_vars):
        mission_ax.set_title(f"{var['name']} vs mission range", fontsize=16)
        mission_ax.set_xlabel("Range (nmi)", fontsize=14)
        mission_ax.set_ylabel(f"{var['name']} ({var['units']})", fontsize=14)
        mission_ax.tick_params(axis="both", labelsize=12)
        mission_ax.grid(True, alpha=0.3)
        for phase in mission_phases:
            mission_ax.plot(
                prob.get_val(f"{phase}.range", units="nmi"),
                prob.get_val(f"{phase}.{var['var']}", units=var["units"]),
                "-o",
                c=phase_colors[phase],
                markersize=2.0,
            )

    for mission_ax in mission_system_axs:
        mission_ax.legend(
            handles=mission_handles,
            loc="best",
            frameon=True,
            fontsize=12,
            markerscale=1.2,
        )
    mission_system_fig.savefig("turboprop_mission_systems_results.svg", transparent=True)

    # =============== Engine shaft power over mission time ================
    power_time_fig, power_time_ax = plt.subplots(figsize=[8, 3.2], constrained_layout=True)
    power_time_ax.set_xlabel("Mission time (min)", fontsize=14)
    power_time_ax.set_ylabel("Engine shaft power (kW)", fontsize=14)
    power_time_ax.set_title("Engine power required", fontsize=16)
    power_time_ax.tick_params(axis="both", labelsize=12)
    power_time_ax.grid(True, alpha=0.3)

    # Build a continuous time axis by offsetting each phase by the duration
    # of all preceding phases.
    cumulative_time_min = 0.0
    for phase in ["groundroll", "climb", "cruise", "descent"]:
        phase_duration_s = prob.get_val(f"{phase}.duration", units="s").item()
        phase_power_kw = prob.get_val(f"{phase}.propmodel.eng1.shaft_power_out", units="kW")
        phase_time_min = np.linspace(
            cumulative_time_min,
            cumulative_time_min + phase_duration_s / 60.0,
            phase_power_kw.size,
        )
        power_time_ax.plot(
            phase_time_min,
            phase_power_kw,
            "-o",
            c=phase_colors[phase],
            markersize=2.0,
            label=phase.capitalize(),
        )
        cumulative_time_min += phase_duration_s / 60.0

    power_time_ax.legend(fontsize=12, markerscale=1.2)
    power_time_fig.savefig("turboprop_engine_power_vs_time.svg", transparent=True)

    # =============== Total aircraft drag over mission time ================
    drag_time_fig, drag_time_ax = plt.subplots(figsize=[8, 3.2], constrained_layout=True)
    drag_time_ax.set_xlabel("Mission time (min)", fontsize=14)
    drag_time_ax.set_ylabel("Total drag (N)", fontsize=14)
    drag_time_ax.set_title("Total drag", fontsize=16)
    drag_time_ax.tick_params(axis="both", labelsize=12)
    drag_time_ax.grid(True, alpha=0.3)

    cumulative_time_min = 0.0
    for phase in ["groundroll", "climb", "cruise", "descent"]:
        phase_duration_s = prob.get_val(f"{phase}.duration", units="s").item()
        phase_drag_n = np.asarray(prob.get_val(f"{phase}.drag", units="N")).ravel()
        phase_time_min = np.linspace(
            cumulative_time_min,
            cumulative_time_min + phase_duration_s / 60.0,
            phase_drag_n.size,
        )
        drag_time_ax.plot(
            phase_time_min,
            phase_drag_n,
            "-o",
            c=phase_colors[phase],
            markersize=2.0,
            label=phase.capitalize(),
        )
        cumulative_time_min += phase_duration_s / 60.0

    drag_time_ax.legend(fontsize=12, markerscale=1.2)
    drag_time_fig.savefig("turboprop_drag_vs_time.svg", transparent=True)

    mission_drag = np.concatenate(
        [
            np.asarray(prob.get_val(f"{phase}.drag", units="N")).ravel()
            for phase in ["groundroll", "climb", "cruise", "descent"]
        ]
    )
    print(f"Mission average drag: {np.mean(mission_drag):.2f} N")

    if not hide_viz:
        plt.show()
    # rst Run (end)
