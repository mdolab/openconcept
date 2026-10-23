"""
Mission simulation and thermal management analysis for a hydrogen-powered PC-12 NGX.

The model represents the ground-roll, climb, cruise, and descent phases, and
calculates aircraft performance, hydrogen consumption, and the behavior of the
thermal management system and heat exchanger (HX).

Initial flight, mission, and geometry values are assigned in set_values().
For the no-optimization mode, set_no_optimization_parameters() also fixes the
HX geometry and the inlet and outlet area ratios.

Available run modes:
  0. No optimization: evaluate the model with the initial values and fixed
     inputs; no variables are optimized.
  1. First optimization only: minimize average drag by adjusting n_wide_cold,
     n_long_cold, and n_tall, subject to the fuel-cell temperature limit.
  2. Both optimizations: first optimize the HX geometry, then adjust the inlet
     and nozzle area ratios at each node to reach the target mean temperature
     without exceeding the maximum temperature.

The second optimization is not available as a standalone mode in the current
menu; selecting mode 2 runs both stages in sequence.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import argparse

import matplotlib.pyplot as plt
import numpy as np

import openmdao.api as om
from openmdao.api import (
    BoundsEnforceLS,
    DirectSolver,
    Group,
    IndepVarComp,
    NewtonSolver,
    Problem,
    ScipyOptimizeDriver,
)

from openconcept.aerodynamics import PolarDrag
from openconcept.examples.aircraft_data.PC12NGXHydrogen import data as acdata
from openconcept.mission import BasicMission
from openconcept.propulsion.systems.simple_fuel_cell_hybrid import (
    FuelCellPropulsionSystemIncompressible,
    LH2TankSizing,
)
from openconcept.utilities import AddSubtractComp, DictIndepVarComp, Integrator
from openconcept.weights import SingleTurboPropEmptyWeight


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================
class PC12NGXHydrogenAirplaneModel(Group):
    """
    A custom aircraft model configuration for a Hydrogen Fuel Cell Pilatus PC-12.
    Passed directly into the FullMissionAnalysis environment.
    """

    def initialize(self) -> None:
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)

    def setup(self) -> None:
        nn = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]

        # Define the propeller rotational speed at every node in the phase.
        controls = self.add_subsystem("controls", IndepVarComp(), promotes_outputs=["*"])
        controls.add_output("prop1rpm", val=np.ones((nn,)) * 1700, units="rpm")

        # Promotes the propulsion configuration inputs and primary outputs.
        propulsion_promotes_outputs = ["thrust", "fuel_flow", "fuel_cell_system_weight"]
        propulsion_promotes_inputs = ["fltcond|*", "ac|propulsion|*", "throttle", "duration"]

        # Combines the motor, propeller, and fuel cell in the propulsion system.
        self.add_subsystem(
            "propmodel",
            FuelCellPropulsionSystemIncompressible(num_nodes=nn),
            promotes_inputs=propulsion_promotes_inputs,
            promotes_outputs=propulsion_promotes_outputs,
        )
        self.connect("prop1rpm", "propmodel.propeller.rpm")

        # ======================================== Aerodynamics ========================================
        # Selects the takeoff or cruise zero-lift drag coefficient for this phase.
        if flight_phase == "v0v1":
            cd0_source = "ac|aero|polar|CD0_TO"
        else:
            cd0_source = "ac|aero|polar|CD0_cruise"

        # Calculates aerodynamic drag from the aircraft airframe.
        self.add_subsystem(
            "drag_airframe",
            PolarDrag(num_nodes=nn),
            promotes_inputs=[
                "fltcond|CL",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                ("CD0", cd0_source),
                "fltcond|q",
                ("e", "ac|aero|polar|e"),
            ],
            promotes_outputs=[("drag", "drag_airframe")],  
        )

        # Adds airframe and heat-exchanger drag to obtain total aircraft drag.
        self.add_subsystem(
            "drag",
            AddSubtractComp(
                output_name="drag",
                input_names=["drag_airframe", "hx_drag"],
                units="N",
                vec_size=[nn, nn],  # Evaluates both drag contributions at every phase node.
                scaling_factors=[1.0, 1.0],  # Adds both contributions to total drag.
            ),
            promotes_outputs=["drag"],  
        )

        # Connects both drag contributions to the summation component.
        self.connect("drag_airframe", "drag.drag_airframe")
        self.connect("propmodel.cooling_duct.drag", "drag.hx_drag")

        # ======================================== Weights ========================================
        self.add_subsystem(
            "OEW",
            SingleTurboPropEmptyWeight(),
            promotes_inputs=["*", ("P_TO", "ac|propulsion|engine|rating")],
            promotes_outputs=["OEW"],
        )
        self.connect("propmodel.propeller.component_weight", "W_propeller")
        self.connect("propmodel.motor.component_weight", "W_engine")

        # Integrates fuel consumption over the phase to calculate the amount used.
        intfuel = self.add_subsystem(
            "intfuel",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_outputs=["fuel_used_final"],
        )
        intfuel.add_integrand("fuel_used", rate_name="fuel_flow", val=1.0, units="kg")
        self.connect("fuel_flow", "intfuel.fuel_flow")

        # Calculates aircraft weight by subtracting fuel used from maximum takeoff weight.
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


class PC12NGXHydrogenAnalysisGroup(om.Group):
    """
    This is an example of a balanced field takeoff and three-phase mission analysis.
    """

    def initialize(self) -> None:
        self.options.declare("num_nodes", default=11)

    def setup(self) -> None:
        nn = self.options["num_nodes"]

        # Exposes aircraft-specific parameters from the acdata dataset.
        dv_comp = self.add_subsystem("dv_comp", DictIndepVarComp(acdata), promotes_outputs=["*"])

        # Aerodynamic parameters.
        dv_comp.add_output_from_dict("ac|aero|CLmax_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|e")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_cruise")

        # Geometric parameters.
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

        # Weight parameters.
        dv_comp.add_output_from_dict("ac|weights|MTOW")
        dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
        dv_comp.add_output_from_dict("ac|weights|MLW")

        # Propulsion parameters.
        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output_from_dict("ac|propulsion|propeller|diameter")

        # Other aircraft parameters and the electric motor's rated power.
        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")
        dv_comp.add_output("ac|propulsion|motor|rating", val=1200, units="hp")
        # Initial HX geometry values are assigned in set_values().

        mission_data_comp = self.add_subsystem("mission_data_comp", IndepVarComp(), promotes_outputs=["*"])
        mission_data_comp.add_output("T_res_initial", val=15.1, units="degC")

        # Builds the mission with ground roll, climb, cruise, and descent phases.
        self.add_subsystem(
            "analysis",
            BasicMission(
                num_nodes=nn,
                aircraft_model=PC12NGXHydrogenAirplaneModel,
                include_ground_roll=True,
                include_descent=True,
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        # Sizes the liquid-hydrogen tank from total mission consumption; the integrated
        # fuel value at the end of descent includes consumption from all previous phases.
        self.add_subsystem(
            "lh2_tank_sizing",
            LH2TankSizing(gravimetric_index=0.20),
            promotes_outputs=[
                ("tank_structural_mass", "ac|weights|W_LH2_tank"),
                ("total_tank_system_mass", "LH2_tank_total_mass"),
            ],
        )
        self.connect("descent.fuel_used_final", "lh2_tank_sizing.fuel_consumed")
        
        self.connect("T_res_initial", "groundroll.propmodel.coolant_reservoir.T_initial")
        self.connect("T_res_initial", "groundroll.propmodel.fc_thermal.T_initial")

        # These components use local integrators, so BasicMission does not automatically
        # link their temperatures between phases. Connect each phase's final state to the
        # next phase's initial state to preserve thermal continuity throughout the mission.
        thermal_phase_links = [
            ("groundroll", "climb"),
            ("climb", "cruise"),
            ("cruise", "descent"),
        ]
        for phase_from, phase_to in thermal_phase_links:
            self.connect(f"{phase_from}.propmodel.fc_thermal.T_final", f"{phase_to}.propmodel.fc_thermal.T_initial")
            self.connect(
                f"{phase_from}.propmodel.coolant_reservoir.T_final",
                f"{phase_to}.propmodel.coolant_reservoir.T_initial",
            )


# ==============================================================================
# PROBLEM CONFIGURATION AND INITIALIZATION FUNCTIONS
# ==============================================================================
def configure_problem(num_nodes: int = 31, temp_limit_c: float = 80.0, maxiter: int = 60, tol: float = 1e-3) -> Problem:
    prob = om.Problem()
    prob.model = PC12NGXHydrogenAnalysisGroup(num_nodes=num_nodes)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["maxiter"] = maxiter
    prob.driver.options["tol"] = tol
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["objs", "nl_cons"]

    # Configures the nonlinear and linear solvers for the coupled model.
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2)
    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

    # Sets the solver convergence criteria and iteration limit.
    prob.model.nonlinear_solver.options["solve_subsystems"] = True
    prob.model.nonlinear_solver.options["maxiter"] = 10
    prob.model.nonlinear_solver.options["atol"] = 1e-3
    prob.model.nonlinear_solver.options["rtol"] = 1e-3

    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar")

    return prob


# FIRST OPTIMIZATION: configured here and executed by prob.run_driver() in the
# main routine. It minimizes average climb/cruise/descent drag by varying the
# heat-exchanger geometry variables n_wide_cold, n_long_cold, and n_tall.
# Fuel-cell temperature is constrained at every node of every mission phase.
def configure_drag_minimization_optimization(prob: Problem, num_nodes: int, temp_limit_c: float = 80.0) -> None:
    """Configure optimization: minimize mission drag while enforcing thermal limit in all phases."""
    # First-stage design variables: discrete dimensions of the heat exchanger.
    prob.model.add_design_var("ac|propulsion|thermal|hx|n_wide_cold", lower=300.0, upper=800.0, scaler=1e-2)
    prob.model.add_design_var("ac|propulsion|thermal|hx|n_long_cold", lower=10.0, upper=23.0)
    prob.model.add_design_var("ac|propulsion|thermal|hx|n_tall", lower=30.0, upper=200.0)

    # Defines the objective: minimize average drag during climb, cruise, and descent.
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

    # Limits fuel-cell temperature at every node in every mission phase.
    max_temp_k = temp_limit_c + 273.15
    mission_vector_phases = ["groundroll", "climb", "cruise", "descent"]
    for phase_name in mission_vector_phases:
        prob.model.add_constraint(
            f"{phase_name}.propmodel.fc_thermal.T",
            upper=max_temp_k * np.ones((num_nodes,)),
        )


def get_max_fuel_cell_temperature(prob: Problem, num_nodes: int) -> float:
    """Return global max fuel-cell temperature across all modeled phases (degC)."""
    phases = ["groundroll", "climb", "cruise", "descent"]
    tmax = -np.inf
    for phase_name in phases:
        try:
            t_phase_k = np.asarray(prob.get_val(f"{phase_name}.propmodel.fc_thermal.T")).ravel()
            t_phase = t_phase_k - 273.15
            t_phase = t_phase[np.isfinite(t_phase)]
            if t_phase.size:
                tmax = max(tmax, float(np.max(t_phase)))
        except KeyError:
            continue
    return float(tmax)


# SECOND OPTIMIZATION: implemented in this function and called from the main
# routine after the first optimization. It adjusts the inlet_area_ratio and
# nozzle_area_ratio at every node across the mission; it does not change the
# heat-exchanger geometry variables. The goal is a mean fuel-cell temperature
# of at least min_mean_temp_c while keeping every node below temp_limit_c.
def optimize_mission_nozzle_area_ratio_fast(
    prob,
    phases: list[str],
    num_nodes: int,
    temp_limit_c: float = 80.0,
    boundary_temp_bonus_c: float = 0.0,
    min_mean_temp_c: float = 73.0,
    initial_step: float = 0.04,
    min_step: float = 0.0001,
    min_ratio_limit: float = 0.03,
    max_ratio_limit: float = 0.45,
    max_outer_iters: int = 500,
    max_node_delta: float = 0.35,
    inlet_min_ratio: float = 0.03,
    inlet_max_ratio: float = 0.45,
    inlet_initial_step: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    phase_sizes = {
        phase: np.asarray(
            prob.get_val(f"{phase}.propmodel.nozzle_area_from_hx.nozzle_area_ratio")
        ).size
        for phase in phases
    }
    total_nodes = sum(phase_sizes.values())

    # Assigns each phase its segment of the global area-ratio vectors.
    def set_mission_geometry(nozzle_ratio_array: np.ndarray, inlet_ratio_array: np.ndarray):
        offset = 0
        for phase in phases:
            end = offset + phase_sizes[phase]
            prob.set_val(
                f"{phase}.propmodel.nozzle_area_from_hx.nozzle_area_ratio",
                nozzle_ratio_array[offset:end],
            )
            prob.set_val(
                f"{phase}.propmodel.inlet_area_from_hx.inlet_area_ratio",
                inlet_ratio_array[offset:end],
            )
            offset = end
            
    # Collects temperatures from all phases into one global vector in degrees Celsius.
    def get_mission_temp() -> np.ndarray:
        temps = []
        for phase in phases:
            t = np.asarray(prob.get_val(f"{phase}.propmodel.fc_thermal.T")).ravel() - 273.15
            temps.append(t)
        return np.concatenate(temps)

    # Applies the thermal limit to every mission node.
    temp_limits = np.full(total_nodes, temp_limit_c)
    if total_nodes > 0:
        temp_limits[0] += boundary_temp_bonus_c
        temp_limits[-1] += boundary_temp_bonus_c

    # Starts with open inlet and nozzle areas to establish a thermally safe baseline.
    baseline_nozzle_ratio = np.full(total_nodes, max_ratio_limit)
    baseline_inlet_ratio = np.full(total_nodes, inlet_max_ratio)
    set_mission_geometry(baseline_nozzle_ratio, baseline_inlet_ratio)
    prob.run_model()
    baseline_temp_c = get_mission_temp()

    last_valid_nozzle_ratio = baseline_nozzle_ratio.copy()
    last_valid_inlet_ratio = baseline_inlet_ratio.copy()
    last_valid_temp_c = baseline_temp_c.copy()

    # If this baseline already exceeds the thermal limit, retain and return it.
    if np.any(baseline_temp_c > temp_limits + 1e-3):
        return last_valid_nozzle_ratio, last_valid_inlet_ratio, last_valid_temp_c

    nozzle_steps = np.full(total_nodes, initial_step)
    inlet_steps = np.full(total_nodes, inlet_initial_step)
    outer_iter = 0

    while outer_iter < max_outer_iters and max(np.max(nozzle_steps), np.max(inlet_steps)) > min_step:
        outer_iter += 1
        current_mean = float(np.mean(last_valid_temp_c))
        current_max = float(np.max(last_valid_temp_c))

        # Stops when the target mean temperature is reached without exceeding the maximum.
        if current_mean >= min_mean_temp_c and current_max <= temp_limit_c + 1e-4:
            break

        trial_nozzle_ratio = last_valid_nozzle_ratio.copy()
        trial_inlet_ratio = last_valid_inlet_ratio.copy()

        for i in range(total_nodes):
            t_val = last_valid_temp_c[i]
            headroom = temp_limit_c - t_val  # Remaining margin to the temperature limit.

            # Near the limit: increase both areas to provide more cooling.
            if headroom < 3.0:
                urgency = max(3.0 - headroom, 0.1)
                trial_nozzle_ratio[i] += nozzle_steps[i] * max(urgency * 1.5, 0.5)
                trial_inlet_ratio[i] += inlet_steps[i] * max(urgency, 0.5)

            # Cold region: reduce both areas to decrease cooling and let the temperature rise.
            elif headroom > 20.0:
                if trial_nozzle_ratio[i] > min_ratio_limit + 1e-4:
                    trial_nozzle_ratio[i] -= nozzle_steps[i] * 0.4
                if trial_inlet_ratio[i] > inlet_min_ratio + 1e-4:
                    trial_inlet_ratio[i] -= inlet_steps[i] * 0.4

            # Intermediate region: make smaller area adjustments as the temperature nears its limit.
            else:
                if trial_nozzle_ratio[i] > min_ratio_limit + 1e-4:
                    slow_factor = 0.1 * (headroom / 20.0)
                    trial_nozzle_ratio[i] -= nozzle_steps[i] * slow_factor
                if trial_inlet_ratio[i] > inlet_min_ratio + 1e-4:
                    trial_inlet_ratio[i] -= inlet_steps[i] * slow_factor

        # Keeps both area ratios within their physical bounds.
        trial_nozzle_ratio = np.clip(trial_nozzle_ratio, min_ratio_limit, max_ratio_limit)
        trial_inlet_ratio = np.clip(trial_inlet_ratio, inlet_min_ratio, inlet_max_ratio)

        # Smooths changes between nodes to avoid abrupt geometry variations.
        for _ in range(2):
            for ratio in (trial_nozzle_ratio, trial_inlet_ratio):
                ratio[1:] = np.maximum(ratio[1:], ratio[:-1] - max_node_delta)
                ratio[1:] = np.minimum(ratio[1:], ratio[:-1] + max_node_delta)
                ratio[:-1] = np.maximum(ratio[:-1], ratio[1:] - max_node_delta)
                ratio[:-1] = np.minimum(ratio[:-1], ratio[1:] + max_node_delta)

        trial_nozzle_ratio = np.clip(trial_nozzle_ratio, min_ratio_limit, max_ratio_limit)
        trial_inlet_ratio = np.clip(trial_inlet_ratio, inlet_min_ratio, inlet_max_ratio)

        # Stops if the proposed ratios do not meaningfully differ from the last valid values.
        if np.allclose(trial_nozzle_ratio, last_valid_nozzle_ratio, atol=1e-5) and np.allclose(
            trial_inlet_ratio, last_valid_inlet_ratio, atol=1e-5
        ):
            break

        # Evaluates the model with the trial geometry across all phases.
        set_mission_geometry(trial_nozzle_ratio, trial_inlet_ratio)
        prob.run_model()
        trial_temp_c = get_mission_temp()

        breached_nodes = trial_temp_c > temp_limits + 1e-4

        if np.any(breached_nodes):
            # Rejects a trial that exceeds the thermal limit and reduces the step where areas were narrowed.
            closing_mask = trial_nozzle_ratio < last_valid_nozzle_ratio - 1e-6
            nozzle_steps[closing_mask] *= 0.5
            inlet_steps[closing_mask] *= 0.5
            
        else:
            # Accepts the trial and saves the latest configuration that satisfies the thermal limit.
            last_valid_nozzle_ratio = trial_nozzle_ratio.copy()
            last_valid_inlet_ratio = trial_inlet_ratio.copy()
            last_valid_temp_c = trial_temp_c.copy()

    # Restores the best valid configuration and recalculates its temperatures.
    set_mission_geometry(last_valid_nozzle_ratio, last_valid_inlet_ratio)
    prob.run_model()
    final_temp_c = get_mission_temp()

    return last_valid_nozzle_ratio, last_valid_inlet_ratio, final_temp_c


def set_values(prob: Problem, num_nodes: int) -> None:
    # Sets the initial flight conditions and mission parameters used to start the model.
    # Ground roll requires a nonzero speed to calculate the flight path and propeller behavior.
    prob.set_val("groundroll.fltcond|Utrue", np.ones((num_nodes)) * 50, units="kn")
    prob.set_val("climb.fltcond|vs", np.linspace(1500, 100, num_nodes), units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.linspace(150, 150, num_nodes), units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 0.01, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.linspace(150, 150, num_nodes), units="kn")
    prob.set_val("descent.fltcond|vs", np.linspace(-750, -100, num_nodes), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.linspace(150, 100, num_nodes), units="kn")
    prob.set_val("cruise|h0", 7620, units="m")  # Cruise altitude equivalent to 25,000 ft.
    prob.set_val("mission_range", 500, units="NM")

    # Sets the initial HX geometry within the first-stage design-variable bounds.
    prob.set_val("ac|propulsion|thermal|hx|n_wide_cold", 800.0)
    prob.set_val("ac|propulsion|thermal|hx|n_long_cold", 5.0)
    prob.set_val("ac|propulsion|thermal|hx|n_tall", 50.0)


def set_no_optimization_parameters(prob: Problem, num_nodes: int) -> None:
    """Fixes the HX geometry and duct areas for a single model evaluation."""
    # To change the no-optimization HX geometry, edit the numeric values
    # assigned to n_wide_cold, n_long_cold, and n_tall below. These values are
    # fixed and are not optimized in mode 0.
    prob.set_val("ac|propulsion|thermal|hx|n_wide_cold", 800.0)
    prob.set_val("ac|propulsion|thermal|hx|n_long_cold", 10.0)
    prob.set_val("ac|propulsion|thermal|hx|n_tall", 10.0)

    # To change the no-optimization duct areas, edit fixed_inlet_ratio and
    # fixed_outlet_ratio. Each is the corresponding physical area divided by
    # the HX frontal area; both remain fixed across the mission in mode 0.
    fixed_inlet_ratio = 0.35
    fixed_outlet_ratio = 0.35

    # Uses the ground-roll frontal area as a reference and calculates phase-specific ratios
    # that preserve the same physical inlet and outlet areas throughout the mission.
    phases = ["groundroll", "climb", "cruise", "descent"]
    prob.run_model()
    reference_frontal_area = float(
        np.asarray(prob.get_val("groundroll.propmodel.hx_frontal_area", units="m**2")).ravel()[0]
    )
    fixed_inlet_area = fixed_inlet_ratio * reference_frontal_area
    fixed_outlet_area = fixed_outlet_ratio * reference_frontal_area
    for phase in phases:
        phase_frontal_area = np.asarray(
            prob.get_val(f"{phase}.propmodel.hx_frontal_area", units="m**2")
        ).ravel()
        phase_frontal_area = np.broadcast_to(phase_frontal_area, (num_nodes,))
        inlet_ratio_by_phase = fixed_inlet_area / phase_frontal_area
        outlet_ratio_by_phase = fixed_outlet_area / phase_frontal_area
        prob.set_val(f"{phase}.propmodel.inlet_area_from_hx.inlet_area_ratio", inlet_ratio_by_phase)
        prob.set_val(f"{phase}.propmodel.nozzle_area_from_hx.nozzle_area_ratio", outlet_ratio_by_phase)


def prompt_optimization_mode() -> tuple[bool, bool]:
    """Ask interactively how many optimization stages to run.

    Returns (no_optimization, skip_nozzle_opt).
    """
    print("\nSelect how to run the PC-12 NGX Hydrogen model:")
    print("  [0] No optimization (run once with the initial values)")
    print("  [1] One optimization (HX drag minimization only)")
    print("  [2] Both optimizations (HX sizing + per-node nozzle/inlet tuning)")
    while True:
        choice = input("Enter 0, 1, or 2 [default: 2]: ").strip()
        if choice in ("", "2"):
            return False, False
        if choice == "1":
            return False, True
        if choice == "0":
            return True, True
        print("Invalid choice. Please enter 0, 1, or 2.")

# ==============================================================================
# ENTRY POINT AND RUN MODES
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hide_visuals",
        default=False,
        action="store_true",
        help="Do not show matplotlib figure or open N2 diagram in browser",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=31,
        help="Number of analysis nodes per phase (default: 31 for intermediate-speed optimization)",
    )
    parser.add_argument(
        "--single_stage_opt",
        default=False,
        action="store_true",
        help="Disable continuation and run a single optimization stage directly at 85 C",
    )
    parser.add_argument(
        "--no_optimization",
        default=False,
        action="store_true",
        help="Run the model once with the initial values and skip all optimization stages",
    )
    parser.add_argument(
        "--skip_nozzle_opt",
        default=False,
        action="store_true",
        help="Skip the second (per-phase nozzle-area-ratio) optimization stage",
    )
    args = parser.parse_args()

    # If no mode is selected with command-line arguments, prompt for one interactively.
    # Explicit arguments keep scripted runs non-interactive.
    if not args.no_optimization and not args.skip_nozzle_opt:
        args.no_optimization, args.skip_nozzle_opt = prompt_optimization_mode()

    hide_viz = args.hide_visuals
    num_nodes = args.num_nodes
    temp_limit_c = 80.0

    prob = configure_problem(num_nodes=num_nodes, temp_limit_c=temp_limit_c, maxiter=60, tol=1e-3)
    configure_drag_minimization_optimization(prob, num_nodes=num_nodes, temp_limit_c=temp_limit_c)
    prob.setup(check=False, mode="fwd")
    set_values(prob, num_nodes)

    # MODE 0 — NO OPTIMIZATION
    # set_no_optimization_parameters() is the place to change the fixed HX
    # geometry (n_wide_cold, n_long_cold, n_tall) and inlet/outlet ratios.
    # This branch evaluates those supplied values without optimizing variables.
    if args.no_optimization:
        prob.model.nonlinear_solver.options["maxiter"] = 10
        set_no_optimization_parameters(prob, num_nodes)

    # Evaluates the initial state, which also serves as the starting point for optimization.
    prob.run_model()

    if args.no_optimization:
        print("\nOptimization skipped; reporting the initial model values.")
    else:
        # MODE 1 — FIRST OPTIMIZATION ONLY; MODE 2 — FIRST OPTIMIZATION STAGE
        # prob.run_driver() executes the design problem configured in
        # configure_drag_minimization_optimization(): minimize average drag by
        # adjusting n_wide_cold, n_long_cold, and n_tall subject to the
        # fuel-cell temperature limit.
        try:
            prob.run_driver()
            print("\nOptimization finished.")
        except Exception as err:
            print(f"\nOptimization failed with error: {err}")
            print("Continuing with last feasible model state.")

        for geometry_var in (
            "ac|propulsion|thermal|hx|n_wide_cold",
            "ac|propulsion|thermal|hx|n_long_cold",
            "ac|propulsion|thermal|hx|n_tall",
        ):
            optimized_value = float(np.asarray(prob.get_val(geometry_var)).ravel()[-1])
            prob.set_val(geometry_var, int(round(optimized_value)))

    # Recalculates the model with the final values before producing results.
    prob.run_model()

    mission_phases = ["groundroll", "climb", "cruise", "descent"]
    inlet_area_vector = np.concatenate(
        [
            np.asarray(
                prob.get_val(f"{phase}.propmodel.cooling_duct.area_inlet", units="m**2")
            ).ravel()
            for phase in mission_phases
        ]
    )
    outlet_area_vector = np.concatenate(
        [
            np.asarray(
                prob.get_val(f"{phase}.propmodel.cooling_duct.area_nozzle", units="m**2")
            ).ravel()
            for phase in mission_phases
        ]
    )
    mdot_vector = np.concatenate(
        [
            np.asarray(
                prob.get_val(f"{phase}.propmodel.cooling_duct.mdot", units="kg/s")
            ).ravel()
            for phase in mission_phases
        ]
    )
    print("\n---------------- Final Inlet Area Vector (m^2) ----------------")
    print(inlet_area_vector)
    print("\n---------------- Final Outlet Area Vector (m^2) ----------------")
    print(outlet_area_vector)
    print("\n---------------- Final Cooling Duct Air Mass Flow Vector (kg/s) ----------------")
    print(mdot_vector)

    total_flight_time_min = sum(
        float(np.asarray(prob.get_val(f"{phase}.duration", units="min")).ravel()[-1]) for phase in mission_phases
    )

    if args.no_optimization:
        print("\n==================== FIXED MODEL INPUTS AND RESULTS ====================\n")
        fixed_n_wide_cold = float(np.asarray(prob.get_val("ac|propulsion|thermal|hx|n_wide_cold")).ravel()[-1])
        fixed_n_long_cold = float(np.asarray(prob.get_val("ac|propulsion|thermal|hx|n_long_cold")).ravel()[-1])
        fixed_n_tall = float(np.asarray(prob.get_val("ac|propulsion|thermal|hx|n_tall")).ravel()[-1])
        fixed_inlet_ratio = float(np.asarray(prob.get_val("cruise.propmodel.inlet_area_from_hx.inlet_area_ratio")).ravel()[0])
        fixed_nozzle_ratio = float(np.asarray(prob.get_val("cruise.propmodel.nozzle_area_from_hx.nozzle_area_ratio")).ravel()[0])
        print(
            "Fixed HX geometry:"
            f" n_wide_cold={fixed_n_wide_cold:.3f},"
            f" n_long_cold={fixed_n_long_cold:.3f},"
            f" n_tall={fixed_n_tall:.3f}"
        )
        print(f"Fixed inlet area ratio: {fixed_inlet_ratio:.4f}")
        print(f"Fixed nozzle area ratio: {fixed_nozzle_ratio:.4f}")
        print(f"Maximum fuel-cell temperature in all phases: {get_max_fuel_cell_temperature(prob, num_nodes):.3f} degC")
    else:
        # Summarizes the final HX dimensions and thermal and aerodynamic values.
        print("\n==================== OPTIMIZATION SUMMARY ====================\n")
        n_wide_cold = np.asarray(prob.get_val("ac|propulsion|thermal|hx|n_wide_cold")).ravel()[-1]
        n_long_cold = np.asarray(prob.get_val("ac|propulsion|thermal|hx|n_long_cold")).ravel()[-1]
        n_tall = np.asarray(prob.get_val("ac|propulsion|thermal|hx|n_tall")).ravel()[-1]
        print(
            "Optimal HX geometry:"
            f" n_wide_cold={int(round(n_wide_cold))},"
            f" n_long_cold={int(round(n_long_cold))},"
            f" n_tall={int(round(n_tall))}"
        )
        print(f"Maximum fuel-cell temperature in all phases: {get_max_fuel_cell_temperature(prob, num_nodes):.3f} degC")
        nozzle_area_cruise = float(np.asarray(prob.get_val("cruise.propmodel.nozzle_area_from_hx.area_nozzle", units="m**2")).ravel()[-1])
        hx_area_cruise = float(np.asarray(prob.get_val("cruise.propmodel.hx_frontal_area", units="m**2")).ravel()[-1])
        print(f"Cruise HX frontal area: {hx_area_cruise:.4f} m^2")
        print(f"Cruise nozzle area (0.4 * frontal area): {nozzle_area_cruise:.4f} m^2")

    om.n2(prob, outfile="pc12ngx_hydrogen_n2.html", show_browser=not hide_viz)

    # MODE 2 — SECOND OPTIMIZATION STAGE
    # optimize_mission_nozzle_area_ratio_fast() adjusts inlet_area_ratio and
    # nozzle_area_ratio at every node to target a mean fuel-cell temperature
    # of at least 73 °C without exceeding 80 °C at any node. The HX geometry
    # variables are not changed in this stage. Use --skip_nozzle_opt to omit it;
    # the second stage cannot run by itself.
    if not args.skip_nozzle_opt and not args.no_optimization:
        print("\n==================== GLOBAL MISSION NOZZLE AREA RATIO OPTIMIZATION ====================\n")

        optimizable_phases = ["groundroll", "climb", "cruise", "descent"]
        
        target_temp_limit_c = 80.0
        boundary_bonus_c = 0.0
        target_min_mean_temp_c = 73.0

        global_nozzle_ratio, global_inlet_ratio, global_temp_c = optimize_mission_nozzle_area_ratio_fast(
            prob=prob,
            phases=optimizable_phases,
            num_nodes=num_nodes,
            temp_limit_c=target_temp_limit_c,
            boundary_temp_bonus_c=boundary_bonus_c,
            min_mean_temp_c=target_min_mean_temp_c,
        )

        mean_temp = float(np.mean(global_temp_c))
        max_temp = float(np.max(global_temp_c))
        min_temp = float(np.min(global_temp_c))

        if mean_temp >= target_min_mean_temp_c - 1e-4 and max_temp <= target_temp_limit_c + 1e-4:
            status = "PASSED (Mission global targets achieved)"
        else:
            status = "STOPPED (Hit thermal or physical limits before reaching mean target)"

        print(f"Mission Status:          {status}")
        print(f"Global Temperature (°C): Min: {min_temp:.2f}  |  Mean: {mean_temp:.2f}  |  Max: {max_temp:.2f}")
        print(f"Global Nozzle Ratio:     Min: {global_nozzle_ratio.min():.4f} |  Mean: {global_nozzle_ratio.mean():.4f} |  Max: {global_nozzle_ratio.max():.4f}")
        print(f"Global Inlet Ratio:      Min: {global_inlet_ratio.min():.4f} |  Mean: {global_inlet_ratio.mean():.4f} |  Max: {global_inlet_ratio.max():.4f}\n")

        global_max_temp = get_max_fuel_cell_temperature(prob, num_nodes)
        print("-" * 110)
        print(f"\nMaximum fuel-cell temperature across all mission phases: {global_max_temp:.3f} °C")

        print("\n---------------- Final Global Nozzle Area Ratio Vector ----------------")
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        print(global_nozzle_ratio)

        print("\n---------------- Final Global Fuel-Cell Temperature Vector ----------------")
        print(np.round(global_temp_c, 2))

        print("\n---------------- Final Global Inlet Area Ratio Vector ----------------")
        print(global_inlet_ratio)

        om.n2(prob, outfile="pc12ngx_hydrogen_nozzle_n2.html", show_browser=not hide_viz)
    else:
        print("\nSkipping the second (nozzle-area-ratio) optimization.")

    print(f"\nFlight duration: {total_flight_time_min:.2f} min")
    drag_metric = float(np.asarray(prob.get_val("avg_drag_metric.drag_metric", units="N")).ravel()[-1])
    print(f"Average drag: {drag_metric:.3f} N")

    # Variables reported in the consumption, mass, and dimension summaries.
    print_vars = [
        {"var": "ac|weights|MTOW", "name": "MTOW", "units": "lb"},
        {"var": "descent.fuel_used_final", "name": "Total H2 required for tank", "units": "kg"},
        {"var": "ac|weights|W_LH2_tank", "name": "LH2 tank structural mass", "units": "kg"},
        {"var": "LH2_tank_total_mass", "name": "LH2 tank total system mass", "units": "kg"},
        {"var": "groundroll.fuel_used_final", "name": "Ground Roll H2 fuel", "units": "lb"},
        {"var": "climb.fuel_used_final", "name": "Climb H2 fuel", "units": "lb"},
        {"var": "cruise.fuel_used_final", "name": "Cruise H2 fuel", "units": "lb"},
        {"var": "descent.fuel_used_final", "name": "Total H2 fuel used", "units": "lb"},
        {"var": "cruise.propmodel.hx.component_weight", "name": "Heat Exchanger Weight", "units": "kg"},
    ]

    thermal_vars = [
        {"var": "cruise.propmodel.fc_thermal.T", "name": "Fuel Cell Temperature (Cruise End)", "units": "degC"},
        {"var": "cruise.propmodel.fuelcell.heat_output", "name": "Heat Generated by Stack", "units": "kW"},
        {"var": "cruise.propmodel.hx.heat_transfer", "name": "Heat Dissipated by Heat Exchanger", "units": "kW"},
        {"var": "cruise.propmodel.cooling_duct.mdot", "name": "Cooling Air Mass Flow Rate", "units": "kg/s"},
        {"var": "cruise.propmodel.hx.component_weight", "name": "Heat Exchanger Weight", "units": "kg"},
    ]

    hx_summary_vars = [
        {"var": "cruise.propmodel.hx.component_weight", "name": "Heat Exchanger Weight", "units": "kg"},
        {"var": "cruise.propmodel.hx_frontal_area", "name": "Heat Exchanger Frontal Area", "units": "m**2"},
        {"var": "cruise.propmodel.hx.length_overall", "name": "Heat Exchanger Length", "units": "m"},
        {"var": "cruise.propmodel.hx.width_overall", "name": "Heat Exchanger Width", "units": "m"},
        {"var": "cruise.propmodel.hx.height_overall", "name": "Heat Exchanger Height", "units": "m"},
        {"var": "cruise.propmodel.hx.heat_transfer_area_cold", "name": "Cold-Side Heat Transfer Area", "units": "m**2"},
        {"var": "cruise.propmodel.hx.UA_overall", "name": "Overall UA", "units": "W/K"},
        {"var": "cruise.propmodel.hx.effectiveness", "name": "Heat Exchanger Effectiveness", "units": None},
        {"var": "cruise.propmodel.hx.delta_p_cold", "name": "Cold-Side Pressure Drop", "units": "Pa"},
    ]

    print("\n==================== NORMAL SIMULATION RESULTS ====================\n")
    for var in thermal_vars:
        try:
            val_array = prob.get_val(var["var"], units=var["units"])
            val = val_array[-1] if np.ndim(val_array) > 0 else val_array.item()
            print(f"{var['name']}: {val:.2f} {var['units']}")
        except KeyError:
            print(f"{var['name']}: [Variable '{var['var']}' not found in the model]")

    print("\n==================== HEAT EXCHANGER SUMMARY ====================\n")
    hx_values = {}
    for var in hx_summary_vars:
        try:
            if var["units"] is None:
                val_array = prob.get_val(var["var"])
            else:
                val_array = prob.get_val(var["var"], units=var["units"])
            val = val_array[-1] if np.ndim(val_array) > 0 else val_array.item()
            hx_values[var["name"]] = val
            units_suffix = "" if var["units"] is None else f" {var['units']}"
            print(f"{var['name']}: {val:.4f}{units_suffix}")
        except KeyError:
            print(f"{var['name']}: [Variable '{var['var']}' not found in the model]")

    if all(name in hx_values for name in ["Heat Exchanger Length", "Heat Exchanger Width", "Heat Exchanger Height"]):
        hx_volume = (
            hx_values["Heat Exchanger Length"]
            * hx_values["Heat Exchanger Width"]
            * hx_values["Heat Exchanger Height"]
        )
        print(f"Heat Exchanger Total Volume: {hx_volume:.6f} m^3")

    print("\n==================== PC-12 NGX HYDROGEN RESULTS ====================\n")
    for var in print_vars:
        try:
            val = prob.get_val(var["var"], units=var["units"]).item()
            print(f"{var['name']}: {val:.2f} {var['units']}")
        except KeyError:
            print(f"{var['name']}: [Variable '{var['var']}' not found in model]")

    print("\n==================== WEIGHT SUMMARY ====================\n")

    def get_weight_value(prob_obj, candidates, units=None):
        for candidate in candidates:
            try:
                value = np.asarray(prob_obj.get_val(candidate, units=units)).ravel()
                if value.size:
                    return float(value[-1])
            except Exception:
                continue
        return np.nan

    conventional_oew_kg = get_weight_value(prob, ["OEW", "cruise.OEW", "climb.OEW", "descent.OEW", "groundroll.OEW"], units="kg")
    conventional_engine_kg = get_weight_value(
        prob,
        [
            f"analysis.{phase}.acmodel.propmodel.motor.component_weight"
            for phase in ["descent", "cruise", "climb", "groundroll"]
        ],
        units="kg",
    )
    conventional_propeller_kg = get_weight_value(
        prob,
        [
            f"analysis.{phase}.acmodel.propmodel.propeller.propweights.component_weight"
            for phase in ["descent", "cruise", "climb", "groundroll"]
        ],
        units="kg",
    )
    fuel_cell_kg = get_weight_value(prob, ["cruise.propmodel.fuelcell.component_weight", "climb.propmodel.fuelcell.component_weight", "descent.propmodel.fuelcell.component_weight"], units="kg")
    bop_kg = get_weight_value(prob, ["cruise.propmodel.bop.component_weight", "climb.propmodel.bop.component_weight", "descent.propmodel.bop.component_weight"], units="kg")
    intercooler_kg = get_weight_value(
        prob,
        ["cruise.propmodel.intercooler_mass", "climb.propmodel.intercooler_mass", "descent.propmodel.intercooler_mass"],
        units="kg",
    )
    humidifier_kg = get_weight_value(
        prob,
        ["cruise.propmodel.humidifier_mass", "climb.propmodel.humidifier_mass", "descent.propmodel.humidifier_mass"],
        units="kg",
    )
    bop_factor = get_weight_value(
        prob,
        [
            "cruise.propmodel.bop_factor",
            "climb.propmodel.bop_factor",
            "descent.propmodel.bop_factor",
            "groundroll.propmodel.bop_factor",
            "ac|propulsion|fuelcell|bop_factor",
        ],
    )
    fuel_cell_adjusted_kg = fuel_cell_kg * (1.0 + bop_factor)
    bop_adjusted_kg = bop_kg * (1.0 + bop_factor)
    intercooler_adjusted_kg = intercooler_kg * (1.0 + bop_factor)
    humidifier_adjusted_kg = humidifier_kg * (1.0 + bop_factor)
    tank_kg = get_weight_value(prob, ["ac|weights|W_LH2_tank"], units="kg")
    hx_kg = get_weight_value(prob, ["cruise.propmodel.hx.component_weight", "climb.propmodel.hx.component_weight", "descent.propmodel.hx.component_weight"], units="kg")
    mission_fuel_kg = get_weight_value(prob, ["descent.fuel_used_final", "cruise.fuel_used_final", "climb.fuel_used_final", "groundroll.fuel_used_final"], units="kg")
    mtow_kg = get_weight_value(prob, ["ac|weights|MTOW"], units="kg")

    fuel_cell_system_kg = get_weight_value(
        prob,
        [
            f"analysis.{phase}.acmodel.propmodel.fuel_cell_system_weight.system_weight"
            for phase in ["descent", "cruise", "climb", "groundroll"]
        ],
        units="kg",
    )
    operating_empty_weight_kg = conventional_oew_kg + fuel_cell_system_kg + tank_kg + hx_kg
    tms_without_hx_kg = 150.0
    operating_empty_weight_kg += tms_without_hx_kg
    payload_available_kg = mtow_kg - mission_fuel_kg - operating_empty_weight_kg

    def print_weight_line(label, value):
        if np.isfinite(value):
            print(f"{label:<32} {value:>10.2f} kg")
        else:
            print(f"{label:<32} {'n/a':>10}")

    print_weight_line("Conventional OEW", conventional_oew_kg)
    print_weight_line("  Electric motor", conventional_engine_kg)
    print_weight_line("  Propeller", conventional_propeller_kg)
    print_weight_line("Hydrogen systems added", fuel_cell_system_kg + tank_kg + hx_kg + tms_without_hx_kg)
    print_weight_line("  Fuel cell Stack incl. factor", fuel_cell_adjusted_kg)
    print_weight_line("  BOP package incl. factor", bop_adjusted_kg)
    print_weight_line("    Intercooler incl. factor", intercooler_adjusted_kg)
    print_weight_line("    Humidifier incl. factor", humidifier_adjusted_kg)
    print_weight_line("  LH2 tank", tank_kg)
    print_weight_line("  Heat exchanger", hx_kg)
    print_weight_line("  TMS without heat exchanger", tms_without_hx_kg)
    print_weight_line("Operating empty weight", operating_empty_weight_kg)
    print_weight_line("Mission fuel used", mission_fuel_kg)
    print_weight_line("Payload available", payload_available_kg)

    # Increases text sizes to improve figure readability.
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
        }
    )

    # Ground-roll plots: altitude, speed, and throttle position versus distance.
    takeoff_fig, takeoff_axs = plt.subplots(1, 3, figsize=[9, 2.7], constrained_layout=True)
    takeoff_axs = takeoff_axs.flatten()

    takeoff_vars = [
        {"var": "fltcond|h", "name": "Altitude", "units": "ft"},
        {"var": "fltcond|Utrue", "name": "True airspeed", "units": "kn"},
        {"var": "throttle", "name": "Throttle", "units": None},
    ]

    colors = ["tab:blue"]
    phases_takeoff = ["groundroll"]

    for idx_fig, var in enumerate(takeoff_vars):
        takeoff_axs[idx_fig].set_xlabel("Range (ft)")
        y_label = f"{var['name']}" if var["units"] is None else f"{var['name']} ({var['units']})"
        takeoff_axs[idx_fig].set_ylabel(y_label)

        for i, phase in enumerate(phases_takeoff):
            try:
                x_val = prob.get_val(f"{phase}.range", units="ft")
                y_val = prob.get_val(f"{phase}.{var['var']}", units=var["units"])
                takeoff_axs[idx_fig].plot(x_val, y_val, "-o", c=colors[i], markersize=2.0)
            except KeyError:
                pass

    takeoff_fig.legend(["Ground Roll"], loc=(0.067, 0.6), fontsize="small")
    takeoff_fig.suptitle("Ground Roll (PC-12 NGX H2)")
    takeoff_fig.savefig("pc12ngx_takeoff_results.svg", transparent=True)

    # Full mission profile: flight conditions, throttle, hydrogen consumption, and weight.
    mission_fig, mission_axs = plt.subplots(2, 3, figsize=[9, 4.8], constrained_layout=True)
    mission_axs = mission_axs.flatten()

    mission_vars = [
        {"var": "fltcond|h", "name": "Altitude", "units": "ft"},
        {"var": "fltcond|vs", "name": "Vertical speed", "units": "ft/min"},
        {"var": "fltcond|Utrue", "name": "True airspeed", "units": "kn"},
        {"var": "throttle", "name": "Throttle", "units": None},
        {"var": "fuel_flow", "name": "H2 Fuel flow", "units": "g/s"},
        {"var": "weight", "name": "Weight", "units": "kg"},
    ]

    phases_mission = ["climb", "cruise", "descent"]
    power_phases = ["groundroll", "climb", "cruise", "descent"]
    phase_colors = {
        "groundroll": "tab:orange",
        "climb": "tab:blue",
        "cruise": "tab:green",
        "descent": "tab:red",
    }

    def get_time_domain_series(phases, y_var, y_units):
        time_segments = []
        value_segments = []
        elapsed_time_min = 0.0

        for phase in phases:
            try:
                y_phase = np.asarray(prob.get_val(f"{phase}.{y_var}", units=y_units)).ravel()
                duration_min = float(np.asarray(prob.get_val(f"{phase}.duration", units="min")).ravel()[-1])
            except KeyError:
                continue

            x_phase = np.linspace(elapsed_time_min, elapsed_time_min + duration_min, len(y_phase))
            finite_mask = np.isfinite(x_phase) & np.isfinite(y_phase)
            if np.any(finite_mask):
                time_segments.append(x_phase[finite_mask])
                value_segments.append(y_phase[finite_mask])

            elapsed_time_min += duration_min

        if not time_segments:
            return None, None

        return np.concatenate(time_segments), np.concatenate(value_segments)

    def get_time_domain_segments(phases, y_var, y_units):
        segments = []
        elapsed_time_min = 0.0

        for phase in phases:
            try:
                y_phase = np.asarray(prob.get_val(f"{phase}.{y_var}", units=y_units)).ravel()
                duration_min = float(np.asarray(prob.get_val(f"{phase}.duration", units="min")).ravel()[-1])
            except KeyError:
                continue

            x_phase = np.linspace(elapsed_time_min, elapsed_time_min + duration_min, len(y_phase))
            finite_mask = np.isfinite(x_phase) & np.isfinite(y_phase)
            if np.any(finite_mask):
                segments.append((phase, x_phase[finite_mask], y_phase[finite_mask]))

            elapsed_time_min += duration_min

        return segments

    def get_range_domain_segments(phases, y_var, y_units):
        segments = []
        for phase in phases:
            try:
                x_phase = np.asarray(prob.get_val(f"{phase}.range", units="nmi")).ravel()
                y_phase = np.asarray(prob.get_val(f"{phase}.{y_var}", units=y_units)).ravel()
            except KeyError:
                continue

            finite_mask = np.isfinite(x_phase) & np.isfinite(y_phase)
            if np.any(finite_mask):
                segments.append((phase, x_phase[finite_mask], y_phase[finite_mask]))

        return segments

    def get_time_domain_delta_segments(phases, var_before, var_after, y_units):
        before_segments = {phase: (x, y) for phase, x, y in get_time_domain_segments(phases, var_before, y_units)}
        after_segments = {phase: (x, y) for phase, x, y in get_time_domain_segments(phases, var_after, y_units)}

        segments = []
        for phase in phases:
            if phase not in before_segments or phase not in after_segments:
                continue
            x_before, y_before = before_segments[phase]
            _, y_after = after_segments[phase]
            n_pts = min(len(y_before), len(y_after))
            if n_pts == 0:
                continue
            segments.append((phase, x_before[:n_pts], y_after[:n_pts] - y_before[:n_pts]))

        return segments

    for idx_fig, var in enumerate(mission_vars):
        mission_axs[idx_fig].set_xlabel("Range (nmi)")
        y_label = f"{var['name']}" if var["units"] is None else f"{var['name']} ({var['units']})"
        mission_axs[idx_fig].set_ylabel(y_label)

        for phase, x_val, y_val in get_range_domain_segments(power_phases, var["var"], var["units"]):
            mission_axs[idx_fig].plot(
                x_val, y_val, "-o", c=phase_colors[phase], markersize=2.0, label=phase.capitalize()
            )

        mission_axs[idx_fig].grid(True, linestyle="--", alpha=0.5)
        mission_axs[idx_fig].legend(fontsize=8)

    mission_fig.suptitle("Full Mission Profile (PC-12 NGX H2)")
    mission_fig.savefig("pc12ngx_mission_results.svg", transparent=True)

    # Time history of fuel-cell power and auxiliary-component power.
    power_fig, power_axs = plt.subplots(2, 2, figsize=[10.0, 6.8], constrained_layout=True)
    power_axs = power_axs.flatten()

    power_plots = [
        {
            "var": "propmodel.fuelcell.elec_power_req",
            "units": "kW",
            "ylabel": "Fuel Cell Power Required (kW)",
            "title": "Fuel Cell Electrical Power Requirement",
        },
        {
            "var": "propmodel.fuelcell.net_bop_power",
            "units": "kW",
            "ylabel": "Net BoP Power (kW)",
            "title": "Net BoP Power",
        },
        {
            "var": "propmodel.bop.compressor_power_demand",
            "units": "kW",
            "ylabel": "Compressor Power (kW)",
            "title": "Compressor Power",
        },
        {
            "var": "propmodel.bop.turbine_power_recovery",
            "units": "kW",
            "ylabel": "Turbine Power (kW)",
            "title": "Turbine Power Recovery",
        },
    ]

    for ax, plot_data in zip(power_axs, power_plots):
        for phase, time_values, y_values in get_time_domain_segments(
            power_phases, plot_data["var"], plot_data["units"]
        ):
            ax.plot(
                time_values,
                y_values,
                "-o",
                c=phase_colors[phase],
                markersize=2.5,
                label=phase.capitalize(),
            )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel(plot_data["ylabel"])
        ax.set_title(plot_data["title"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    power_fig.suptitle("Power Time-Domain Profile (PC-12 NGX H2)")
    power_fig.savefig("pc12ngx_power_time_domain_results.svg", transparent=True)

    # Time history of heat generation, heat rejection, and system temperatures.
    thermal_fig, thermal_axs = plt.subplots(2, 2, figsize=[10.0, 6.8], constrained_layout=True)
    thermal_axs = thermal_axs.flatten()

    thermal_plots = [
        {
            "var": "propmodel.fuelcell.heat_output",
            "units": "kW",
            "ylabel": "Fuel Cell Heat Output (kW)",
            "title": "Fuel Cell Heat Produced",
        },
        {
            "var": "propmodel.hx.heat_transfer",
            "units": "kW",
            "ylabel": "Heat Exchanger Heat Transfer (kW)",
            "title": "Heat Dissipated by Heat Exchanger",
        },
        {
            "var": "propmodel.fc_thermal.T",
            "units": "degC",
            "ylabel": "Fuel Cell Temperature (degC)",
            "title": "Fuel Cell Temperature",
        },
        {
            "var": "propmodel.coolant_reservoir.T_out",
            "units": "degC",
            "ylabel": "Fluid Temperature (degC)",
            "title": "Coolant Fluid Temperature",
        },
    ]

    for ax, plot_data in zip(thermal_axs, thermal_plots):
        for phase, time_values, y_values in get_time_domain_segments(
            power_phases, plot_data["var"], plot_data["units"]
        ):
            ax.plot(
                time_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize()
            )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel(plot_data["ylabel"])
        ax.set_title(plot_data["title"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    thermal_fig.suptitle("Thermal Time-Domain Profile (PC-12 NGX H2)")
    thermal_fig.savefig("pc12ngx_thermal_time_domain_results.svg", transparent=True)

    # Coolant and air temperatures upstream and downstream of the heat exchanger.
    hx_duct_temps_fig, hx_duct_temps_axs = plt.subplots(2, 2, figsize=[10.0, 6.8], constrained_layout=True)
    hx_duct_temps_axs = hx_duct_temps_axs.flatten()

    hx_duct_temps_plots = [
        {
            "var": "propmodel.hx.T_in_hot",
            "units": "degC",
            "ylabel": "Coolant Temperature (degC)",
            "title": "Coolant Temperature Before HX",
        },
        {
            "var": "propmodel.hx.T_out_hot",
            "units": "degC",
            "ylabel": "Coolant Temperature (degC)",
            "title": "Coolant Temperature After HX",
        },
        {
            "var": "fltcond|T",
            "units": "degC",
            "ylabel": "Air Temperature (degC)",
            "title": "Duct Inlet Air Temperature",
        },
        {
            "var": "propmodel.hx.T_out_cold",
            "units": "degC",
            "ylabel": "Air Temperature (degC)",
            "title": "Duct Outlet Air Temperature",
        },
    ]

    for ax, plot_data in zip(hx_duct_temps_axs, hx_duct_temps_plots):
        for phase, time_values, y_values in get_time_domain_segments(
            power_phases, plot_data["var"], plot_data["units"]
        ):
            ax.plot(
                time_values,
                y_values,
                "-o",
                c=phase_colors[phase],
                markersize=2.5,
                label=phase.capitalize(),
            )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel(plot_data["ylabel"])
        ax.set_title(plot_data["title"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    hx_duct_temps_fig.suptitle("HX and Duct Temperatures (PC-12 NGX H2)")
    hx_duct_temps_fig.savefig("pc12ngx_hx_duct_temperatures_vs_time.svg", transparent=True)

    # Air temperature at the heat-exchanger and duct inlet and outlet locations.
    hx_air_temp_fig, hx_air_temp_axs = plt.subplots(1, 3, figsize=[13.0, 4.2], constrained_layout=True)
    hx_air_temp_plots = [
        {"var": "fltcond|T", "title": "HX Air Inlet"},
        {"var": "propmodel.hx.T_out_cold", "title": "HX Air Outlet"},
        {"var": "propmodel.hx.T_out_cold", "title": "Duct/Nozzle Outlet"},
    ]

    for ax, plot_data in zip(hx_air_temp_axs, hx_air_temp_plots):
        for phase, time_values, y_values in get_time_domain_segments(power_phases, plot_data["var"], "degC"):
            ax.plot(
                time_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize()
            )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Air Temperature (degC)")
        ax.set_title(plot_data["title"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    hx_air_temp_fig.suptitle("Cooling-Air Temperature Through HX and Duct")
    hx_air_temp_fig.savefig("pc12ngx_hx_air_temperature_path.svg", transparent=True)

    # Mass flow rate of air through the cooling duct.
    mdot_fig, mdot_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)

    for phase, time_values, y_values in get_time_domain_segments(
        power_phases, "propmodel.cooling_duct.mdot", "kg/s"
    ):
        mdot_ax.plot(
            time_values,
            y_values,
            "-o",
            c=phase_colors[phase],
            markersize=2.5,
            label=phase.capitalize(),
        )

    mdot_ax.set_xlabel("Time (min)")
    mdot_ax.set_ylabel("Cooling Duct Air Mass Flow (kg/s)")
    mdot_ax.set_title("Cooling Duct Air Mass Flow")
    mdot_ax.grid(True, linestyle="--", alpha=0.5)
    mdot_ax.legend(fontsize=8)
    mdot_fig.savefig("pc12ngx_cooling_duct_mdot_vs_time.svg", transparent=True)

    # Heat removed by the evaporator and the resulting load on the thermal management system.
    evaporator_fig, evaporator_axs = plt.subplots(1, 2, figsize=[10.0, 4.0], constrained_layout=True)
    evaporator_axs = evaporator_axs.flatten()

    evaporator_plots = [
        {
            "var": "propmodel.evaporator.heat_removed",
            "units": "kW",
            "ylabel": "Heat Removed by LH2 (kW)",
            "title": "Heat Removed by Evaporator",
        },
        {
            "var": "propmodel.evaporator.effective_heat_load",
            "units": "kW",
            "ylabel": "Effective Heat Load to TMS (kW)",
            "title": "Effective Heat Load After Evaporator",
        },
    ]

    for ax, plot_data in zip(evaporator_axs, evaporator_plots):
        for phase, time_values, y_values in get_time_domain_segments(
            power_phases, plot_data["var"], plot_data["units"]
        ):
            ax.plot(
                time_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize()
            )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel(plot_data["ylabel"])
        ax.set_title(plot_data["title"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    evaporator_fig.suptitle("Evaporator Thermal Response (PC-12 NGX H2)")
    evaporator_fig.savefig("pc12ngx_evaporator_results.svg", transparent=True)

    # Total, airframe, and cooling-system drag breakdown versus mission distance.
    drag_fig, drag_axs = plt.subplots(2, 3, figsize=[13.0, 7.2], constrained_layout=True)
    drag_axs = drag_axs.flatten()

    drag_plots = [
        {
            "var": "drag",
            "units": "N",
            "ylabel": "Total Drag (N)",
            "title": "Total Drag",
        },
        {
            "var": "drag_airframe",
            "units": "N",
            "ylabel": "Airframe Drag (N)",
            "title": "Airframe Drag",
        },
        {
            "var": "propmodel.cooling_duct.drag",
            "units": "N",
            "ylabel": "Heat Exchanger Drag (N)",
            "title": "Heat Exchanger Drag",
        },
        {
            "var": "propmodel.cooling_duct.spillage_drag",
            "units": "N",
            "ylabel": "Spillage Drag (N)",
            "title": "Inlet Spillage Drag",
        },
        {
            "var": "propmodel.cooling_duct.internal_drag",
            "units": "N",
            "ylabel": "Internal Drag (N)",
            "title": "Internal Duct Drag",
        },
    ]

    for ax, plot_data in zip(drag_axs[:5], drag_plots):
        for phase, x_values, y_values in get_range_domain_segments(power_phases, plot_data["var"], plot_data["units"]):
            ax.plot(x_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize())
        ax.set_xlabel("Range (nmi)")
        ax.set_ylabel(plot_data["ylabel"])
        ax.set_title(plot_data["title"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    # The sixth panel shows the heat exchanger's percentage of total drag.
    drag_total_by_phase = {phase: (x, y) for phase, x, y in get_range_domain_segments(power_phases, "drag", "N")}
    drag_hx_by_phase = {
        phase: (x, y) for phase, x, y in get_range_domain_segments(power_phases, "propmodel.cooling_duct.drag", "N")
    }
    for phase in power_phases:
        if phase not in drag_total_by_phase or phase not in drag_hx_by_phase:
            continue
        x_total, y_total = drag_total_by_phase[phase]
        _, y_hx = drag_hx_by_phase[phase]
        common_len = min(len(y_total), len(y_hx), len(x_total))
        if not common_len:
            continue
        drag_share_pct = 100.0 * y_hx[:common_len] / np.maximum(y_total[:common_len], 1e-9)
        finite_mask = np.isfinite(x_total[:common_len]) & np.isfinite(drag_share_pct)
        if np.any(finite_mask):
            drag_axs[5].plot(
                x_total[:common_len][finite_mask],
                drag_share_pct[finite_mask],
                "-o",
                c=phase_colors[phase],
                markersize=2.5,
                label=phase.capitalize(),
            )

    drag_axs[5].set_xlabel("Range (nmi)")
    drag_axs[5].set_ylabel("HX Drag Share (%)")
    drag_axs[5].set_title("Heat Exchanger Drag Share")
    drag_axs[5].grid(True, linestyle="--", alpha=0.5)
    drag_axs[5].legend(fontsize=8)

    drag_fig.suptitle("Drag Breakdown vs. Mission Range")
    drag_fig.savefig("pc12ngx_drag_breakdown_vs_range.svg", transparent=True)

    # Cooling-duct inlet and outlet areas across the mission phases.
    area_fig, area_axs = plt.subplots(1, 2, figsize=[13.0, 4.2], constrained_layout=True)
    area_plots = [
        {
            "var": "propmodel.cooling_duct.area_inlet",
            "title": "Inlet Area",
            "fixed_vector": inlet_area_vector,
        },
        {
            "var": "propmodel.cooling_duct.area_nozzle",
            "title": "Outlet/Nozzle Area",
            "fixed_vector": outlet_area_vector,
        },
    ]
    for ax, plot_data in zip(area_axs, area_plots):
        for phase, x_values, y_values in get_range_domain_segments(power_phases, plot_data["var"], "m**2"):
            if args.no_optimization and y_values.size:
                # In fixed mode, plot the same constant value reported in the printed vector.
                fixed_value = float(np.nanmedian(plot_data["fixed_vector"]))
                y_values = np.full(x_values.shape, fixed_value, dtype=float)
            ax.plot(x_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize())

        ax.set_xlabel("Range (nmi)")
        ax.set_ylabel("Area (m^2)")
        ax.set_title(plot_data["title"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    area_fig.suptitle("Inlet and Outlet Areas vs. Mission Range")
    area_fig.savefig("pc12ngx_inlet_outlet_area_vs_range.svg", transparent=True)

    # Thrust produced during each phase versus mission distance.
    thrust_fig, thrust_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)

    for phase in power_phases:
        try:
            x_phase = np.asarray(prob.get_val(f"{phase}.range", units="nmi")).ravel()
            try:
                y_phase = np.asarray(prob.get_val(f"{phase}.thrust", units="N")).ravel()
            except KeyError:
                y_phase = np.asarray(prob.get_val(f"{phase}.propmodel.thrust", units="N")).ravel()

            finite_mask = np.isfinite(x_phase) & np.isfinite(y_phase)
            if np.any(finite_mask):
                thrust_ax.plot(
                    x_phase[finite_mask],
                    y_phase[finite_mask],
                    "-o",
                    c=phase_colors[phase],
                    markersize=2.5,
                    label=phase.capitalize(),
                )
        except KeyError:
            pass

    thrust_ax.set_xlabel("Range (nmi)")
    thrust_ax.set_ylabel("Thrust (N)")
    thrust_ax.set_title("Thrust vs. Mission Range")
    thrust_ax.grid(True, linestyle="--", alpha=0.5)
    thrust_ax.legend(fontsize=8)
    thrust_fig.savefig("pc12ngx_thrust_vs_range.svg", transparent=True)

    # Throttle position versus mission distance.
    throttle_fig, throttle_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    for phase, x_values, y_values in get_range_domain_segments(power_phases, "throttle", None):
        throttle_ax.plot(x_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize())
    throttle_ax.set_xlabel("Range (nmi)")
    throttle_ax.set_ylabel("Throttle")
    throttle_ax.set_title("Throttle vs. Mission Range")
    throttle_ax.grid(True, linestyle="--", alpha=0.5)
    throttle_ax.legend()
    throttle_fig.savefig("pc12ngx_throttle_vs_range.svg", transparent=True)

    # Electrical power required from the fuel cell throughout the mission.
    fc_power_fig, fc_power_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    for phase, time_values, y_values in get_time_domain_segments(
        power_phases, "propmodel.fuelcell.elec_power_req", "kW"
    ):
        fc_power_ax.plot(
            time_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize()
        )
    fc_power_ax.set_xlabel("Time (min)")
    fc_power_ax.set_ylabel("Fuel Cell Power Required (kW)")
    fc_power_ax.set_title("Fuel Cell Electrical Power Requirement")
    fc_power_ax.grid(True, linestyle="--", alpha=0.5)
    fc_power_ax.legend()
    fc_power_fig.savefig("pc12ngx_fuel_cell_power_requirement.svg", transparent=True)

    # Total aircraft drag versus mission distance.
    total_drag_fig, total_drag_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    for phase, x_values, y_values in get_range_domain_segments(power_phases, "drag", "N"):
        total_drag_ax.plot(x_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize())
    total_drag_ax.set_xlabel("Range (nmi)")
    total_drag_ax.set_ylabel("Total Drag (N)")
    total_drag_ax.set_title("Total Drag vs. Mission Range")
    total_drag_ax.grid(True, linestyle="--", alpha=0.5)
    total_drag_ax.legend()
    total_drag_fig.savefig("pc12ngx_total_drag_standalone.svg", transparent=True)

    # Fuel-cell temperature throughout the mission.
    fc_temp_fig, fc_temp_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    for phase, time_values, y_values in get_time_domain_segments(power_phases, "propmodel.fc_thermal.T", "degC"):
        fc_temp_ax.plot(time_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize())
    fc_temp_ax.set_xlabel("Time (min)")
    fc_temp_ax.set_ylabel("Fuel Cell Temperature (degC)")
    fc_temp_ax.set_title("Fuel Cell Temperature")
    fc_temp_ax.grid(True, linestyle="--", alpha=0.5)
    fc_temp_ax.legend()
    fc_temp_fig.savefig("pc12ngx_fuel_cell_temperature_standalone.svg", transparent=True)

    # Coolant temperature change across the heat exchanger.
    coolant_delta_fig, coolant_delta_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    for phase, time_values, y_values in get_time_domain_delta_segments(
        power_phases, "propmodel.hx.T_in_hot", "propmodel.hx.T_out_hot", "degC"
    ):
        coolant_delta_ax.plot(time_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize())
    coolant_delta_ax.set_xlabel("Time (min)")
    coolant_delta_ax.set_ylabel("Delta T (degC)")
    coolant_delta_ax.set_title("Coolant Temperature Drop Across the Heat Exchanger")
    coolant_delta_ax.grid(True, linestyle="--", alpha=0.5)
    coolant_delta_ax.legend()
    coolant_delta_fig.savefig("pc12ngx_coolant_delta_t_standalone.svg", transparent=True)

    # Air temperature change across the heat exchanger; fltcond|T is the cold-side inlet.
    hx_delta_fig, hx_delta_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    for phase, time_values, y_values in get_time_domain_delta_segments(
        power_phases, "fltcond|T", "propmodel.hx.T_out_cold", "degC"
    ):
        hx_delta_ax.plot(time_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize())
    hx_delta_ax.set_xlabel("Time (min)")
    hx_delta_ax.set_ylabel("Delta T (outlet - inlet, degC)")
    hx_delta_ax.set_title("Heat Exchanger Air Temperature Rise")
    hx_delta_ax.grid(True, linestyle="--", alpha=0.5)
    hx_delta_ax.legend()
    hx_delta_fig.savefig("pc12ngx_hx_air_delta_t_standalone.svg", transparent=True)

    # The duct outlet temperature is represented by the HX cold-side outlet temperature.
    duct_out_temp_fig, duct_out_temp_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    for phase, time_values, y_values in get_time_domain_segments(
        power_phases, "propmodel.hx.T_out_cold", "degC"
    ):
        duct_out_temp_ax.plot(
            time_values, y_values, "-o", c=phase_colors[phase], markersize=2.5, label=phase.capitalize()
        )
    duct_out_temp_ax.set_xlabel("Time (min)")
    duct_out_temp_ax.set_ylabel("Air Temperature (degC)")
    duct_out_temp_ax.set_title("Duct Outlet Air Temperature")
    duct_out_temp_ax.grid(True, linestyle="--", alpha=0.5)
    duct_out_temp_ax.legend()
    duct_out_temp_fig.savefig("pc12ngx_duct_outlet_air_temperature_standalone.svg", transparent=True)

    # Duct inlet air temperature, shown using a single color.
    duct_in_temp_fig, duct_in_temp_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    duct_in_time_min, duct_in_temp_c = get_time_domain_series(power_phases, "fltcond|T", "degC")
    if duct_in_time_min is not None:
        duct_in_temp_ax.plot(duct_in_time_min, duct_in_temp_c, "-", color="tab:blue", linewidth=1.8)
    duct_in_temp_ax.set_xlabel("Time (min)")
    duct_in_temp_ax.set_ylabel("Air Temperature (degC)")
    duct_in_temp_ax.set_title("Duct Inlet Air Temperature")
    duct_in_temp_ax.grid(True, linestyle="--", alpha=0.5)
    duct_in_temp_fig.savefig("pc12ngx_duct_inlet_air_temperature_standalone.svg", transparent=True)

    # Optional n_long_cold sensitivity analysis and its thermal plots (currently commented out).
    # if args.no_optimization:
    #     n_long_cold_sweep_values = [5, 10, 15, 20, 25]
    #     sweep_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    #     original_n_long_cold = float(np.asarray(prob.get_val("ac|propulsion|thermal|hx|n_long_cold")).ravel()[-1])
    #
    #     fc_temp_sweep_fig, fc_temp_sweep_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    #     coolant_delta_sweep_fig, coolant_delta_sweep_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    #     duct_out_sweep_fig, duct_out_sweep_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    #
    #     for n_long_cold_val, color in zip(n_long_cold_sweep_values, sweep_colors):
    #         prob.set_val("ac|propulsion|thermal|hx|n_long_cold", n_long_cold_val)
    #         prob.run_model()
    #         label = f"n_long_cold = {n_long_cold_val}"
    #
    #         time_min, fc_temp_c = get_time_domain_series(power_phases, "propmodel.fc_thermal.T", "degC")
    #         if time_min is not None:
    #             fc_temp_sweep_ax.plot(time_min, fc_temp_c, "-", color=color, linewidth=1.8, label=label)
    #
    #         time_min_hot, t_hot_in = get_time_domain_series(power_phases, "propmodel.hx.T_in_hot", "degC")
    #         _, t_hot_out = get_time_domain_series(power_phases, "propmodel.hx.T_out_hot", "degC")
    #         if time_min_hot is not None and t_hot_out is not None:
    #             n_pts = min(len(t_hot_in), len(t_hot_out))
    #             coolant_delta_sweep_ax.plot(
    #                 time_min_hot[:n_pts],
    #                 t_hot_out[:n_pts] - t_hot_in[:n_pts],
    #                 "-",
    #                 color=color,
    #                 linewidth=1.8,
    #                 label=label,
    #             )
    #
    #         time_min_duct, duct_out_temp_c = get_time_domain_series(power_phases, "propmodel.hx.T_out_cold", "degC")
    #         if time_min_duct is not None:
    #             duct_out_sweep_ax.plot(time_min_duct, duct_out_temp_c, "-", color=color, linewidth=1.8, label=label)
    #
    #     # Restore the original fixed value and reevaluate the model.
    #     prob.set_val("ac|propulsion|thermal|hx|n_long_cold", original_n_long_cold)
    #     prob.run_model()
    #
    #     fc_temp_sweep_ax.set_xlabel("Time (min)")
    #     fc_temp_sweep_ax.set_ylabel("Fuel Cell Temperature (degC)")
    #     fc_temp_sweep_ax.set_title("Fuel Cell Temperature")
    #     fc_temp_sweep_ax.grid(True, linestyle="--", alpha=0.5)
    #     fc_temp_sweep_ax.legend()
    #     fc_temp_sweep_fig.savefig("pc12ngx_fc_temp_vs_n_long_cold_sweep.svg", transparent=True)
    #
    #     coolant_delta_sweep_ax.set_xlabel("Time (min)")
    #     coolant_delta_sweep_ax.set_ylabel("Delta T (degC)")
    #     coolant_delta_sweep_ax.set_title(
    #         "Coolant Temperature Drop Across the Heat Exchanger"
    #     )
    #     coolant_delta_sweep_ax.grid(True, linestyle="--", alpha=0.5)
    #     coolant_delta_sweep_ax.legend()
    #     coolant_delta_sweep_fig.savefig("pc12ngx_coolant_delta_t_vs_n_long_cold_sweep.svg", transparent=True)
    #
    #     duct_out_sweep_ax.set_xlabel("Time (min)")
    #     duct_out_sweep_ax.set_ylabel("Air Temperature (degC)")
    #     duct_out_sweep_ax.set_title("Duct Outlet Air Temperature")
    #     duct_out_sweep_ax.grid(True, linestyle="--", alpha=0.5)
    #     duct_out_sweep_ax.legend()
    #     duct_out_sweep_fig.savefig("pc12ngx_duct_outlet_air_temperature_vs_n_long_cold_sweep.svg", transparent=True)

    # Optional n_tall sensitivity analysis for temperature, airflow, and HX drag (currently commented out).
    # if args.no_optimization:
    #     n_tall_sweep_values = [10, 50, 90, 130, 170]
    #     n_tall_sweep_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    #     original_n_tall = float(np.asarray(prob.get_val("ac|propulsion|thermal|hx|n_tall")).ravel()[-1])
    #
    #     n_tall_fc_temp_fig, n_tall_fc_temp_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    #     n_tall_coolant_delta_fig, n_tall_coolant_delta_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    #     n_tall_mdot_fig, n_tall_mdot_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    #     n_tall_drag_fig, n_tall_drag_ax = plt.subplots(figsize=[7.2, 3.6], constrained_layout=True)
    #
    #     for n_tall_val, color in zip(n_tall_sweep_values, n_tall_sweep_colors):
    #         prob.set_val("ac|propulsion|thermal|hx|n_tall", n_tall_val)
    #         prob.run_model()
    #         label = f"n_tall = {n_tall_val}"
    #
    #         time_min, fc_temp_c = get_time_domain_series(power_phases, "propmodel.fc_thermal.T", "degC")
    #         if time_min is not None:
    #             n_tall_fc_temp_ax.plot(time_min, fc_temp_c, "-", color=color, linewidth=1.8, label=label)
    #
    #         time_min_hot, t_hot_in = get_time_domain_series(power_phases, "propmodel.hx.T_in_hot", "degC")
    #         _, t_hot_out = get_time_domain_series(power_phases, "propmodel.hx.T_out_hot", "degC")
    #         if time_min_hot is not None and t_hot_out is not None:
    #             n_pts = min(len(t_hot_in), len(t_hot_out))
    #             n_tall_coolant_delta_ax.plot(
    #                 time_min_hot[:n_pts],
    #                 t_hot_out[:n_pts] - t_hot_in[:n_pts],
    #                 "-",
    #                 color=color,
    #                 linewidth=1.8,
    #                 label=label,
    #             )
    #
    #         time_min_mdot, mdot_kgs = get_time_domain_series(power_phases, "propmodel.cooling_duct.mdot", "kg/s")
    #         if time_min_mdot is not None:
    #             n_tall_mdot_ax.plot(time_min_mdot, mdot_kgs, "-", color=color, linewidth=1.8, label=label)
    #
    #         time_min_drag, drag_n = get_time_domain_series(power_phases, "propmodel.cooling_duct.drag", "N")
    #         if time_min_drag is not None:
    #             n_tall_drag_ax.plot(time_min_drag, drag_n, "-", color=color, linewidth=1.8, label=label)
    #
    #     # Restore the original fixed value and reevaluate the model.
    #     prob.set_val("ac|propulsion|thermal|hx|n_tall", original_n_tall)
    #     prob.run_model()
    #
    #     n_tall_fc_temp_ax.set_xlabel("Time (min)")
    #     n_tall_fc_temp_ax.set_ylabel("Fuel Cell Temperature (degC)")
    #     n_tall_fc_temp_ax.set_title("Fuel Cell Temperature")
    #     n_tall_fc_temp_ax.grid(True, linestyle="--", alpha=0.5)
    #     n_tall_fc_temp_ax.legend()
    #     n_tall_fc_temp_fig.savefig("pc12ngx_fc_temp_vs_n_tall_sweep.svg", transparent=True)
    #
    #     n_tall_coolant_delta_ax.set_xlabel("Time (min)")
    #     n_tall_coolant_delta_ax.set_ylabel("Delta T (degC)")
    #     n_tall_coolant_delta_ax.set_title(
    #         "Coolant Temperature Drop Across the Heat Exchanger"
    #     )
    #     n_tall_coolant_delta_ax.grid(True, linestyle="--", alpha=0.5)
    #     n_tall_coolant_delta_ax.legend()
    #     n_tall_coolant_delta_fig.savefig("pc12ngx_coolant_delta_t_vs_n_tall_sweep.svg", transparent=True)
    #
    #     n_tall_mdot_ax.set_xlabel("Time (min)")
    #     n_tall_mdot_ax.set_ylabel("Cooling Duct Air Mass Flow (kg/s)")
    #     n_tall_mdot_ax.set_title("Cooling Duct Air Mass Flow")
    #     n_tall_mdot_ax.grid(True, linestyle="--", alpha=0.5)
    #     n_tall_mdot_ax.legend()
    #     n_tall_mdot_fig.savefig("pc12ngx_mdot_vs_n_tall_sweep.svg", transparent=True)
    #
    #     n_tall_drag_ax.set_xlabel("Time (min)")
    #     n_tall_drag_ax.set_ylabel("Heat Exchanger Drag (N)")
    #     n_tall_drag_ax.set_title("Heat Exchanger Drag")
    #     n_tall_drag_ax.grid(True, linestyle="--", alpha=0.5)
    #     n_tall_drag_ax.legend()
    #     n_tall_drag_fig.savefig("pc12ngx_hx_drag_vs_n_tall_sweep.svg", transparent=True)

    if not hide_viz:
        plt.show(block=False)
        input("\nPress Enter to close all figures and exit...")
        plt.close("all")