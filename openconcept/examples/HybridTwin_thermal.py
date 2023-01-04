import os
import logging
import numpy as np

from openmdao.api import (
    Problem,
    Group,
    ScipyOptimizeDriver,
    ExplicitComponent,
    ExecComp,
    SqliteRecorder,
    DirectSolver,
    IndepVarComp,
    NewtonSolver,
    CaseReader,
)

# imports for the airplane model itself
from openconcept.aerodynamics import PolarDrag
from openconcept.weights import TwinSeriesHybridEmptyWeight
from openconcept.propulsion import TwinSeriesHybridElectricThermalPropulsionSystem
from openconcept.mission import FullMissionAnalysis
from openconcept.examples.aircraft_data.KingAirC90GT import data as acdata
from openconcept.utilities import (
    AddSubtractComp,
    MaxComp,
    Integrator,
    DictIndepVarComp,
    LinearInterpolator,
    plot_trajectory,
)


class AugmentedFBObjective(ExplicitComponent):
    def setup(self):
        self.add_input("fuel_burn", units="kg")
        self.add_input("ac|weights|MTOW", units="kg")
        self.add_output("mixed_objective", units="kg")
        self.declare_partials(["mixed_objective"], ["fuel_burn"], val=1)
        self.declare_partials(["mixed_objective"], ["ac|weights|MTOW"], val=1 / 100)

    def compute(self, inputs, outputs):
        outputs["mixed_objective"] = inputs["fuel_burn"] + inputs["ac|weights|MTOW"] / 100


class SeriesHybridTwinModel(Group):
    """
    A custom model specific to a series hybrid twin turboprop-class airplane
    This class will be passed in to the mission analysis code.

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]

        # any control variables other than throttle and braking need to be defined here
        controls = self.add_subsystem("controls", IndepVarComp(), promotes_outputs=["*"])
        controls.add_output("proprpm", val=np.ones((nn,)) * 2000, units="rpm")
        controls.add_output("ac|propulsion|thermal|hx|mdot_coolant", val=0.1 * np.ones((nn,)), units="kg/s")

        # assume TO happens on battery backup
        if flight_phase in ["climb", "cruise", "descent"]:
            controls.add_output("hybridization", val=0.0)
        else:
            controls.add_output("hybridization", val=1.0)

        self.add_subsystem(
            "hybrid_factor",
            LinearInterpolator(num_nodes=nn),
            promotes_inputs=[("start_val", "hybridization"), ("end_val", "hybridization")],
        )

        propulsion_promotes_outputs = ["fuel_flow", "thrust", "ac|propulsion|thermal|duct|area_nozzle"]
        propulsion_promotes_inputs = [
            "fltcond|*",
            "ac|propulsion|*",
            "throttle",
            "propulsor_active",
            "ac|weights*",
            "duration",
        ]

        self.add_subsystem(
            "propmodel",
            TwinSeriesHybridElectricThermalPropulsionSystem(num_nodes=nn),
            promotes_inputs=propulsion_promotes_inputs,
            promotes_outputs=propulsion_promotes_outputs,
        )
        self.connect("proprpm", ["propmodel.prop1.rpm", "propmodel.prop2.rpm"])
        self.connect("hybrid_factor.vec", "propmodel.hybrid_split.power_split_fraction")

        # use a different drag coefficient for takeoff versus cruise
        if flight_phase not in ["v0v1", "v1v0", "v1vr", "rotate"]:
            cd0_source = "ac|aero|polar|CD0_cruise"
        else:
            cd0_source = "ac|aero|polar|CD0_TO"
        self.add_subsystem(
            "drag",
            PolarDrag(num_nodes=nn),
            promotes_inputs=["fltcond|CL", "ac|geom|*", ("CD0", cd0_source), "fltcond|q", ("e", "ac|aero|polar|e")],
        )

        self.add_subsystem(
            "OEW", TwinSeriesHybridEmptyWeight(), promotes_inputs=["*", ("P_TO", "ac|propulsion|engine|rating")]
        )
        self.connect("propmodel.propellers_weight", "W_propeller")
        self.connect("propmodel.eng1.component_weight", "W_engine")
        self.connect("propmodel.gen1.component_weight", "W_generator")
        self.connect("propmodel.motors_weight", "W_motors")

        hxadder = AddSubtractComp()
        hxadder.add_equation("OEW", ["OEW_orig", "W_hx", "W_coolant"], scaling_factors=[1, 1, 1], units="kg")
        hxadder.add_equation("drag", ["drag_orig", "drag_hx"], vec_size=nn, units="N", scaling_factors=[1, 1])
        hxadder.add_equation(
            "area_constraint", ["hx_frontal_area", "nozzle_area"], units="m**2", scaling_factors=[1, -1]
        )
        self.add_subsystem(
            "hxadder",
            hxadder,
            promotes_inputs=[("W_coolant", "ac|propulsion|thermal|hx|coolant_mass")],
            promotes_outputs=["OEW", "drag"],
        )
        self.connect("drag.drag", "hxadder.drag_orig")
        self.connect("OEW.OEW", "hxadder.OEW_orig")
        self.connect("propmodel.hx.component_weight", "hxadder.W_hx")
        self.connect("propmodel.duct.drag", "hxadder.drag_hx")
        self.connect("propmodel.hx.frontal_area", "hxadder.hx_frontal_area")
        self.add_subsystem("nozzle_area", MaxComp(num_nodes=nn, units="m**2"))
        self.connect("ac|propulsion|thermal|duct|area_nozzle", "nozzle_area.array")
        self.connect("nozzle_area.max", "hxadder.nozzle_area")
        intfuel = self.add_subsystem(
            "intfuel",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        intfuel.add_integrand("fuel_used", rate_name="fuel_flow", val=1.0, units="kg")
        self.add_subsystem(
            "weight",
            AddSubtractComp(
                output_name="weight",
                input_names=["ac|weights|MTOW", "fuel_used"],
                units="kg",
                vec_size=[1, nn],
                scaling_factors=[1, -1],
            ),
            promotes_inputs=["*"],
            promotes_outputs=["weight"],
        )


class ElectricTwinAnalysisGroup(Group):
    """This is an example of a balanced field takeoff and three-phase mission analysis."""

    def setup(self):
        # Define number of analysis points to run pers mission segment
        nn = 11

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
        dv_comp.add_output_from_dict("ac|geom|fuselage|S_wet")
        dv_comp.add_output_from_dict("ac|geom|fuselage|width")
        dv_comp.add_output_from_dict("ac|geom|fuselage|length")
        dv_comp.add_output_from_dict("ac|geom|fuselage|height")
        dv_comp.add_output_from_dict("ac|geom|nosegear|length")
        dv_comp.add_output_from_dict("ac|geom|maingear|length")

        dv_comp.add_output_from_dict("ac|weights|MTOW")
        dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
        dv_comp.add_output_from_dict("ac|weights|MLW")
        dv_comp.add_output_from_dict("ac|weights|W_battery")

        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output_from_dict("ac|propulsion|propeller|diameter")
        dv_comp.add_output_from_dict("ac|propulsion|generator|rating")
        dv_comp.add_output_from_dict("ac|propulsion|motor|rating")

        dv_comp.add_output("ac|propulsion|thermal|hx|coolant_mass", val=10.0, units="kg")
        dv_comp.add_output("ac|propulsion|thermal|hx|channel_width", val=1.0, units="mm")
        dv_comp.add_output("ac|propulsion|thermal|hx|channel_height", 20.0, units="mm")
        dv_comp.add_output("ac|propulsion|thermal|hx|channel_length", val=0.2, units="m")
        dv_comp.add_output("ac|propulsion|thermal|hx|n_parallel", val=50, units=None)
        # dv_comp.add_output('ac|propulsion|thermal|duct|area_nozzle',val=58.*np.ones((nn,)),units='inch**2')
        dv_comp.add_output("ac|propulsion|thermal|hx|n_wide_cold", val=430, units=None)
        dv_comp.add_output("ac|propulsion|battery|specific_energy", val=300, units="W*h/kg")

        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")
        dv_comp.add_output_from_dict("ac|num_engines")

        mission_data_comp = self.add_subsystem("mission_data_comp", IndepVarComp(), promotes_outputs=["*"])
        mission_data_comp.add_output("batt_soc_target", val=0.1, units=None)
        mission_data_comp.add_output("T_motor_initial", val=15, units="degC")
        mission_data_comp.add_output("T_res_initial", val=15.1, units="degC")
        mission_data_comp.add_output("T_batt_initial", val=10.1, units="degC")

        # Ensure that any state variables are connected across the mission as intended
        self.add_subsystem(
            "analysis",
            FullMissionAnalysis(num_nodes=nn, aircraft_model=SeriesHybridTwinModel),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.add_subsystem(
            "margins",
            ExecComp(
                "MTOW_margin = MTOW - OEW - total_fuel - W_battery - payload",
                MTOW_margin={"units": "lbm", "val": 100},
                MTOW={"units": "lbm", "val": 10000},
                OEW={"units": "lbm", "val": 5000},
                total_fuel={"units": "lbm", "val": 1000},
                W_battery={"units": "lbm", "val": 1000},
                payload={"units": "lbm", "val": 1000},
            ),
            promotes_inputs=["payload"],
        )
        self.connect("cruise.OEW", "margins.OEW")
        self.connect("descent.fuel_used_final", "margins.total_fuel")
        self.connect("ac|weights|MTOW", "margins.MTOW")
        self.connect("ac|weights|W_battery", "margins.W_battery")

        self.add_subsystem("aug_obj", AugmentedFBObjective(), promotes_outputs=["mixed_objective"])
        self.connect("ac|weights|MTOW", "aug_obj.ac|weights|MTOW")
        self.connect("descent.fuel_used_final", "aug_obj.fuel_burn")

        self.connect("T_motor_initial", "v0v1.propmodel.motorheatsink.T_initial")
        self.connect("T_res_initial", "v0v1.propmodel.reservoir.T_initial")
        self.connect("T_batt_initial", "v0v1.propmodel.batteryheatsink.T_initial")


def configure_problem():
    prob = Problem()
    prob.model = ElectricTwinAnalysisGroup()
    prob.model.nonlinear_solver = NewtonSolver(iprint=2)
    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options["solve_subsystems"] = True
    prob.model.nonlinear_solver.options["maxiter"] = 10
    prob.model.nonlinear_solver.options["atol"] = 1e-8
    prob.model.nonlinear_solver.options["rtol"] = 1e-8
    return prob


def set_values(prob, num_nodes, design_range, spec_energy):
    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val("climb.fltcond|vs", np.ones((num_nodes,)) * 1500, units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.ones((num_nodes,)) * 124, units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 0.01, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.ones((num_nodes,)) * 170, units="kn")
    prob.set_val("descent.fltcond|vs", np.ones((num_nodes,)) * (-600), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.ones((num_nodes,)) * 140, units="kn")

    prob.set_val("cruise|h0", 29000, units="ft")
    prob.set_val("mission_range", design_range, units="NM")
    prob.set_val("payload", 1000, units="lb")
    prob.set_val("ac|propulsion|battery|specific_energy", spec_energy, units="W*h/kg")

    # (optional) guesses for takeoff speeds may help with convergence
    prob.set_val("v0v1.fltcond|Utrue", np.ones((num_nodes)) * 50, units="kn")
    prob.set_val("v1vr.fltcond|Utrue", np.ones((num_nodes)) * 85, units="kn")
    prob.set_val("v1v0.fltcond|Utrue", np.ones((num_nodes)) * 85, units="kn")

    # set some airplane-specific values
    prob["analysis.cruise.acmodel.OEW.const.structural_fudge"] = 2.0
    prob["ac|propulsion|propeller|diameter"] = 2.2
    prob["ac|propulsion|engine|rating"] = 1117.2


def run_hybrid_twin_thermal_analysis(plots=False):
    prob = configure_problem()
    prob.setup(check=False)
    prob["cruise.hybridization"] = 0.05778372636876463
    set_values(prob, 11, 500, 450)
    prob.run_model()
    if plots:
        show_outputs(prob)
    return prob


def show_outputs(prob):
    # print some outputs
    vars_list = [
        "ac|weights|MTOW",
        "climb.OEW",
        "descent.fuel_used_final",
        "rotate.range_final",
        "descent.propmodel.batt1.SOC_final",
        "cruise.hybridization",
        "ac|weights|W_battery",
        "margins.MTOW_margin",
        "ac|propulsion|motor|rating",
        "ac|propulsion|generator|rating",
        "ac|propulsion|engine|rating",
        "ac|geom|wing|S_ref",
        "v0v1.Vstall_eas",
        "v0v1.takeoff|vr",
        "engineoutclimb.gamma",
        "cruise.propmodel.duct.drag",
        "ac|propulsion|thermal|hx|coolant_mass",
        "climb.propmodel.duct.mdot",
    ]
    units = [
        "lb",
        "lb",
        "lb",
        "ft",
        None,
        None,
        "lb",
        "lb",
        "hp",
        "hp",
        "hp",
        "ft**2",
        "kn",
        "kn",
        "deg",
        "lbf",
        "lb",
        "lb/s",
    ]
    nice_print_names = [
        "MTOW",
        "OEW",
        "Fuel used",
        "TOFL (over 35ft obstacle)",
        "Final state of charge",
        "Cruise hybridization",
        "Battery weight",
        "MTOW margin",
        "Motor rating",
        "Generator rating",
        "Engine rating",
        "Wing area",
        "Stall speed",
        "Rotate speed",
        "Engine out climb angle",
        "Coolant duct cruise drag",
        "Coolant mass",
        "Coolant duct mass flow",
    ]
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i] + ": " + str(prob.get_val(thing, units=units[i])[0]) + " " + str(units[i]))

    print("Motor temps (climb): " + str(prob.get_val("climb.propmodel.motorheatsink.T", units="degC")))
    print("Battery temps (climb): " + str(prob.get_val("climb.propmodel.batteryheatsink.T", units="degC")))

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
            "propmodel.batt1.SOC",
            "propmodel.motorheatsink.T",
            "propmodel.batteryheatsink.T",
            "propmodel.reservoir.T_out",
        ]
        y_units = ["ft", "kn", "lbm", None, "ft/min", None, "degC", "degC", "degC"]
        x_label = "Range (nmi)"
        y_labels = [
            "Altitude (ft)",
            "Veas airspeed (knots)",
            "Fuel used (lb)",
            "Throttle setting",
            "Vertical speed (ft/min)",
            "Battery SOC",
            "Motor temp",
            "Battery temp",
            "Reservoir outlet temp",
        ]
        phases = ["v0v1", "v1vr", "v1v0", "rotate"]
        plot_trajectory(
            prob,
            x_var,
            x_unit,
            y_vars,
            y_units,
            phases,
            x_label=x_label,
            y_labels=y_labels,
            marker="o",
            plot_title="Takeoff Profile",
        )

        phases = ["v0v1", "v1vr", "rotate", "climb", "cruise", "descent"]
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
            plot_title="Full Mission Profile",
        )


if __name__ == "__main__":
    # for run type choose choose optimization, comp_sizing, or analysis
    run_type = "example"
    num_nodes = 11

    if run_type == "example":
        # runs a default analysis-only mission (no optimization)
        run_hybrid_twin_thermal_analysis(plots=True)

    else:
        # can run a sweep of design range and spec energy (not tested)
        # design_ranges = [300,350,400,450,500,550,600,650,700]
        # specific_energies = [250,300,350,400,450,500,550,600,650,700,750,800]

        # or a single point
        design_ranges = [500]
        specific_energies = [450]

        write_logs = False
        if write_logs:
            logging.basicConfig(filename="opt.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s")
        last_successful_opt = None

        # run a sweep of cases at various specific energies and ranges
        for this_spec_energy in specific_energies:
            for design_range in design_ranges:
                try:
                    prob = configure_problem()
                    spec_energy = this_spec_energy
                    if run_type == "optimization":
                        print("======Performing Multidisciplinary Design Optimization===========")
                        prob.model.add_design_var("ac|weights|MTOW", lower=4000, upper=5700)
                        prob.model.add_design_var("ac|geom|wing|S_ref", lower=15, upper=40)
                        prob.model.add_design_var("ac|propulsion|engine|rating", lower=1, upper=3000)
                        prob.model.add_design_var("ac|propulsion|motor|rating", lower=450, upper=3000)
                        prob.model.add_design_var("ac|propulsion|generator|rating", lower=1, upper=3000)
                        prob.model.add_design_var("ac|weights|W_battery", lower=20, upper=2250)
                        prob.model.add_design_var("ac|weights|W_fuel_max", lower=500, upper=3000)
                        prob.model.add_design_var("cruise.hybridization", lower=0.001, upper=0.999)
                        prob.model.add_design_var("climb.hybridization", lower=0.001, upper=0.999)
                        prob.model.add_design_var("descent.hybridization", lower=0.01, upper=1.0)
                        prob.model.add_design_var("ac|propulsion|thermal|duct|area_nozzle", lower=1.0, upper=200.0)
                        prob.model.add_design_var("ac|propulsion|thermal|hx|n_wide_cold", lower=100.0, upper=1500.0)
                        prob.model.add_constraint("margins.MTOW_margin", lower=0.0)
                        prob.model.add_design_var("ac|propulsion|thermal|hx|coolant_mass", lower=5.0, upper=15.0)

                        # prob.model.add_constraint('design_mission.residuals.fuel_capacity_margin',lower=0.0)

                        prob.model.add_constraint("rotate.range_final", upper=1357)
                        prob.model.add_constraint("v0v1.Vstall_eas", upper=42.0)
                        prob.model.add_constraint("descent.propmodel.batt1.SOC_final", lower=0.0)
                        prob.model.add_constraint("climb.throttle", upper=1.05 * np.ones(num_nodes))
                        prob.model.add_constraint(
                            "climb.propmodel.eng1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "climb.propmodel.gen1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "climb.propmodel.batt1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "cruise.propmodel.eng1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "cruise.propmodel.gen1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "cruise.propmodel.batt1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "descent.propmodel.eng1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "descent.propmodel.gen1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "descent.propmodel.batt1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "v0v1.propmodel.batt1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint("engineoutclimb.gamma", lower=0.02)
                        prob.model.add_constraint("climb.propmodel.motorheatsink.T", upper=363.0 * np.ones(num_nodes))
                        prob.model.add_constraint("cruise.propmodel.motorheatsink.T", upper=363.0 * np.ones(num_nodes))
                        prob.model.add_constraint("climb.propmodel.batteryheatsink.T", upper=323.0 * np.ones(num_nodes))
                        prob.model.add_constraint(
                            "cruise.propmodel.batteryheatsink.T", upper=323.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint("climb.hxadder.area_constraint", lower=0.001)
                        prob.model.add_objective("mixed_objective")  # TODO add this objective

                    elif run_type == "comp_sizing":
                        print("======Performing Component Sizing Optimization===========")
                        prob.model.add_design_var("ac|propulsion|engine|rating", lower=1, upper=3000)
                        prob.model.add_design_var("ac|propulsion|motor|rating", lower=1, upper=3000)
                        prob.model.add_design_var("ac|propulsion|generator|rating", lower=1, upper=3000)
                        prob.model.add_design_var("ac|weights|W_battery", lower=20, upper=2250)
                        prob.model.add_design_var("cruise.hybridization", lower=0.01, upper=0.5)

                        prob.model.add_constraint("margins.MTOW_margin", equals=0.0)  # TODO implement
                        prob.model.add_constraint("rotate.range_final", upper=1357)  # TODO check units
                        prob.model.add_constraint("descent.propmodel.batt1.SOC_final", lower=0.0)
                        prob.model.add_constraint(
                            "v0v1.propmodel.eng1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "v0v1.propmodel.gen1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "v0v1.propmodel.batt1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "climb.propmodel.eng1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "climb.propmodel.gen1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint(
                            "climb.propmodel.batt1.component_sizing_margin", upper=1.0 * np.ones(num_nodes)
                        )
                        prob.model.add_constraint("climb.throttle", upper=1.05 * np.ones(num_nodes))
                        prob.model.add_objective("fuel_burn")

                    else:
                        print("======Analyzing Fuel Burn for Given Mision============")
                        prob.model.add_design_var("cruise.hybridization", lower=0.01, upper=0.5)
                        prob.model.add_constraint("descent.propmodel.batt1.SOC_final", lower=0.0)
                        prob.model.add_objective("descent.fuel_used_final")

                    prob.driver = ScipyOptimizeDriver()
                    if write_logs:
                        filename_to_save = "case_" + str(spec_energy) + "_" + str(design_range) + ".sql"
                        if os.path.isfile(filename_to_save):
                            if design_range != 300:
                                last_successful_opt = filename_to_save
                            else:
                                last_successful_opt = "case_" + str(spec_energy + 50) + "_" + str(700) + ".sql"
                            print("Skipping " + filename_to_save)
                            continue
                        recorder = SqliteRecorder(filename_to_save)
                        prob.driver.add_recorder(recorder)
                        prob.driver.recording_options["includes"] = []
                        prob.driver.recording_options["record_objectives"] = True
                        prob.driver.recording_options["record_constraints"] = True
                        prob.driver.recording_options["record_desvars"] = True

                    prob.setup(check=False)
                    set_values(prob, num_nodes, design_range, spec_energy)

                    if last_successful_opt is not None:
                        cr = CaseReader(last_successful_opt)
                        driver_cases = cr.list_cases("driver")
                        case = cr.get_case(driver_cases[-1])
                        design_vars = case.get_design_vars()
                        for key in design_vars.keys():
                            prob.set_val(key, design_vars[key])

                    run_flag = prob.run_driver()
                    if run_flag:
                        raise ValueError("Opt failed")
                    else:
                        if write_logs:
                            last_successful_opt = filename_to_save
                        prob.cleanup()
                except BaseException as e:
                    if write_logs:
                        logging.error("Optimization " + filename_to_save + " failed because " + repr(e))
                    prob.cleanup()
                    try:
                        if write_logs:
                            os.rename(filename_to_save, filename_to_save.split(".sql")[0] + "_failed.sql")
                    except WindowsError as we:
                        if write_logs:
                            logging.error("Error renaming file: " + repr(we))
                            os.remove(filename_to_save)

        show_outputs(prob)
