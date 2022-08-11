import numpy as np

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, IndepVarComp, NewtonSolver, BoundsEnforceLS
from openconcept.utilities import DictIndepVarComp, plot_trajectory, LinearInterpolator

# imports for the airplane model itself
from openconcept.aerodynamics import PolarDrag
from openconcept.propulsion import AllElectricSinglePropulsionSystemWithThermal_Incompressible
from openconcept.examples.aircraft_data.TBM850 import data as acdata
from openconcept.mission import FullMissionAnalysis


class ElectricTBM850Model(Group):
    """
    A custom model specific to an electrified TBM 850 airplane
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
        controls.add_output("prop1rpm", val=np.ones((nn,)) * 2000, units="rpm")

        propulsion_promotes_outputs = ["thrust"]
        propulsion_promotes_inputs = ["fltcond|*", "ac|propulsion|*", "throttle", "ac|weights|*", "duration"]

        self.add_subsystem(
            "propmodel",
            AllElectricSinglePropulsionSystemWithThermal_Incompressible(num_nodes=nn),
            promotes_inputs=propulsion_promotes_inputs,
            promotes_outputs=propulsion_promotes_outputs,
        )
        self.connect("prop1rpm", "propmodel.prop1.rpm")

        # use a different drag coefficient for takeoff versus cruise
        if flight_phase not in ["v0v1", "v1v0", "v1vr", "rotate"]:
            cd0_source = "ac|aero|polar|CD0_cruise"
        else:
            cd0_source = "ac|aero|polar|CD0_TO"
        self.add_subsystem(
            "drag",
            PolarDrag(num_nodes=nn),
            promotes_inputs=["fltcond|CL", "ac|geom|*", ("CD0", cd0_source), "fltcond|q", ("e", "ac|aero|polar|e")],
            promotes_outputs=["drag"],
        )
        self.add_subsystem(
            "weight",
            LinearInterpolator(num_nodes=nn, units="kg"),
            promotes_inputs=[("start_val", "ac|weights|MTOW"), ("end_val", "ac|weights|MTOW")],
            promotes_outputs=[("vec", "weight")],
        )


class ElectricTBMAnalysisGroup(Group):
    """This is an example of a balanced field takeoff and three-phase mission analysis."""

    def setup(self):
        nn = 11

        dv_comp = self.add_subsystem("dv_comp", DictIndepVarComp(acdata), promotes_outputs=["*"])
        # eventually replace the following aerodynamic parameters with an analysis module (maybe OpenAeroStruct)
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

        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output_from_dict("ac|propulsion|propeller|diameter")

        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")
        dv_comp.add_output("ac|propulsion|motor|rating", val=850, units="hp")
        dv_comp.add_output("ac|weights|W_battery", val=2000, units="lb")

        mission_data_comp = self.add_subsystem("mission_data_comp", IndepVarComp(), promotes_outputs=["*"])
        # mission_data_comp.add_output('cruise|h0',val=6000, units='m')
        # mission_data_comp.add_output('design_range',val=150,units='NM')
        mission_data_comp.add_output("T_motor_initial", val=15, units="degC")
        mission_data_comp.add_output("T_res_initial", val=15.1, units="degC")

        self.add_subsystem(
            "analysis",
            FullMissionAnalysis(num_nodes=nn, aircraft_model=ElectricTBM850Model, transition_method="ode"),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.connect("T_motor_initial", "v0v1.propmodel.motorheatsink.T_initial")
        self.connect("T_res_initial", "v0v1.propmodel.reservoir.T_initial")


def configure_problem():
    prob = Problem()
    prob.model = ElectricTBMAnalysisGroup()

    prob.model.nonlinear_solver = NewtonSolver(iprint=2)
    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options["solve_subsystems"] = True
    prob.model.nonlinear_solver.options["maxiter"] = 20
    prob.model.nonlinear_solver.options["atol"] = 1e-8
    prob.model.nonlinear_solver.options["rtol"] = 1e-8
    prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement="scalar", print_bound_enforce=False)
    prob.model.add_design_var("mission_range", lower=100, upper=300, scaler=1e-2)
    prob.model.add_constraint("descent.propmodel.batt1.SOC_final", lower=0.0)
    prob.model.add_objective("mission_range", scaler=-1.0)
    prob.driver = ScipyOptimizeDriver()
    return prob


def set_values(prob, num_nodes):
    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val("rotate.fltcond|Utrue", np.ones((num_nodes)) * 80, units="kn")
    prob.set_val("rotate.accel_vert", np.ones((num_nodes)) * 0.1, units="m/s**2")
    prob.set_val("climb.fltcond|vs", np.ones((num_nodes,)) * 1000, units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.ones((num_nodes,)) * 140, units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 0.01, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.ones((num_nodes,)) * 140, units="kn")
    prob.set_val("descent.fltcond|vs", np.ones((num_nodes,)) * (-600), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.ones((num_nodes,)) * 140, units="kn")

    prob.set_val("cruise|h0", 6000, units="m")
    prob.set_val("mission_range", 150, units="NM")

    # set some (optional) guesses for takeoff speeds and (required) mission parameters
    prob.set_val("v0v1.fltcond|Utrue", np.ones((num_nodes)) * 50, units="kn")
    prob.set_val("v1vr.fltcond|Utrue", np.ones((num_nodes)) * 85, units="kn")
    prob.set_val("v1v0.fltcond|Utrue", np.ones((num_nodes)) * 85, units="kn")


def show_outputs(prob):
    # print some outputs

    vars_list = ["ac|weights|MTOW", "descent.propmodel.batt1.SOC_final", "rotate.range_final"]
    units = ["lb", None, "ft"]
    nice_print_names = ["MTOW", "Final battery state of charge", "TOFL (over 35ft obstacle)"]
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i] + ": " + str(prob.get_val(thing, units=units[i])[0]) + " " + str(units[i]))

    # plot some stuff
    plots = True
    if plots:
        x_var = "range"
        x_unit = "ft"
        y_vars = [
            "fltcond|h",
            "fltcond|Ueas",
            "throttle",
            "fltcond|vs",
            "propmodel.batt1.SOC",
            "propmodel.motorheatsink.T",
            "propmodel.reservoir.T_out",
            "propmodel.duct.mdot",
        ]
        y_units = ["ft", "kn", None, "ft/min", None, "degC", "degC", "lb/s"]
        x_label = "Distance (ft)"
        y_labels = [
            "Altitude (ft)",
            "Veas airspeed (knots)",
            "Throttle",
            "Vertical speed (ft/min)",
            "Battery SOC",
            "Motor temp (C)",
            "Reservoir outlet temp (C)",
            "Cooling duct mass flow (lb/s)",
        ]
        phases = ["v0v1", "v1vr", "rotate", "v1v0"]
        plot_trajectory(
            prob,
            x_var,
            x_unit,
            y_vars,
            y_units,
            phases,
            x_label=x_label,
            y_labels=y_labels,
            plot_title="Elec Single Takeoff",
        )

        x_var = "range"
        x_unit = "NM"
        x_label = "Range (nmi)"
        phases = ["climb", "cruise", "descent"]
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
            plot_title="Elec Single Mission Profile",
        )


def run_electricsingle_analysis(plots=False):
    num_nodes = 11
    prob = configure_problem()
    prob.setup(check=True, mode="fwd")
    set_values(prob, num_nodes)
    prob.run_model()
    if plots:
        show_outputs(prob)
    return prob


if __name__ == "__main__":
    run_electricsingle_analysis(plots=True)
