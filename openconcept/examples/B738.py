import numpy as np

import openmdao.api as om
from openconcept.utilities import AddSubtractComp, DictIndepVarComp, plot_trajectory

# imports for the airplane model itself
from openconcept.aerodynamics import PolarDrag
from openconcept.examples.aircraft_data.B738 import data as acdata
from openconcept.mission import MissionWithReserve, IntegratorGroup
from openconcept.propulsion import CFM56


class B738AirplaneModel(IntegratorGroup):
    """
    A custom model specific to the Boeing 737-800 airplane.
    This class will be passed in to the mission analysis code.

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]

        # a propulsion system needs to be defined in order to provide thrust
        # information for the mission analysis code
        propulsion_promotes_inputs = ["fltcond|*", "throttle"]

        self.add_subsystem("propmodel", CFM56(num_nodes=nn, plot=False), promotes_inputs=propulsion_promotes_inputs)

        doubler = om.ExecComp(
            ["thrust=2*thrust_in", "fuel_flow=2*fuel_flow_in"],
            thrust_in={"val": 1.0 * np.ones((nn,)), "units": "kN"},
            thrust={"val": 1.0 * np.ones((nn,)), "units": "kN"},
            fuel_flow={
                "val": 1.0 * np.ones((nn,)),
                "units": "kg/s",
                "tags": ["integrate", "state_name:fuel_used", "state_units:kg", "state_val:1.0", "state_promotes:True"],
            },
            fuel_flow_in={"val": 1.0 * np.ones((nn,)), "units": "kg/s"},
        )

        self.add_subsystem("doubler", doubler, promotes_outputs=["*"])
        self.connect("propmodel.thrust", "doubler.thrust_in")
        self.connect("propmodel.fuel_flow", "doubler.fuel_flow_in")

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

        # generally the weights module will be custom to each airplane
        passthru = om.ExecComp("OEW=x", x={"val": 1.0, "units": "kg"}, OEW={"val": 1.0, "units": "kg"})
        self.add_subsystem("OEW", passthru, promotes_inputs=[("x", "ac|weights|OEW")], promotes_outputs=["OEW"])

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


class B738AnalysisGroup(om.Group):
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

        dv_comp.add_output_from_dict("ac|geom|nosegear|length")
        dv_comp.add_output_from_dict("ac|geom|maingear|length")

        dv_comp.add_output_from_dict("ac|weights|MTOW")
        dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
        dv_comp.add_output_from_dict("ac|weights|MLW")
        dv_comp.add_output_from_dict("ac|weights|OEW")

        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")

        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")

        # Run a full mission analysis including takeoff, reserve_, cruise,reserve_ and descereserve_nt
        self.add_subsystem(
            "analysis",
            MissionWithReserve(num_nodes=nn, aircraft_model=B738AirplaneModel),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )


def configure_problem():
    prob = om.Problem()
    prob.model = B738AnalysisGroup()
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options["maxiter"] = 20
    prob.model.nonlinear_solver.options["atol"] = 1e-6
    prob.model.nonlinear_solver.options["rtol"] = 1e-6
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar", print_bound_enforce=False)
    return prob


def set_values(prob, num_nodes):
    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val("climb.fltcond|vs", np.linspace(2300.0, 600.0, num_nodes), units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.linspace(230, 220, num_nodes), units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 4.0, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.linspace(265, 258, num_nodes), units="kn")
    prob.set_val("descent.fltcond|vs", np.linspace(-1000, -150, num_nodes), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.ones((num_nodes,)) * 250, units="kn")
    prob.set_val("reserve_climb.fltcond|vs", np.linspace(3000.0, 2300.0, num_nodes), units="ft/min")
    prob.set_val("reserve_climb.fltcond|Ueas", np.linspace(230, 230, num_nodes), units="kn")
    prob.set_val("reserve_cruise.fltcond|vs", np.ones((num_nodes,)) * 4.0, units="ft/min")
    prob.set_val("reserve_cruise.fltcond|Ueas", np.linspace(250, 250, num_nodes), units="kn")
    prob.set_val("reserve_descent.fltcond|vs", np.linspace(-800, -800, num_nodes), units="ft/min")
    prob.set_val("reserve_descent.fltcond|Ueas", np.ones((num_nodes,)) * 250, units="kn")
    prob.set_val("loiter.fltcond|vs", np.linspace(0.0, 0.0, num_nodes), units="ft/min")
    prob.set_val("loiter.fltcond|Ueas", np.ones((num_nodes,)) * 200, units="kn")
    prob.set_val("cruise|h0", 33000.0, units="ft")
    prob.set_val("reserve|h0", 15000.0, units="ft")
    prob.set_val("mission_range", 2050, units="NM")


def show_outputs(prob):
    # print some outputs
    vars_list = ["descent.fuel_used_final", "loiter.fuel_used_final"]
    units = ["lb", "lb"]
    nice_print_names = ["Block fuel", "Total fuel"]
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i] + ": " + str(prob.get_val(thing, units=units[i])[0]) + " " + units[i])

    # plot some stuff
    plots = True
    if plots:
        x_var = "range"
        x_unit = "NM"
        y_vars = ["fltcond|h", "fltcond|Ueas", "fuel_used", "throttle", "fltcond|vs", "fltcond|M", "fltcond|CL"]
        y_units = ["ft", "kn", "lbm", None, "ft/min", None, None]
        x_label = "Range (nmi)"
        y_labels = [
            "Altitude (ft)",
            "Veas airspeed (knots)",
            "Fuel used (lb)",
            "Throttle setting",
            "Vertical speed (ft/min)",
            "Mach number",
            "CL",
        ]
        phases = ["climb", "cruise", "descent", "reserve_climb", "reserve_cruise", "reserve_descent", "loiter"]
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
            plot_title="737-800 Mission Profile",
        )


def run_738_analysis(plots=False):
    num_nodes = 11
    prob = configure_problem()
    prob.setup(check=True, mode="fwd")
    set_values(prob, num_nodes)
    prob.run_model()
    prob.model.list_outputs()
    if plots:
        show_outputs(prob)
    return prob


if __name__ == "__main__":
    run_738_analysis(plots=True)
