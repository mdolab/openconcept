import numpy as np

import openmdao.api as om
from openconcept.utilities import Integrator, AddSubtractComp, DictIndepVarComp

# imports for the airplane model itself
from openconcept.examples.aircraft_data.caravan import data as acdata
from openconcept.propulsion import TurbopropPropulsionSystem
from openconcept.weights import SingleTurboPropEmptyWeight

from openconcept.aerodynamics import PolarDrag
from openconcept.mission import FullMissionAnalysis


class CaravanAirplaneModel(om.Group):
    """
    A custom model specific to the Cessna Caravan airplane.
    This class will be passed in to the mission analysis code.

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]

        # any control variables other than throttle and braking need to be defined here
        controls = self.add_subsystem("controls", om.IndepVarComp(), promotes_outputs=["*"])
        controls.add_output("prop1rpm", val=np.ones((nn,)) * 2000, units="rpm")

        # a propulsion system needs to be defined in order to provide thrust
        # information for the mission analysis code
        propulsion_promotes_outputs = ["fuel_flow", "thrust"]
        propulsion_promotes_inputs = ["fltcond|*", "ac|propulsion|*", "throttle"]

        self.add_subsystem(
            "propmodel",
            TurbopropPropulsionSystem(num_nodes=nn),
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

        # generally the weights module will be custom to each airplane
        self.add_subsystem(
            "OEW",
            SingleTurboPropEmptyWeight(),
            promotes_inputs=["*", ("P_TO", "ac|propulsion|engine|rating")],
            promotes_outputs=["OEW"],
        )
        self.connect("propmodel.prop1.component_weight", "W_propeller")
        self.connect("propmodel.eng1.component_weight", "W_engine")

        # airplanes which consume fuel will need to integrate
        # fuel usage across the mission and subtract it from TOW
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


class CaravanAnalysisGroup(om.Group):
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

        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output_from_dict("ac|propulsion|propeller|diameter")

        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")

        self.add_subsystem(
            "analysis",
            FullMissionAnalysis(num_nodes=nn, aircraft_model=CaravanAirplaneModel),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )


def run_caravan_analysis():
    # Set up OpenMDAO to analyze the airplane
    num_nodes = 11
    prob = om.Problem()
    prob.model = CaravanAnalysisGroup()
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options["maxiter"] = 20
    prob.model.nonlinear_solver.options["atol"] = 1e-6
    prob.model.nonlinear_solver.options["rtol"] = 1e-6
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar", print_bound_enforce=False)
    prob.setup(check=True, mode="fwd")

    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val("climb.fltcond|vs", np.ones((num_nodes,)) * 850, units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.ones((num_nodes,)) * 104, units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 0.01, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.ones((num_nodes,)) * 129, units="kn")
    prob.set_val("descent.fltcond|vs", np.ones((num_nodes,)) * (-400), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.ones((num_nodes,)) * 100, units="kn")

    prob.set_val("cruise|h0", 18000.0, units="ft")
    prob.set_val("mission_range", 250, units="NM")

    # (optional) guesses for takeoff speeds may help with convergence
    prob.set_val("v0v1.fltcond|Utrue", np.ones((num_nodes)) * 40, units="kn")
    prob.set_val("v1vr.fltcond|Utrue", np.ones((num_nodes)) * 70, units="kn")
    prob.set_val("v1v0.fltcond|Utrue", np.ones((num_nodes)) * 60, units="kn")
    prob.set_val("rotate.fltcond|Utrue", np.ones((num_nodes)) * 80, units="kn")

    # set some airplane-specific values.
    prob["climb.OEW.structural_fudge"] = 1.67
    prob["v0v1.throttle"] = np.ones((num_nodes))
    prob["v1vr.throttle"] = np.ones((num_nodes))
    prob["rotate.throttle"] = np.ones((num_nodes))

    prob.run_model()
    return prob


if __name__ == "__main__":
    from openconcept.utilities.visualization import plot_trajectory

    # run the analysis
    prob = run_caravan_analysis()

    # print some outputs
    vars_list = ["ac|weights|MTOW", "climb.OEW", "descent.fuel_used_final", "v1vr.range_final"]
    units = ["lb", "lb", "lb", "ft"]
    nice_print_names = ["MTOW", "OEW", "Fuel used", "TOFL ground roll"]
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i] + ": " + str(prob.get_val(thing, units=units[i])[0]) + " " + units[i])

    # plot some stuff
    plots = True
    if plots:
        x_var = "range"
        x_unit = "ft"
        y_vars = ["fltcond|Ueas", "fltcond|h"]
        y_units = ["kn", "ft"]
        x_label = "Distance (ft)"
        y_labels = ["Veas airspeed (knots)", "Altitude (ft)"]
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
            plot_title="Caravan Takeoff",
        )

        x_var = "range"
        x_unit = "NM"
        y_vars = ["fltcond|h", "fltcond|Ueas", "fuel_used", "throttle", "fltcond|vs"]
        y_units = ["ft", "kn", "lbm", None, "ft/min"]
        x_label = "Range (nmi)"
        y_labels = [
            "Altitude (ft)",
            "Veas airspeed (knots)",
            "Fuel used (lb)",
            "Throttle setting",
            "Vertical speed (ft/min)",
        ]
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
            plot_title="Caravan Mission Profile",
        )
