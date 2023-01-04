"""
This work was the basis of the following paper.
Please cite it if you use this for your own publication!

@InProceedings{Adler2022a,
    author      = {Eytan J. Adler and Joaquim R. R. A. Martins},
    title       = {Aerostructural wing design optimization considering full mission analysis},
    booktitle   = {AIAA SciTech Forum},
    doi         = {10.2514/6.2022-0382},
    month       = {January},
    year        = {2022}
}

Eytan Adler (Jan 2022)
"""

import numpy as np

import openmdao.api as om
from openconcept.utilities import AddSubtractComp, DictIndepVarComp, plot_trajectory

# imports for the airplane model itself
from openconcept.mission import IntegratorGroup, BasicMission
from openconcept.aerodynamics import AerostructDragPolar
from openconcept.aerodynamics.openaerostruct import Aerostruct, AerostructDragPolarExact
from openconcept.examples.aircraft_data.B738 import data as acdata
from openconcept.propulsion import CFM56
from openconcept.aerodynamics import Lift
from openconcept.atmospherics import DynamicPressureComp

NUM_X = 5
NUM_Y = 15
NUM_TWIST = 3
NUM_TOVERC = 3
NUM_SKIN = 3
NUM_SPAR = 3
USE_SURROGATE = True


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

        oas_surf_dict = {}  # options for OpenAeroStruct
        # Grid size and number of spline control points (must be same as B738AnalysisGroup)
        global NUM_X, NUM_Y, NUM_TWIST, NUM_TOVERC, NUM_SKIN, NUM_SPAR, USE_SURROGATE
        if USE_SURROGATE:
            self.add_subsystem(
                "drag",
                AerostructDragPolar(
                    num_nodes=nn,
                    num_x=NUM_X,
                    num_y=NUM_Y,
                    num_twist=NUM_TWIST,
                    num_toverc=NUM_TOVERC,
                    num_skin=NUM_SKIN,
                    num_spar=NUM_SPAR,
                    surf_options=oas_surf_dict,
                ),
                promotes_inputs=[
                    "fltcond|CL",
                    "fltcond|M",
                    "fltcond|h",
                    "fltcond|q",
                    "ac|geom|wing|S_ref",
                    "ac|geom|wing|AR",
                    "ac|geom|wing|taper",
                    "ac|geom|wing|c4sweep",
                    "ac|geom|wing|twist",
                    "ac|geom|wing|toverc",
                    "ac|geom|wing|skin_thickness",
                    "ac|geom|wing|spar_thickness",
                    "ac|aero|CD_nonwing",
                ],
                promotes_outputs=["drag", "ac|weights|W_wing", ("failure", "ac|struct|failure")],
            )
        else:
            self.add_subsystem(
                "drag",
                AerostructDragPolarExact(
                    num_nodes=nn,
                    num_x=NUM_X,
                    num_y=NUM_Y,
                    num_twist=NUM_TWIST,
                    num_toverc=NUM_TOVERC,
                    num_skin=NUM_SKIN,
                    num_spar=NUM_SPAR,
                    surf_options=oas_surf_dict,
                ),
                promotes_inputs=[
                    "fltcond|CL",
                    "fltcond|M",
                    "fltcond|h",
                    "fltcond|q",
                    "ac|geom|wing|S_ref",
                    "ac|geom|wing|AR",
                    "ac|geom|wing|taper",
                    "ac|geom|wing|c4sweep",
                    "ac|geom|wing|twist",
                    "ac|geom|wing|toverc",
                    "ac|geom|wing|skin_thickness",
                    "ac|geom|wing|spar_thickness",
                    "ac|aero|CD_nonwing",
                ],
                promotes_outputs=["drag", "ac|weights|W_wing", ("failure", "ac|struct|failure")],
            )

        # generally the weights module will be custom to each airplane
        passthru = om.ExecComp("OEW=x", x={"val": 1.0, "units": "kg"}, OEW={"val": 1.0, "units": "kg"})
        self.add_subsystem("OEW", passthru, promotes_inputs=[("x", "ac|weights|OEW")], promotes_outputs=["OEW"])

        # Use Raymer as estimate for 737 original wing weight, subtract it
        # out, then add in OpenAeroStruct wing weight estimate
        self.add_subsystem(
            "weight",
            AddSubtractComp(
                output_name="weight",
                input_names=["ac|weights|MTOW", "fuel_used", "ac|weights|orig_W_wing", "ac|weights|W_wing"],
                units="kg",
                vec_size=[1, nn, 1, 1],
                scaling_factors=[1, -1, -1, 1],
            ),
            promotes_inputs=["*"],
            promotes_outputs=["weight"],
        )


class B738AnalysisGroup(om.Group):
    def initialize(self):
        self.options.declare("num_nodes", default=11, desc="Number of analysis points per flight segment")
        self.options.declare("num_x", default=3, desc="Aerostructural chordwise nodes")
        self.options.declare("num_y", default=7, desc="Aerostructural halfspan nodes")
        self.options.declare("num_twist", default=3, desc="Number of twist control points")
        self.options.declare("num_toverc", default=3, desc="Number of t/c control points")
        self.options.declare("num_skin", default=3, desc="Number of skin control points")
        self.options.declare("num_spar", default=3, desc="Number of spar control points")
        self.options.declare(
            "use_surrogate",
            default=True,
            desc="Use surrogate for aerostructural drag " + "polar instead of OpenAeroStruct directly",
        )

    def setup(self):
        # Define number of analysis points to run pers mission segment
        nn = self.options["num_nodes"]

        global NUM_X, NUM_Y, NUM_TWIST, NUM_TOVERC, NUM_SKIN, NUM_SPAR, USE_SURROGATE
        NUM_X = self.options["num_x"]
        NUM_Y = self.options["num_y"]
        NUM_TWIST = self.options["num_twist"]
        NUM_TOVERC = self.options["num_toverc"]
        NUM_SKIN = self.options["num_skin"]
        NUM_SPAR = self.options["num_spar"]
        USE_SURROGATE = self.options["use_surrogate"]

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
        # dv_comp.add_output_from_dict('ac|geom|wing|toverc')
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

        # Aerostructural design parameters
        twist = np.linspace(-2, 2, NUM_TWIST)
        toverc = acdata["ac"]["geom"]["wing"]["toverc"]["value"] * np.ones(NUM_TOVERC)
        t_skin = np.linspace(0.005, 0.015, NUM_SKIN)
        t_spar = np.linspace(0.005, 0.01, NUM_SPAR)
        self.set_input_defaults("ac|geom|wing|twist", twist, units="deg")
        self.set_input_defaults("ac|geom|wing|toverc", toverc)
        self.set_input_defaults("ac|geom|wing|skin_thickness", t_skin, units="m")
        self.set_input_defaults("ac|geom|wing|spar_thickness", t_spar, units="m")
        self.set_input_defaults("ac|aero|CD_nonwing", 0.0145)  # based on matching fuel burn of B738.py example

        # Compute Raymer wing weight to know what to subtract from the MTOW before adding the OpenAeroStruct weight
        W_dg = 174.2e3  # design gross weight, lbs
        N_z = 1.5 * 3.0  # ultimate load factor (1.5 x limit load factor of 3g)
        S_w = 1368.0  # trapezoidal wing area, ft^2 (from photogrammetry)
        A = 9.44  # aspect ratio
        t_c = 0.12  # root thickness to chord ratio
        taper = 0.159  # taper ratio
        sweep = 25.0  # wing sweep at 25% MAC
        S_csw = 196.8  # wing-mounted control surface area, ft^2 (from photogrammetry)
        W_wing_raymer = (
            0.0051
            * (W_dg * N_z) ** 0.557
            * S_w**0.649
            * A**0.5
            * (t_c) ** (-0.4)
            * (1 + taper) ** 0.1
            / np.cos(np.deg2rad(sweep))
            * S_csw**0.1
        )
        self.set_input_defaults("ac|weights|orig_W_wing", W_wing_raymer, units="lb")

        # ======================== Mission analysis ========================
        # Run a full mission analysis including takeoff, reserve_, cruise,reserve_ and descereserve_nt
        self.add_subsystem(
            "analysis",
            BasicMission(num_nodes=nn, aircraft_model=B738AirplaneModel),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        # ======================== Aerostructural sizing at 2.5g ========================
        # Add single point aerostructural analysis at 2.5g and MTOW to size the wingbox structure
        self.add_subsystem(
            "aerostructural_maneuver",
            Aerostruct(
                num_x=NUM_X,
                num_y=NUM_Y,
                num_twist=NUM_TWIST,
                num_toverc=NUM_TOVERC,
                num_skin=NUM_SKIN,
                num_spar=NUM_SPAR,
            ),
            promotes_inputs=[
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                "ac|geom|wing|taper",
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                "ac|geom|wing|skin_thickness",
                "ac|geom|wing|spar_thickness",
                "ac|geom|wing|twist",
                "load_factor",
            ],
            promotes_outputs=[("failure", "2_5g_KS_failure")],
        )

        # Flight condition of 2.5g maneuver load case
        self.set_input_defaults("aerostructural_maneuver.fltcond|M", 0.8)
        self.set_input_defaults("aerostructural_maneuver.fltcond|h", 20e3, units="ft")
        self.set_input_defaults("load_factor", 2.5)  # multiplier on weights in structural problem

        # Find angle of attack for 2.5g sizing flight condition such that lift = 2.5 * MTOW
        self.add_subsystem("dyn_pressure", DynamicPressureComp(num_nodes=1))
        self.add_subsystem("lift", Lift(num_nodes=1), promotes_inputs=["ac|geom|wing|S_ref"])
        self.add_subsystem(
            "kg_to_N",
            om.ExecComp(
                "lift = load_factor * (MTOW - orig_W_wing + W_wing) * a",
                lift={"units": "N"},
                MTOW={"units": "kg"},
                orig_W_wing={"units": "kg", "val": W_wing_raymer / 2.20462},
                W_wing={"units": "kg"},
                a={"units": "m/s**2", "val": 9.807},
            ),
            promotes_inputs=["load_factor", ("MTOW", "ac|weights|MTOW")],
        )
        self.add_subsystem(
            "struct_sizing_AoA",
            om.BalanceComp("alpha", eq_units="N", lhs_name="MTOW", rhs_name="lift", units="deg", val=10.0, lower=0.0),
        )
        self.connect("climb.ac|weights|W_wing", "kg_to_N.W_wing")
        self.connect("kg_to_N.lift", "struct_sizing_AoA.MTOW")
        self.connect("aerostructural_maneuver.density.fltcond|rho", "dyn_pressure.fltcond|rho")
        self.connect("aerostructural_maneuver.airspeed.Utrue", "dyn_pressure.fltcond|Utrue")
        self.connect("dyn_pressure.fltcond|q", "lift.fltcond|q")
        self.connect("aerostructural_maneuver.fltcond|CL", "lift.fltcond|CL")
        self.connect("lift.lift", "struct_sizing_AoA.lift")
        self.connect("struct_sizing_AoA.alpha", "aerostructural_maneuver.fltcond|alpha")


def configure_problem(num_nodes):
    prob = om.Problem()
    prob.model.add_subsystem("analysis", B738AnalysisGroup(num_nodes=num_nodes), promotes=["*"])
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options["maxiter"] = 10
    prob.model.nonlinear_solver.options["atol"] = 1e-6
    prob.model.nonlinear_solver.options["rtol"] = 1e-6
    prob.model.nonlinear_solver.options["err_on_non_converge"] = True
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar", print_bound_enforce=True)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.opt_settings["tol"] = 1e-5
    prob.driver.options["debug_print"] = ["objs", "desvars", "nl_cons"]

    # =========================== Mission design variables/constraints ===========================
    prob.model.add_objective("descent.fuel_used_final", scaler=1e-4)  # minimize block fuel burn
    prob.model.add_constraint("climb.throttle", lower=0.01, upper=1.05)
    prob.model.add_constraint("cruise.throttle", lower=0.01, upper=1.05)
    prob.model.add_constraint("descent.throttle", lower=0.01, upper=1.05)

    # =========================== Aerostructural wing design variables/constraints ===========================
    # Find twist distribution that minimizes fuel burn; lock the twist tip in place
    # to prevent rigid rotation of the whole wing
    prob.model.add_design_var(
        "ac|geom|wing|twist", lower=np.array([0, -10, -10]), upper=np.array([0, 10, 10]), units="deg"
    )
    prob.model.add_design_var("ac|geom|wing|AR", lower=5.0, upper=10.4)  # limit to fit in group III gate
    prob.model.add_design_var("ac|geom|wing|c4sweep", lower=0.0, upper=35.0)
    prob.model.add_design_var("ac|geom|wing|toverc", lower=np.linspace(0.03, 0.1, NUM_TOVERC), upper=0.25)
    prob.model.add_design_var("ac|geom|wing|spar_thickness", lower=0.003, upper=0.1, scaler=1e2, units="m")
    prob.model.add_design_var("ac|geom|wing|skin_thickness", lower=0.003, upper=0.1, scaler=1e2, units="m")
    prob.model.add_design_var("ac|geom|wing|taper", lower=0.01, upper=0.35, scaler=1e1)
    prob.model.add_constraint("2_5g_KS_failure", upper=0.0)

    return prob


def set_values(prob, num_nodes, mission_range=2050):
    # set some (required) mission parameters. Each phase needs a vertical and air-speed
    # the entire mission needs a cruise altitude and mission range
    prob.set_val("cruise|h0", 35000.0, units="ft")
    prob.set_val("mission_range", mission_range, units="NM")
    prob.set_val("climb.fltcond|vs", np.linspace(2000.0, 400.0, num_nodes), units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.linspace(220, 200, num_nodes), units="kn")
    prob.set_val("cruise.fltcond|vs", np.zeros((num_nodes,)), units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.linspace(250.279, 250.279, num_nodes), units="kn")  # M 0.78 @ 35k ft
    prob.set_val("descent.fltcond|vs", np.linspace(-2000, -1000, num_nodes), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.linspace(240, 250, num_nodes), units="kn")


def show_outputs(prob, plots=True):
    # print some outputs
    vars_list = ["descent.fuel_used_final"]
    units = ["lb", "lb"]
    nice_print_names = ["Block fuel", "Total fuel"]
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i] + ": " + str(prob.get_val(thing, units=units[i])[0]) + " " + units[i])

    # plot some stuff
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
            plot_title="737-800 Mission Profile",
        )


def run_738_analysis(plots=False):
    num_nodes = 11
    global NUM_X, NUM_Y
    NUM_X = 3
    NUM_Y = 7
    prob = configure_problem(num_nodes)
    prob.setup(check=False, mode="fwd")
    set_values(prob, num_nodes)
    prob.run_model()
    om.n2(prob, show_browser=False)
    show_outputs(prob, plots=plots)
    print(f"Wing weight = {prob.get_val('ac|weights|W_wing', units='lb')[0]} lb")
    print(f"Raymer wing weight = {prob.get_val('ac|weights|orig_W_wing', units='lb')[0]} lb")
    print(f"2.5g failure = {prob.get_val('2_5g_KS_failure')}")
    print(f"Climb failure = {prob.get_val('climb.ac|struct|failure')}")
    print(f"Cruise failure = {prob.get_val('cruise.ac|struct|failure')}")
    print(f"Descent failure = {prob.get_val('descent.ac|struct|failure')}")
    return prob


def run_738_optimization(plots=False):
    num_nodes = 11
    global NUM_X, NUM_Y
    NUM_X = 3
    NUM_Y = 7
    prob = configure_problem(num_nodes)
    prob.setup(check=True, mode="fwd")
    set_values(prob, num_nodes)
    prob.run_driver()
    prob.list_problem_vars(driver_scaling=False)
    print(f"Wing weight = {prob.get_val('ac|weights|W_wing', units='lb')[0]} lb")
    print(f"Raymer wing weight = {prob.get_val('ac|weights|orig_W_wing', units='lb')[0]} lb")
    print(f"2.5g failure = {prob.get_val('2_5g_KS_failure')}")
    print(f"Climb failure = {prob.get_val('climb.ac|struct|failure')}")
    print(f"Cruise failure = {prob.get_val('cruise.ac|struct|failure')}")
    print(f"Descent failure = {prob.get_val('descent.ac|struct|failure')}")
    if plots:
        show_outputs(prob)
    return prob


if __name__ == "__main__":
    run_738_analysis(plots=False)
    # run_738_optimization(plots=True)
