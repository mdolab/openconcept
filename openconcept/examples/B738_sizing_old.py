# Modified by Cody Karcher, March 2023
# Original file written by Venkat Subramaniam
#    Avaliable here:  https://github.com/vrsub/openconcept/blob/sizing/examples/B738_sizing.py

# File does not run to completion, encounters an error.  OpenConcept has changed since original writing, bug unclear.

# ==========================================================================================================================================================================================================================================
# Standard Imports
# ==========================================================================================================================================================================================================================================
import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())
import openmdao.api as om
from openconcept.mission import IntegratorGroup
from openconcept.utilities import Integrator
from openconcept.utilities import AddSubtractComp, DictIndepVarComp, plot_trajectory

# imports for the airplane model itself
from openconcept.aerodynamics import PolarDrag
from openconcept.mission.profiles import (
    # FullMissionWithReserve,
    MissionWithReserve,
    FullMissionAnalysis,
    BasicMission,
)
from openconcept.propulsion import CFM56
from openconcept.aerodynamics import VLMDragPolar

# ==========================================================================================================================================================================================================================================
# Experimental stuff
# ==========================================================================================================================================================================================================================================
from openconcept.mission import FullMissionWithReserve

from openconcept.stability import (
    HStabSizing_JetTransport,
    VStabSizing_JetTransport,
)

from openconcept.geometry import (
    WingMAC_Trapezoidal,
    WingRoot_LinearTaper,
    WingSpan,
)

# new aerodynamics sub-folder, need to be clear on aerodynamic analyses
from openconcept.aerodynamics.sizing import Cd0_NonWing_JetTransport
from openconcept.aerodynamics.sizing import CL_MAX_cruise

from openconcept.weights import (
    JetTransportEmptyWeight
)

from openconcept.examples.aircraft_data.B738_sizing import data as acdata



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

        # doubles outputs from propulsion model to scale for twin engine aircraft
        doubler = om.ExecComp(
            [
                "thrust=thrust_in * (1 + propulsor_active) * (rating/27000)",
                "fuel_flow=fuel_flow_in * (1 + propulsor_active) * (rating/27000)",
            ],
            thrust_in={"val": 1.0 * np.ones((nn,)), "units": "kN"},
            thrust={"val": 1.0 * np.ones((nn,)), "units": "kN"},
            fuel_flow={
                "val": 1.0 * np.ones((nn,)),
                "units": "kg/s",
            },
            propulsor_active={"val": 1.0 * np.ones((nn,))},
            rating={"val": 27000, "units": "lbf"},
            fuel_flow_in={"val": 1.0 * np.ones((nn,)), "units": "kg/s"},
        )

        self.add_subsystem(
            "doubler",
            doubler,
            promotes_inputs=["propulsor_active", ("rating", "ac|propulsion|engine|rating")],
            promotes_outputs=["*"],
        )
        self.connect("propmodel.thrust", "doubler.thrust_in")
        self.connect("propmodel.fuel_flow", "doubler.fuel_flow_in")

        intfuel = self.add_subsystem(
            "intfuel",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        intfuel.add_integrand("fuel_used", rate_name="fuel_flow", val=1.0, units="kg")

        # use a different drag coefficient for takeoff versus cruise
        if flight_phase not in ["v0v1", "v1v0", "v1vr", "rotate"]:
            # self.set_input_defaults('ac|aero|polar|CD0_cruise', 0.0145)
            cd0_source = "ac|aero|polar|CD0_cruise"
        else:
            cd0_source = "ac|aero|polar|CD0_TO"

        oas_surf_dict = {}  # options for OpenAeroStruct
        oas_surf_dict["t_over_c"] = acdata["ac"]["geom"]["wing"]["toverc"]["value"]
        self.add_subsystem(
            "drag",
            VLMDragPolar(num_nodes=nn, num_x=4, num_y=10, num_twist=5, surf_options=oas_surf_dict), # OpenAeroStruct function call for VLM drag computation
            promotes_inputs=[
                "fltcond|CL",
                "fltcond|M",
                "fltcond|h",
                "fltcond|q",
                "ac|geom|*",
                ("ac|aero|CD_nonwing", cd0_source),
            ],
            promotes_outputs=["drag"],
        )
        self.set_input_defaults("ac|geom|wing|twist", np.zeros(5), units="deg")

        # compute aircraft weight over each flight segment
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

        # computes the difference between flight CL and CL_max to ensure aircraft does not exceed CL_max
        self.add_subsystem(
            "Cl_diff",
            AddSubtractComp(
                output_name="CL_diff",
                input_names=["ac|aero|CL_max", "fltcond|CL"],
                vec_size=[1, nn],
                scaling_factors=[1, -1],
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
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
        dv_comp.add_output_from_dict("ac|aero|Vstall_land")
        dv_comp.add_output_from_dict("ac|aero|Cl_max")

        dv_comp.add_output_from_dict("ac|geom|wing|S_ref")
        dv_comp.add_output_from_dict("ac|geom|wing|AR")
        dv_comp.add_output_from_dict("ac|geom|wing|c4sweep")
        dv_comp.add_output_from_dict("ac|geom|wing|taper")
        dv_comp.add_output_from_dict("ac|geom|wing|toverc")
        dv_comp.add_output_from_dict("ac|geom|hstab|c4_to_wing_c4")
        dv_comp.add_output_from_dict("ac|geom|hstab|AR")
        dv_comp.add_output_from_dict("ac|geom|hstab|c4sweep")
        dv_comp.add_output_from_dict("ac|geom|hstab|taper")
        dv_comp.add_output_from_dict("ac|geom|vstab|AR")
        dv_comp.add_output_from_dict("ac|geom|vstab|toverc")
        dv_comp.add_output_from_dict("ac|geom|vstab|c4sweep")

        dv_comp.add_output_from_dict("ac|geom|fuselage|S_wet")
        dv_comp.add_output_from_dict("ac|geom|fuselage|length")
        dv_comp.add_output_from_dict("ac|geom|fuselage|width")
        dv_comp.add_output_from_dict("ac|geom|fuselage|height")

        dv_comp.add_output_from_dict("ac|geom|nosegear|length")
        dv_comp.add_output_from_dict("ac|geom|nosegear|num_wheels")
        dv_comp.add_output_from_dict("ac|geom|maingear|length")
        dv_comp.add_output_from_dict("ac|geom|maingear|num_wheels")
        dv_comp.add_output_from_dict("ac|geom|maingear|num_shock_struts")

        dv_comp.add_output_from_dict("ac|weights|MLW")

        dv_comp.add_output_from_dict("ac|weights|max_payload")

        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output_from_dict("ac|propulsion|engine|BPR")
        dv_comp.add_output_from_dict("ac|propulsion|num_engines")

        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|num_flight_deck_crew")
        dv_comp.add_output_from_dict("ac|num_cabin_crew")
        dv_comp.add_output_from_dict("ac|cabin_pressure")
        dv_comp.add_output_from_dict("ac|q_cruise")

        # The follow subsystems compute all geometry parameters that are dependent on wing area and other defined geometry
        self.add_subsystem(
            "Wing_Root",
            WingRoot_LinearTaper(),
            promotes_inputs=["*"],
            promotes_outputs=[("C_root", "ac|geom|wing|root_chord")],
        )

        self.add_subsystem(
            "Wing_MAC", WingMAC_Trapezoidal(), promotes_inputs=["*"], promotes_outputs=[("MAC", "ac|geom|wing|MAC")]
        )

        self.add_subsystem(
            "HStab",
            HStabSizing_JetTransport(),
            promotes_inputs=["*"],
            promotes_outputs=[("hstab_area", "ac|geom|hstab|S_ref")],
        )

        self.add_subsystem(
            "VStab",
            VStabSizing_JetTransport(),
            promotes_inputs=["*"],
            promotes_outputs=[("vstab_area", "ac|geom|vstab|S_ref")],
        )

        self.add_subsystem(
            "wingspan", WingSpan(), promotes_inputs=["*"], promotes_outputs=[("span", "ac|geom|wing|span")]
        )

        self.add_subsystem(
            "cd0_estimation",
            Cd0_NonWing_JetTransport(),
            promotes_inputs=["*"],
            promotes_outputs=[("C_d0", "ac|aero|polar|CD0_cruise")],
        )

        # calls empty weight build up function to store operating empty weight
        self.add_subsystem(
            "OEW",
            JetTransportEmptyWeight(),
            promotes_inputs=["*"],
            promotes_outputs=[("OEW", "ac|weights|OEW")],
        )

        # computes aircraft MTOW in the top level model based on implicitly solved fuel burn
        self.add_subsystem(
            "MTOW",
            AddSubtractComp(
                output_name="ac|weights|MTOW",
                input_names=["ac|weights|OEW", "ac|weights|W_fuel_max", "ac|weights|max_payload"],
                units="kg",
                vec_size=[1, 1, 1],
                scaling_factors=[1, 1, 1],
                lower=0,
            ),
            promotes_outputs=["ac|weights|MTOW"],
            promotes_inputs=["*"],
        )

        # Run a full mission analysis including takeoff, reserve_, cruise,reserve_ and descereserve_n

        self.connect("climb.duration", "mission_duration.climb")
        self.connect("cruise.duration", "mission_duration.cruise")
        self.connect("descent.duration", "mission_duration.descent")

        self.add_subsystem(
            "CL_max", CL_MAX_cruise(), promotes_inputs=["*"], promotes_outputs=[("Wing_CL_max", "ac|aero|CL_max")]
        )

        # Runs mission analysis for aircraft model
        analysis = self.add_subsystem(
            "analysis",
            FullMissionWithReserve(num_nodes=nn, aircraft_model=B738AirplaneModel),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        # the following sections compute cash operating cost
        self.add_subsystem(
            "mission_duration",
            AddSubtractComp(
                output_name="total_block_time",
                input_names=["climb", "cruise", "descent"],
                units="s",
                vec_size=[1, 1, 1],
                scaling_factors=[1, 1, 1],
                lower=0,
            ),
            promotes_outputs=[("total_block_time", "block_time")],
        )

        # self.add_subsystem("COC", JetTransportCOC(), promotes_inputs=["*"], promotes_outputs=["*"])

        self.connect("loiter.fuel_used_final", "ac|weights|W_fuel_max")
        self.set_input_defaults(
            "ac|weights|MTOW", acdata["ac"]["weights"]["MTOW"]["value"], units=acdata["ac"]["weights"]["MTOW"]["units"]
        )
        self.set_input_defaults(
            "ac|weights|W_fuel_max",
            acdata["ac"]["weights"]["W_fuel_max"]["value"],
            units=acdata["ac"]["weights"]["W_fuel_max"]["units"],
        )


def configure_problem():
    prob = om.Problem()
    prob.model = B738AnalysisGroup()
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options["maxiter"] = 20
    prob.model.nonlinear_solver.options["atol"] = 1e-6
    prob.model.nonlinear_solver.options["rtol"] = 1e-6
    # prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement="scalar", print_bound_enforce=False, iprint=2)
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar", print_bound_enforce=False)

    # declare optimization driver and history files
    # prob.driver = om.pyOptSparseDriver(optimizer="IPOPT")
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    # prob.driver.opt_settings["limited_memory_max_history"] = 1000
    # prob.driver.opt_settings["tol"] = 1e-5
    # prob.driver.opt_settings["constr_viol_tol"] = 1e-9
    # prob.driver.hist_file = "B738_fullsizing_highCl_BFL_COC.hst"
    
    # define design variables
    prob.model.add_design_var("ac|geom|wing|S_ref", lower=100, upper=200, units="m**2")
    prob.model.add_design_var("ac|propulsion|engine|rating", lower=20000, upper=35000, units="lbf")
    prob.model.add_design_var("ac|geom|wing|AR", lower=5, upper=15)
    prob.model.add_design_var(
        "ac|geom|wing|twist", lower=np.array([0, -5, -5, -5, -5]), upper=np.array([0, 5, 5, 5, 5]), units="deg", ref=1
    )
    prob.model.add_design_var("ac|geom|wing|c4sweep", lower=-2, upper=50, units="deg", ref=1)
    prob.model.add_design_var("ac|geom|wing|taper", lower=0, upper=1, ref=1)
    
    # define objective functions
    # prob.model.add_objective("COC")
    prob.model.add_objective("descent.fuel_used_final")

    # define constraints for optimizer
    prob.model.add_constraint("ac|geom|wing|span", upper=36, units="m")
    prob.model.add_constraint("climb.throttle", upper=1.0)  # these constraints limit throttle
    prob.model.add_constraint("cruise.throttle", upper=1.0)
    prob.model.add_constraint("descent.throttle", upper=1.0)
    prob.model.add_constraint("reserve_climb.throttle", upper=1.0)  # these constraints limit throttle
    prob.model.add_constraint("reserve_cruise.throttle", upper=1.0)
    prob.model.add_constraint("reserve_descent.throttle", upper=1.0)
    prob.model.add_constraint("loiter.throttle", upper=1.0)

    prob.model.add_constraint("climb.CL_diff", lower=0)
    prob.model.add_constraint("cruise.CL_diff", lower=0)
    prob.model.add_constraint("descent.CL_diff", lower=0)
    prob.model.add_constraint("reserve_climb.CL_diff", lower=0)
    prob.model.add_constraint("reserve_cruise.CL_diff", lower=0)
    prob.model.add_constraint("reserve_descent.CL_diff", lower=0)
    prob.model.add_constraint("loiter.CL_diff", lower=0)

    prob.model.add_constraint("rotate.range_final", upper=6000)
    prob.model.add_constraint("v1v0.range_final", upper=6000)

    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons"]

    return prob


def set_values(prob, num_nodes):
    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val("climb.fltcond|vs", np.linspace(2300.0, 400.0, num_nodes), units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.linspace(230, 230, num_nodes), units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 4.0, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.linspace(265, 258, num_nodes), units="kn")
    prob.set_val("descent.fltcond|vs", np.linspace(-500, -150, num_nodes), units="ft/min")  # set this in optimizer
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

    prob.set_val("v0v1.fltcond|Utrue", np.ones((num_nodes)) * 50, units="kn")
    prob.set_val("v1vr.fltcond|Utrue", np.ones((num_nodes)) * 100, units="kn")
    prob.set_val("v1v0.fltcond|Utrue", np.ones((num_nodes)) * 100, units="kn")


def show_outputs(prob):
    # print some outputs
    vars_list = ["descent.fuel_used_final"]#, "COC"]
    units = ["lb", "USD"]
    nice_print_names = ["Block fuel"]#, "Mission Cash Operating Cost"]
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


def run_738_analysis(plots=True):
    num_nodes = 11
    prob = configure_problem()
    prob.setup(check=True, mode="fwd")
    set_values(prob, num_nodes)
    prob.run_model()
    # prob.run_driver()
    prob.model.list_outputs()
    om.n2(prob, outfile="B738_fullsizing_originaldesign.html")
    if plots:
        show_outputs(prob)
    return prob


if __name__ == "__main__":
    run_738_analysis(plots=True)