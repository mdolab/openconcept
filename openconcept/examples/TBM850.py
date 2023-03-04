# rst Imports (beg)
import numpy as np
import openmdao.api as om

# OpenConcept imports for the airplane model
from openconcept.propulsion import TurbopropPropulsionSystem
from openconcept.aerodynamics import PolarDrag
from openconcept.weights import SingleTurboPropEmptyWeight
from openconcept.mission import FullMissionAnalysis
from openconcept.examples.aircraft_data.TBM850 import data as acdata
from openconcept.utilities import AddSubtractComp, Integrator, DictIndepVarComp

# rst Imports (end)


# rst Aircraft (beg)
class TBM850AirplaneModel(om.Group):
    """
    A custom model specific to the TBM 850 airplane
    This class will be passed in to the mission analysis code.

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
        # A propulsion system needs to be defined in order to provide thrust information
        self.add_subsystem(
            "propmodel",
            TurbopropPropulsionSystem(num_nodes=nn),
            promotes_inputs=[
                "fltcond|rho",
                "fltcond|Utrue",
                "ac|propulsion|engine|rating",
                "ac|propulsion|propeller|diameter",
                "throttle",
            ],
            promotes_outputs=["thrust"],
        )
        self.set_input_defaults("propmodel.prop1.rpm", val=np.full(nn, 2000.0), units="rpm")
        # rst Propulsion (end)

        # ======================================== Aerodynamics ========================================
        # rst Aero (beg)
        # Use a different drag coefficient for takeoff versus cruise
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
        # Empty weight calculation; requires many aircraft inputs, see SingleTurboPropEmptyWeight source for more details.
        # This OEW calculation is not used in the weight calculation, but it is a useful output for aircraft design/optimization.
        self.add_subsystem(
            "OEW",
            SingleTurboPropEmptyWeight(),
            promotes_inputs=["*", ("P_TO", "ac|propulsion|engine|rating")],
            promotes_outputs=["OEW"],
        )
        self.connect("propmodel.prop1.component_weight", "W_propeller")
        self.connect("propmodel.eng1.component_weight", "W_engine")

        # Airplanes that consume fuel need to integrate fuel usage across the mission and subtract it from TOW
        intfuel = self.add_subsystem(
            "intfuel",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_outputs=["fuel_used_final"],
        )
        intfuel.add_integrand("fuel_used", rate_name="fuel_flow", val=1.0, units="kg")
        self.connect("propmodel.fuel_flow", "intfuel.fuel_flow")

        # Compute weight as MTOW minus fuel burned (assumes takeoff at MTOW)
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
class TBMAnalysisGroup(om.Group):
    """
    This is an example of a balanced field takeoff and three-phase mission analysis.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=11)

    def setup(self):
        nn = self.options["num_nodes"]

        # Define a bunch of design varaiables and airplane-specific parameters
        dv_comp = self.add_subsystem("dv_comp", DictIndepVarComp(acdata), promotes_outputs=["*"])

        # Aerodynamic parameters
        dv_comp.add_output_from_dict("ac|aero|CLmax_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|e")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_cruise")

        # Geometric parameters
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

        # Weight parameters
        dv_comp.add_output_from_dict("ac|weights|MTOW")
        dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
        dv_comp.add_output_from_dict("ac|weights|MLW")

        # Propulsion parameters
        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")
        dv_comp.add_output_from_dict("ac|propulsion|propeller|diameter")

        # Other parameters
        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|q_cruise")

        # Run a full mission analysis including takeoff, climb, cruise, and descent
        self.add_subsystem(
            "analysis",
            FullMissionAnalysis(num_nodes=nn, aircraft_model=TBM850AirplaneModel),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        # rst Mission (end)


# rst Setup problem (beg)
def run_tbm_analysis():
    # Set up OpenMDAO to analyze the airplane
    nn = 11
    prob = om.Problem()
    prob.model = TBMAnalysisGroup(num_nodes=nn)
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.setup()

    # Set required mission parameters. Each phase needs a vertical speed and airspeed.
    # The entire mission needs a cruise altitude and range.
    prob.set_val("climb.fltcond|vs", np.full(nn, 1500.0), units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.full(nn, 124.0), units="kn")
    prob.set_val("cruise.fltcond|vs", np.full(nn, 0.01), units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.full(nn, 201.0), units="kn")
    prob.set_val("descent.fltcond|vs", np.full(nn, -600.0), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.full(nn, 140.0), units="kn")

    prob.set_val("cruise|h0", 28e3, units="ft")
    prob.set_val("mission_range", 500, units="nmi")

    # Guesses for takeoff speeds to help with convergence
    prob.set_val("v0v1.fltcond|Utrue", np.full(nn, 50), units="kn")
    prob.set_val("v1vr.fltcond|Utrue", np.full(nn, 85), units="kn")
    prob.set_val("v1v0.fltcond|Utrue", np.full(nn, 85), units="kn")

    # Set some airplane-specific values. The throttle edits are to derate the takeoff power of the PT6A
    prob["climb.OEW.structural_fudge"] = 1.67
    prob["v0v1.throttle"] = np.full(nn, 0.826)
    prob["v1vr.throttle"] = np.full(nn, 0.826)
    prob["rotate.throttle"] = np.full(nn, 0.826)

    return prob
    # rst Setup problem (end)


# rst Run (beg)
if __name__ == "__main__":
    # Process command line argument to optionally not show figures and N2 diagram
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hide_visuals",
        default=False,
        action="store_true",
        help="Do not show matplotlib figure or open N2 diagram in browser",
    )
    hide_viz = parser.parse_args().hide_visuals

    # Run the analysis
    prob = run_tbm_analysis()
    prob.run_model()

    # Generate N2 diagram
    om.n2(prob, outfile="turboprop_n2.html", show_browser=not hide_viz)

    # =============== Print some useful outputs ================
    print_vars = [
        {"var": "ac|weights|MTOW", "name": "MTOW", "units": "lb"},
        {"var": "climb.OEW", "name": "OEW", "units": "lb"},
        {"var": "rotate.fuel_used_final", "name": "Rotate fuel", "units": "lb"},
        {"var": "climb.fuel_used_final", "name": "Climb fuel", "units": "lb"},
        {"var": "cruise.fuel_used_final", "name": "Cruise fuel", "units": "lb"},
        {"var": "descent.fuel_used_final", "name": "Fuel used", "units": "lb"},
        {"var": "rotate.range_final", "name": "TOFL (over 35ft obstacle)", "units": "ft"},
        {"var": "engineoutclimb.gamma", "name": "Climb angle at V2", "units": "deg"},
    ]
    print("\n=======================================================================\n")
    for var in print_vars:
        print(f"{var['name']}: {prob.get_val(var['var'], units=var['units']).item()} {var['units']}")

    # =============== Takeoff plot ================
    import matplotlib.pyplot as plt

    takeoff_fig, takeoff_axs = plt.subplots(1, 3, figsize=[9, 2.7], constrained_layout=True)
    takeoff_axs = takeoff_axs.flatten()  # change 1x3 mtx of axes into 4-element vector

    # Define variables to plot
    takeoff_vars = [
        {"var": "fltcond|h", "name": "Altitude", "units": "ft"},
        {"var": "fltcond|Utrue", "name": "True airspeed", "units": "kn"},
        {"var": "throttle", "name": "Throttle", "units": None},
    ]

    for idx_fig, var in enumerate(takeoff_vars):
        takeoff_axs[idx_fig].set_xlabel("Range (ft)")
        takeoff_axs[idx_fig].set_ylabel(f"{var['name']}" if var["units"] is None else f"{var['name']} ({var['units']})")

        # Loop through each flight phase and plot the current variable from each
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i, phase in enumerate(["v0v1", "v1vr", "rotate", "v1v0"]):
            takeoff_axs[idx_fig].plot(
                prob.get_val(f"{phase}.range", units="ft"),
                prob.get_val(f"{phase}.{var['var']}", units=var["units"]),
                "-o",
                c=colors[i],
                markersize=2.0,
            )

    takeoff_fig.legend(
        [r"V0 $\rightarrow$ V1", r"V1 $\rightarrow$ Vr", "Rotate", r"V1 $\rightarrow$ V0"],
        loc=(0.067, 0.6),
        fontsize="small",
    )
    takeoff_fig.suptitle("Takeoff phases")
    takeoff_fig.savefig("turboprop_takeoff_results.svg", transparent=True)

    # =============== Mission plot ================
    mission_fig, mission_axs = plt.subplots(2, 3, figsize=[9, 4.8], constrained_layout=True)
    mission_axs = mission_axs.flatten()  # change 2x2 mtx of axes into 4-element vector

    # Define variables to plot
    mission_vars = [
        {"var": "fltcond|h", "name": "Altitude", "units": "ft"},
        {"var": "fltcond|vs", "name": "Vertical speed", "units": "ft/min"},
        {"var": "fltcond|Utrue", "name": "True airspeed", "units": "kn"},
        {"var": "throttle", "name": "Throttle", "units": None},
        {"var": "propmodel.fuel_flow", "name": "Fuel flow", "units": "g/s"},
        {"var": "weight", "name": "Weight", "units": "kg"},
    ]

    for idx_fig, var in enumerate(mission_vars):
        mission_axs[idx_fig].set_xlabel("Range (nmi)")
        mission_axs[idx_fig].set_ylabel(f"{var['name']}" if var["units"] is None else f"{var['name']} ({var['units']})")

        # Loop through each flight phase and plot the current variable from each
        for phase in ["climb", "cruise", "descent"]:
            mission_axs[idx_fig].plot(
                prob.get_val(f"{phase}.range", units="nmi"),
                prob.get_val(f"{phase}.{var['var']}", units=var["units"]),
                "-o",
                c="tab:blue",
                markersize=2.0,
            )

    mission_fig.suptitle("Mission")
    mission_fig.savefig("turboprop_mission_results.svg", transparent=True)
    if not hide_viz:
        plt.show()
    # rst Run (end)
