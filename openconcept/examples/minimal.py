"""
This is a minimal example meant to show the absolute
simplest aircraft model and mission analysis you can
set up. This does not use OpenConcept's models for
the aircraft propulsion and aerodynamics, but it
does use the mission analysis methods.
"""
# rst Imports (beg)
import openmdao.api as om
from openconcept.mission import BasicMission
import numpy as np

# rst Imports (end)


# rst Aircraft (beg)
class Aircraft(om.ExplicitComponent):
    """
    An overly simplified aircraft model. This one is defined as an explicit component to use simple equations
    to compute weight, drag, and thrust. In practice, it would be an OpenMDAO group that integrates models for
    propulsion, aerodynamics, weights, etc.
    """

    # rst Options
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")
        self.options.declare("flight_phase", default=None)  # required by OpenConcept but unused in this example

    # rst Setup
    def setup(self):
        nn = self.options["num_nodes"]

        # ======== Inputs passed from the mission analysis ========
        # These are required by OpenConcept
        self.add_input("fltcond|CL", shape=nn)  # lift coefficient
        self.add_input("throttle", shape=nn)  # throttle from 0 to 1

        # These are additional inputs used by the model
        self.add_input("fltcond|q", shape=nn, units="Pa")  # dynamic pressure
        self.add_input("ac|geom|wing|S_ref", shape=1, units="m**2")  # wing planform area
        self.add_input("ac|weights|TOW", shape=1, units="kg")  # constant weight value
        self.add_input("ac|propulsion|max_thrust", shape=1, units="N")
        self.add_input("ac|aero|L_over_D", shape=1)

        # ======== Outputs sent back to the mission analysis ========
        self.add_output("weight", shape=nn, units="kg")
        self.add_output("drag", shape=nn, units="N")
        self.add_output("thrust", shape=nn, units="N")

        # ======== Use complex step for this simple example ========
        self.declare_partials(["*"], ["*"], method="cs")

    # rst Compute
    def compute(self, inputs, outputs):
        outputs["weight"] = inputs["ac|weights|TOW"]
        outputs["thrust"] = inputs["throttle"] * inputs["ac|propulsion|max_thrust"]
        outputs["drag"] = (
            inputs["fltcond|q"] * inputs["fltcond|CL"] * inputs["ac|geom|wing|S_ref"] / inputs["ac|aero|L_over_D"]
        )
        # rst Aircraft (end)


# rst Mission (beg)
class MissionAnalysis(om.Group):
    """
    OpenMDAO group for basic three-phase climb, cruise, descent mission.
    The only top-level aircraft variable that the aircraft model uses is
    the wing area, so that must be defined.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")

    def setup(self):
        iv = self.add_subsystem("ac_vars", om.IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("ac|geom|wing|S_ref", val=25.0, units="m**2")
        iv.add_output("ac|weights|TOW", val=5e3, units="kg")
        iv.add_output("ac|propulsion|max_thrust", val=1e4, units="N")
        iv.add_output("ac|aero|L_over_D", val=10.0)

        # Define the mission
        self.add_subsystem(
            "mission",
            BasicMission(aircraft_model=Aircraft, num_nodes=self.options["num_nodes"]),
            promotes_inputs=["ac|*"],
        )
        # rst Mission (end)


# rst Setup problem (beg)
def setup_problem(model=MissionAnalysis):
    """
    Define the OpenMDAO problem
    """
    nn = 11
    prob = om.Problem()
    prob.model = model(num_nodes=nn)

    # Set up the solver
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()

    # Set up the problem
    prob.setup()

    # Define the mission profile by setting vertical speed and airspeed for each segment
    prob.set_val("mission.climb.fltcond|vs", np.full(nn, 500.0), units="ft/min")
    prob.set_val("mission.cruise.fltcond|vs", np.full(nn, 0.0), units="ft/min")
    prob.set_val("mission.descent.fltcond|vs", np.full(nn, -500.0), units="ft/min")
    prob.set_val("mission.climb.fltcond|Ueas", np.full(nn, 150.0), units="kn")
    prob.set_val("mission.cruise.fltcond|Ueas", np.full(nn, 200.0), units="kn")
    prob.set_val("mission.descent.fltcond|Ueas", np.full(nn, 150.0), units="kn")

    # The mission also needs the cruise altitude and the range
    prob.set_val("mission.cruise|h0", 15e3, units="ft")
    prob.set_val("mission.mission_range", 400.0, units="nmi")

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

    # Setup the problem and run the analysis
    prob = setup_problem()
    prob.run_model()

    # Generate N2 diagram
    om.n2(prob, outfile="minimal_example_n2.html", show_browser=not hide_viz)

    # Create plot with results
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs = axs.flatten()  # change 2x2 mtx of axes into 4-element vector

    # Define variables to plot
    plot_vars = [
        {"var": "fltcond|h", "name": "Altitude", "units": "ft"},
        {"var": "fltcond|vs", "name": "Vertical speed", "units": "ft/min"},
        {"var": "fltcond|Utrue", "name": "True airspeed", "units": "kn"},
        {"var": "throttle", "name": "Throttle", "units": None},
    ]

    for idx_fig, var in enumerate(plot_vars):
        axs[idx_fig].set_xlabel("Range (nmi)")
        axs[idx_fig].set_ylabel(f"{var['name']}" if var["units"] is None else f"{var['name']} ({var['units']})")

        # Loop through each flight phase and plot the current variable from each
        for phase in ["climb", "cruise", "descent"]:
            axs[idx_fig].plot(
                prob.get_val(f"mission.{phase}.range", units="nmi"),
                prob.get_val(f"mission.{phase}.{var['var']}", units=var["units"]),
                "-o",
                c="tab:blue",
                markersize=2.0,
            )

    fig.savefig("minimal_example_results.svg", transparent=True)
    if not hide_viz:
        plt.show()
# rst Run (end)
