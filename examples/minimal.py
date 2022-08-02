"""
This is a minimal example meant to show the absolute
simplest aircraft model and mission analysis you can
set up. This does not use OpenConcept's models for
the aircraft propulsion and aerodynamics, but it
does use the mission analysis methods.
"""
import numpy as np
import openmdao.api as om
from openconcept.analysis.performance.mission_profiles import BasicMission

# rst Aircraft (beg)
class Aircraft(om.ExplicitComponent):
    """
    An overly simplified aircraft model. This one is
    defined as an explicit component to use simple equations
    to compute weight, drag, and thrust. In practice, it would be
    an OpenMDAO group that integrates models for propulsion,
    aerodynamics, weights, etc.
    """
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")
        self.options.declare("flight_phase", default=None)  # required by OpenConcept but unused in this example
        self.options.declare("weight", default=2e3, desc="Aircraft weight in kg")
        self.options.declare("max_thrust", default=1e4, desc="Maximum thrust in Newtons")
        self.options.declare("L/D", default=10., desc="Lift to drag ratio")

    def setup(self):
        nn = self.options["num_nodes"]

        # Inputs passed from the mission analysis
        self.add_input("fltcond|CL", shape=nn)  # lift coefficient
        self.add_input("fltcond|q", shape=nn, units="Pa")  # dynamic pressure
        self.add_input("throttle", shape=nn)  # throttle from 0 to 1
        self.add_input("ac|geom|wing|S_ref", shape=1, units="m**2")  # wing planform area

        # Outputs sent back to the mission analysis
        self.add_output("weight", shape=nn, units="kg")
        self.add_output("drag", shape=nn, units="N")
        self.add_output("thrust", shape=nn, units="N")

        # Use complex step for this simple example
        self.declare_partials(["*"], ["*"], method="cs")
    
    def compute(self, inputs, outputs):
        outputs["weight"] = self.options["weight"]
        outputs["thrust"] = inputs["throttle"] * self.options["max_thrust"]
        outputs["drag"] = inputs["fltcond|q"] * inputs["fltcond|CL"] * inputs["ac|geom|wing|S_ref"] / self.options["L/D"]
# rst Aircraft (end)

class MissionAnalysis(om.Group):
    """
    OpenMDAO group for basic three-phase climb, cruise, descent mission.
    The only top-level aircraft variable that the aircraft model uses is
    the wing area, so that must be defined.
    """
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")

    def setup(self):
         # Define the mission
        self.add_subsystem("mission", BasicMission(num_nodes=self.options["num_nodes"], aircraft_model=Aircraft))

        # Set the wing area that is promoted by BasicMisison
        self.set_input_defaults("mission.ac|geom|wing|S_ref", val=25., units="m**2")


# Define the OpenMDAO problem
nn = 11
prob = om.Problem()
prob.model = MissionAnalysis(num_nodes=nn)

# Set up the solver
prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
prob.model.linear_solver = om.DirectSolver()

# Set up the problem
prob.setup()

# Define the mission profile by setting vertical speed and airspeed for each segment
prob.set_val("mission.climb.fltcond|vs", np.full(nn, 500.), units="ft/min")
prob.set_val("mission.cruise.fltcond|vs", np.full(nn, 0.), units="ft/min")
prob.set_val("mission.descent.fltcond|vs", np.full(nn, -500.), units="ft/min")
prob.set_val("mission.climb.fltcond|Ueas", np.full(nn, 150.), units="kn")
prob.set_val("mission.cruise.fltcond|Ueas", np.full(nn, 200.), units="kn")
prob.set_val("mission.descent.fltcond|Ueas", np.full(nn, 150.), units="kn")

# The mission also needs the cruise altitude and the range
prob.set_val("mission.cruise|h0", 15e3, units="ft")
prob.set_val("mission.mission_range", 400., units="nmi")

# Run the analysis
prob.run_model()

om.n2(prob, show_browser=False)

import matplotlib.pyplot as plt

var = "fltcond|h"
units = "ft"

for phase in ["climb", "cruise", "descent"]:
    plt.plot(
        prob.get_val(f"mission.{phase}.range", units="nmi"),
        prob.get_val(f"mission.{phase}.{var}", units=units),
        "-b"
    )

plt.xlabel("Range (nmi)")
plt.ylabel(f"{var}, {units}")
plt.show()
