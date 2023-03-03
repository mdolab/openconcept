"""
This example builds off the original minimal example,
but adds a numerical integrator to integrate fuel burn
and update the weight accordingly.
"""
# rst Imports (beg)
import openmdao.api as om
from openconcept.examples.minimal import Aircraft, setup_problem  # build off this aircraft model
from openconcept.mission import BasicMission
from openconcept.utilities import Integrator

# rst Imports (end)


# rst Aircraft (beg)
class AircraftWithFuelBurn(om.Group):
    """
    This model takes the simplified aircraft model from the minimal example, but adds
    a fuel flow computation using TSFC and integrates it to compute fuel burn.
    """

    # rst Options
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")
        self.options.declare("flight_phase", default=None)  # required by OpenConcept but unused in this example

    # rst Setup
    def setup(self):
        nn = self.options["num_nodes"]

        # Add the aircraft model from the minimal example to build off of.
        # Don't promote the weight from this model because we're going to compute a new
        # one using the fuel burn.
        # rst Simple aircraft (beg)
        self.add_subsystem(
            "simple_aircraft",
            Aircraft(num_nodes=nn),
            promotes_inputs=[
                "fltcond|CL",
                "throttle",
                "fltcond|q",
                "ac|geom|wing|S_ref",
                "ac|weights|TOW",
                "ac|propulsion|max_thrust",
                "ac|aero|L_over_D",
            ],
            promotes_outputs=["thrust", "drag"],
        )
        # rst Simple aircraft (end)

        # Use an OpenMDAO ExecComp to compute the fuel flow rate using the thrust and TSFC
        # rst Fuel flow (beg)
        self.add_subsystem(
            "fuel_flow_calc",
            om.ExecComp(
                "fuel_flow = TSFC * thrust",
                fuel_flow={"units": "kg/s", "shape": nn},
                TSFC={"units": "kg/N/s", "shape": 1},
                thrust={"units": "N", "shape": nn},
            ),
            promotes_inputs=[("TSFC", "ac|propulsion|TSFC"), "thrust"],
        )
        # rst Fuel flow (end)

        # Integrate the fuel flow rate to compute fuel burn
        # rst Integrator (beg)
        integ = self.add_subsystem(
            "fuel_integrator", Integrator(num_nodes=nn, diff_units="s", time_setup="duration", method="simpson")
        )
        integ.add_integrand("fuel_burned", rate_name="fuel_flow", units="kg")

        self.connect("fuel_flow_calc.fuel_flow", "fuel_integrator.fuel_flow")
        # rst Integrator (end)

        # Compute the current weight by subtracting the fuel burned from the takeoff weight
        # rst Weight (beg)
        self.add_subsystem(
            "weight_calc",
            om.ExecComp(
                "weight = TOW - fuel_burned",
                units="kg",
                weight={"shape": nn},
                TOW={"shape": 1},
                fuel_burned={"shape": nn},
            ),
            promotes_inputs=[("TOW", "ac|weights|TOW")],
            promotes_outputs=["weight"],
        )
        self.connect("fuel_integrator.fuel_burned", "weight_calc.fuel_burned")
        # rst Weight (end)


# rst Mission (beg)
class MissionAnalysisWithFuelBurn(om.Group):
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
        iv.add_output("ac|propulsion|TSFC", val=20.0, units="g/kN/s")
        iv.add_output("ac|aero|L_over_D", val=10.0)

        # Define the mission
        self.add_subsystem(
            "mission",
            BasicMission(aircraft_model=AircraftWithFuelBurn, num_nodes=self.options["num_nodes"]),
            promotes_inputs=["ac|*"],
        )
        # rst Mission (end)


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
    prob = setup_problem(model=MissionAnalysisWithFuelBurn)
    prob.run_model()

    # Generate N2 diagram
    om.n2(prob, outfile="minimal_integrator_n2.html", show_browser=not hide_viz)

    # Create plot with results
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=[9, 4.8], constrained_layout=True)
    axs = axs.flatten()  # change 2x3 mtx of axes into 4-element vector

    # Define variables to plot
    plot_vars = [
        {"var": "fltcond|h", "name": "Altitude", "units": "ft"},
        {"var": "fltcond|vs", "name": "Vertical speed", "units": "ft/min"},
        {"var": "fltcond|Utrue", "name": "True airspeed", "units": "kn"},
        {"var": "throttle", "name": "Throttle", "units": None},
        {"var": "fuel_flow_calc.fuel_flow", "name": "Fuel flow", "units": "g/s"},
        {"var": "weight", "name": "Weight", "units": "kg"},
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

    fig.savefig("minimal_integrator_results.svg", transparent=True)
    if not hide_viz:
        plt.show()
# rst Run (end)
