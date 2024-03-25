"""
@File    :   B738_sizing.py
@Date    :   2023/03/25
@Author  :   Eytan Adler
@Description : Boeing 737-800 estimate using empirical weight and drag buildups
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================
from openconcept.examples.aircraft_data.B738_sizing import data as acdata
from openconcept.aerodynamics import PolarDrag, ParasiteDragCoefficient_JetTransport, CleanCLmax, FlapCLmax
from openconcept.propulsion import RubberizedTurbofan
from openconcept.geometry import CylinderSurfaceArea, WingMACTrapezoidal
from openconcept.stability import HStabVolumeCoefficientSizing, VStabVolumeCoefficientSizing
from openconcept.weights import JetTransportEmptyWeight
from openconcept.mission import FullMissionWithReserve
from openconcept.utilities import Integrator, AddSubtractComp, ElementMultiplyDivideComp, DictIndepVarComp


class B738AircraftModel(om.Group):
    """
    A Boeing 737-800 aircraft model group. Instead of using known weight
    and drag estimates of the existing airplane, this group uses empirical
    weight and drag buildups to enable the design of clean sheet aircraft.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int, desc="Number of analysis points to run")
        self.options.declare("flight_phase", default=None, types=str, desc="Phase of mission this group lives in")

    def setup(self):
        nn = self.options["num_nodes"]
        phase = self.options["flight_phase"]
        in_takeoff = phase in ["v0v1", "v1v0", "v1vr", "rotate"]

        # ==============================================================================
        # Aerodynamics
        # ==============================================================================
        # -------------- Zero-lift drag coefficient buildup --------------
        drag_buildup_promotes = [
            "fltcond|Utrue",
            "fltcond|rho",
            "fltcond|T",
            "ac|geom|fuselage|length",
            "ac|geom|fuselage|height",
            "ac|geom|fuselage|S_wet",
            "ac|geom|hstab|S_ref",
            "ac|geom|hstab|AR",
            "ac|geom|hstab|taper",
            "ac|geom|hstab|toverc",
            "ac|geom|vstab|S_ref",
            "ac|geom|vstab|AR",
            "ac|geom|vstab|taper",
            "ac|geom|vstab|toverc",
            "ac|geom|wing|S_ref",
            "ac|geom|wing|AR",
            "ac|geom|wing|taper",
            "ac|geom|wing|toverc",
            "ac|geom|nacelle|length",
            "ac|geom|nacelle|S_wet",
            "ac|propulsion|num_engines",
        ]
        if in_takeoff:
            drag_buildup_promotes += ["ac|aero|takeoff_flap_deg", "ac|geom|wing|c4sweep"]
        self.add_subsystem(
            "zero_lift_drag",
            ParasiteDragCoefficient_JetTransport(num_nodes=nn, configuration="takeoff" if in_takeoff else "clean"),
            promotes_inputs=drag_buildup_promotes,
        )

        # -------------- Drag polar --------------
        self.add_subsystem(
            "drag_polar",
            PolarDrag(num_nodes=nn, vec_CD0=True),
            promotes_inputs=[
                "fltcond|CL",
                "fltcond|q",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                ("e", "ac|aero|polar|e"),
            ],
            promotes_outputs=["drag"],
        )
        self.connect("zero_lift_drag.CD0", "drag_polar.CD0")

        # ==============================================================================
        # Propulsion
        # ==============================================================================
        # -------------- CFM56 engine surrogate model --------------
        self.add_subsystem(
            "CFM56",
            RubberizedTurbofan(num_nodes=nn, engine="CFM56"),
            promotes_inputs=["throttle", "fltcond|h", "fltcond|M", "ac|propulsion|engine|rating"],
        )

        # -------------- Multiply fuel flow and thrust by the number of active engines --------------
        # propulsor_active is 0 if failed engine and 1 otherwise, so
        # num active engines = num engines - 1 + propulsor_active
        self.add_subsystem(
            "num_engine_calc",
            AddSubtractComp(
                output_name="num_active_engines",
                input_names=["num_engines", "propulsor_active", "one"],
                vec_size=[1, nn, 1],
                scaling_factors=[1, 1, -1],
            ),
            promotes_inputs=[("num_engines", "ac|propulsion|num_engines"), "propulsor_active"],
        )
        self.set_input_defaults("num_engine_calc.one", 1.0)

        prop_mult = self.add_subsystem(
            "propulsion_multiplier", ElementMultiplyDivideComp(), promotes_outputs=["thrust"]
        )
        prop_mult.add_equation(
            output_name="thrust",
            input_names=["thrust_per_engine", "num_active_engines_1"],
            vec_size=nn,
            input_units=["lbf", None],
        )
        prop_mult.add_equation(
            output_name="fuel_flow",
            input_names=["fuel_flow_per_engine", "num_active_engines_2"],
            vec_size=nn,
            input_units=["kg/s", None],
        )
        self.connect("CFM56.fuel_flow", "propulsion_multiplier.fuel_flow_per_engine")
        self.connect("CFM56.thrust", "propulsion_multiplier.thrust_per_engine")

        # This hacky thing is necessary to enable two equations to pull from the same input
        self.connect(
            "num_engine_calc.num_active_engines",
            ["propulsion_multiplier.num_active_engines_1", "propulsion_multiplier.num_active_engines_2"],
        )

        # ==============================================================================
        # Weight
        # ==============================================================================
        # -------------- Integrate fuel burn --------------
        integ = self.add_subsystem(
            "fuel_burn_integ", Integrator(num_nodes=nn, diff_units="s", method="simpson", time_setup="duration")
        )
        integ.add_integrand(
            "fuel_burn",
            rate_name="fuel_flow",
            rate_units="kg/s",
            lower=0.0,
            upper=1e6,
        )
        self.connect("propulsion_multiplier.fuel_flow", "fuel_burn_integ.fuel_flow")

        # -------------- Subtract fuel burn from takeoff weight --------------
        self.add_subsystem(
            "weight_calc",
            AddSubtractComp(
                output_name="weight",
                input_names=["ac|weights|MTOW", "fuel_burn"],
                scaling_factors=[1, -1],
                vec_size=[1, nn],
                units="kg",
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["weight"],
        )
        self.connect("fuel_burn_integ.fuel_burn", "weight_calc.fuel_burn")


class B738SizingMissionAnalysis(om.Group):
    """
    Group that performs mission analysis, geometry calculations, and operating empty weight estimate.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=11, types=int, desc="Analysis points per mission phase")

    def setup(self):
        nn = self.options["num_nodes"]

        # ==============================================================================
        # Variables from B738_sizing data file
        # ==============================================================================
        dv = self.add_subsystem("ac_vars", DictIndepVarComp(acdata), promotes_outputs=["*"])
        dv_outputs = [
            # -------------- Aero --------------
            "ac|aero|polar|e",
            "ac|aero|Mach_max",
            "ac|aero|Vstall_land",
            "ac|aero|airfoil_Cl_max",
            "ac|aero|takeoff_flap_deg",
            # -------------- Propulsion --------------
            "ac|propulsion|engine|rating",
            "ac|propulsion|num_engines",
            # -------------- Geometry --------------
            # Wing
            "ac|geom|wing|S_ref",
            "ac|geom|wing|AR",
            "ac|geom|wing|c4sweep",
            "ac|geom|wing|taper",
            "ac|geom|wing|toverc",
            # Horizontal stabilizer
            "ac|geom|hstab|AR",
            "ac|geom|hstab|c4sweep",
            "ac|geom|hstab|taper",
            "ac|geom|hstab|toverc",
            # Vertical stabilizer
            "ac|geom|vstab|AR",
            "ac|geom|vstab|c4sweep",
            "ac|geom|vstab|taper",
            "ac|geom|vstab|toverc",
            # Fuselage
            "ac|geom|fuselage|length",
            "ac|geom|fuselage|height",
            # Nacelle
            "ac|geom|nacelle|length",
            "ac|geom|nacelle|diameter",
            # Main gear
            "ac|geom|maingear|length",
            "ac|geom|maingear|num_wheels",
            "ac|geom|maingear|num_shock_struts",
            # Nose gear
            "ac|geom|nosegear|length",
            "ac|geom|nosegear|num_wheels",
            # -------------- Weights --------------
            "ac|weights|W_payload",
            # -------------- Miscellaneous --------------
            "ac|num_passengers_max",
            "ac|num_flight_deck_crew",
            "ac|num_cabin_crew",
            "ac|cabin_pressure",
        ]
        for output_name in dv_outputs:
            dv.add_output_from_dict(output_name)

        # ==============================================================================
        # Geometry
        # ==============================================================================
        # -------------- Estimate wing to tail quarter chord as half fuselage length --------------
        self.add_subsystem(
            "tail_lever_arm_estimate",
            AddSubtractComp(
                output_name="c4_to_wing_c4",
                input_names=["fuselage_length"],
                units="m",
                scaling_factors=[0.5],
            ),
            promotes_inputs=[("fuselage_length", "ac|geom|fuselage|length")],
        )
        self.connect(
            "tail_lever_arm_estimate.c4_to_wing_c4", ["ac|geom|hstab|c4_to_wing_c4", "ac|geom|vstab|c4_to_wing_c4"]
        )

        # -------------- Compute mean aerodynamic chord assuming trapezoidal wing --------------
        self.add_subsystem(
            "wing_MAC",
            WingMACTrapezoidal(),
            promotes_inputs=[
                ("S_ref", "ac|geom|wing|S_ref"),
                ("AR", "ac|geom|wing|AR"),
                ("taper", "ac|geom|wing|taper"),
            ],
            promotes_outputs=[("MAC", "ac|geom|wing|MAC")],
        )

        # -------------- Vertical and horizontal tail area --------------
        self.add_subsystem(
            "vstab_area",
            VStabVolumeCoefficientSizing(),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|AR", "ac|geom|vstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|vstab|S_ref"],
        )
        self.add_subsystem(
            "hstab_area",
            HStabVolumeCoefficientSizing(),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|MAC", "ac|geom|hstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|hstab|S_ref"],
        )

        # -------------- Compute the fuselage and nacelle wetted areas assuming a cylinder --------------
        self.add_subsystem(
            "nacelle_wetted_area",
            CylinderSurfaceArea(),
            promotes_inputs=[("L", "ac|geom|nacelle|length"), ("D", "ac|geom|nacelle|diameter")],
            promotes_outputs=[("A", "ac|geom|nacelle|S_wet")],
        )
        self.add_subsystem(
            "fuselage_wetted_area",
            CylinderSurfaceArea(),
            promotes_inputs=[("L", "ac|geom|fuselage|length"), ("D", "ac|geom|fuselage|height")],
            promotes_outputs=[("A", "ac|geom|fuselage|S_wet")],
        )

        # ==============================================================================
        # Operating empty weight
        # ==============================================================================
        # Estimate MLW as 80% of MTOW so it's not necessary to know apriori.
        # MLW is used only for the landing gear weight estimate, so it's accuracy isn't hugely important.
        self.add_subsystem(
            "MLW_calc",
            AddSubtractComp(
                output_name="ac|weights|MLW",
                input_names=["ac|weights|MTOW"],
                units="kg",
                scaling_factors=[0.8],
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["ac|weights|MLW"],
        )

        self.add_subsystem(
            "empty_weight",
            JetTransportEmptyWeight(),
            promotes_inputs=[
                "ac|num_passengers_max",
                "ac|num_flight_deck_crew",
                "ac|num_cabin_crew",
                "ac|cabin_pressure",
                "ac|aero|Mach_max",
                "ac|aero|Vstall_land",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|taper",
                "ac|geom|wing|toverc",
                "ac|geom|hstab|S_ref",
                "ac|geom|hstab|AR",
                "ac|geom|hstab|c4sweep",
                "ac|geom|hstab|c4_to_wing_c4",
                "ac|geom|vstab|S_ref",
                "ac|geom|vstab|AR",
                "ac|geom|vstab|c4sweep",
                "ac|geom|vstab|toverc",
                "ac|geom|vstab|c4_to_wing_c4",
                "ac|geom|fuselage|height",
                "ac|geom|fuselage|length",
                "ac|geom|fuselage|S_wet",
                "ac|geom|maingear|length",
                "ac|geom|maingear|num_wheels",
                "ac|geom|maingear|num_shock_struts",
                "ac|geom|nosegear|length",
                "ac|geom|nosegear|num_wheels",
                "ac|propulsion|engine|rating",
                "ac|propulsion|num_engines",
                "ac|weights|MTOW",
                "ac|weights|MLW",
            ],
            promotes_outputs=[("OEW", "ac|weights|OEW")],
        )

        # ==============================================================================
        # CL max in cruise and takeoff
        # ==============================================================================
        self.add_subsystem(
            "CL_max_cruise",
            CleanCLmax(),
            promotes_inputs=["ac|aero|airfoil_Cl_max", "ac|geom|wing|c4sweep"],
            promotes_outputs=[("CL_max_clean", "ac|aero|CLmax_cruise")],
        )
        self.add_subsystem(
            "CL_max_takeoff",
            FlapCLmax(),
            promotes_inputs=[
                ("flap_extension", "ac|aero|takeoff_flap_deg"),
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                ("CL_max_clean", "ac|aero|CLmax_cruise"),
            ],
            promotes_outputs=[("CL_max_flap", "ac|aero|CLmax_TO")],
        )

        # ==============================================================================
        # Remaining misc stuff and input settings
        # ==============================================================================
        # -------------- Set MTOW to OEW + payload + fuel burn --------------
        self.add_subsystem(
            "MTOW_calc",
            AddSubtractComp(
                output_name="MTOW",
                input_names=["OEW", "W_payload", "W_fuel"],
                units="kg",
                lower=1e-6,
            ),
            promotes_inputs=[("OEW", "ac|weights|OEW"), ("W_payload", "ac|weights|W_payload")],
            promotes_outputs=[("MTOW", "ac|weights|MTOW")],
        )
        self.connect(
            "mission.loiter.fuel_burn_integ.fuel_burn_final",
            ["MTOW_calc.W_fuel", "empty_weight.ac|weights|W_fuel_max"],
        )

        # -------------- Initial guesses for important solver states for better performance --------------
        self.set_input_defaults("ac|weights|MTOW", 50e3, units="kg")
        for fuel_burn_var in ["MTOW_calc.W_fuel", "empty_weight.ac|weights|W_fuel_max"]:
            self.set_input_defaults(fuel_burn_var, 30e3, units="kg")

        # ==============================================================================
        # Mission analysis
        # ==============================================================================
        self.add_subsystem(
            "mission",
            FullMissionWithReserve(num_nodes=nn, aircraft_model=B738AircraftModel),
            promotes_inputs=["ac|*"],
        )


def set_mission_profile(prob):
    """
    Set the parameters in the OpenMDAO problem that define the mission profile.

    Parameters
    ----------
    prob : OpenMDAO Problem
        Problem with B378MissionAnalysis as model in which to set values
    """
    # Get the number of nodes in the mission
    nn = prob.model.options["num_nodes"]

    # ==============================================================================
    # Basic mission phases
    # ==============================================================================
    # -------------- Climb --------------
    prob.set_val("mission.climb.fltcond|vs", np.linspace(2300.0, 400.0, nn), units="ft/min")
    prob.set_val("mission.climb.fltcond|Ueas", np.linspace(230, 252, nn), units="kn")

    # -------------- Cruise --------------
    prob.set_val("mission.cruise.fltcond|vs", np.full((nn,), 0.0), units="ft/min")
    prob.set_val("mission.cruise.fltcond|Ueas", np.linspace(252, 252, nn), units="kn")

    # -------------- Descent --------------
    prob.set_val("mission.descent.fltcond|vs", np.linspace(-1300, -800, nn), units="ft/min")
    prob.set_val("mission.descent.fltcond|Ueas", np.linspace(252, 250, nn), units="kn")

    # ==============================================================================
    # Reserve mission phases
    # ==============================================================================
    # -------------- Reserve climb --------------
    prob.set_val("mission.reserve_climb.fltcond|vs", np.linspace(3000.0, 2300.0, nn), units="ft/min")
    prob.set_val("mission.reserve_climb.fltcond|Ueas", np.linspace(230, 230, nn), units="kn")

    # -------------- Reserve cruise --------------
    prob.set_val("mission.reserve_cruise.fltcond|vs", np.full((nn,), 0.0), units="ft/min")
    prob.set_val("mission.reserve_cruise.fltcond|Ueas", np.linspace(250, 250, nn), units="kn")

    # -------------- Reserve descent --------------
    prob.set_val("mission.reserve_descent.fltcond|vs", np.linspace(-800, -800, nn), units="ft/min")
    prob.set_val("mission.reserve_descent.fltcond|Ueas", np.full((nn,), 250.0), units="kn")

    # -------------- Loiter --------------
    prob.set_val("mission.loiter.fltcond|vs", np.linspace(0.0, 0.0, nn), units="ft/min")
    prob.set_val("mission.loiter.fltcond|Ueas", np.full((nn,), 250.0), units="kn")

    # ==============================================================================
    # Other parameters
    # ==============================================================================
    prob.set_val("mission.cruise|h0", 35000.0, units="ft")
    prob.set_val("mission.reserve|h0", 15000.0, units="ft")
    prob.set_val("mission.mission_range", 2800, units="nmi")

    # -------------- Set takeoff speed guesses to improve solver performance --------------
    prob.set_val("mission.v0v1.fltcond|Utrue", np.full((nn,), 100.0), units="kn")
    prob.set_val("mission.v1vr.fltcond|Utrue", np.full((nn,), 100.0), units="kn")
    prob.set_val("mission.v1v0.fltcond|Utrue", np.full((nn,), 100.0), units="kn")

    # Converge the model first with an easier mission profile and work up to the intended
    # mission profile. This is needed to help the Newton solver converge the actual mission.
    prob.set_val("mission.descent.fltcond|vs", np.linspace(-800, -800, nn), units="ft/min")
    prob.set_val("mission.cruise|h0", 5000.0, units="ft")
    prob.set_val("mission.reserve|h0", 1000.0, units="ft")
    prob.set_val("mission.mission_range", 500, units="nmi")
    prob.set_val("mission.reserve_range", 100, units="nmi")
    prob.run_model()

    # Almost there, just not quite with descent rate
    prob.set_val("mission.cruise|h0", 35000.0, units="ft")
    prob.set_val("mission.reserve|h0", 15000.0, units="ft")
    prob.set_val("mission.mission_range", 2800, units="nmi")
    prob.set_val("mission.reserve_range", 200, units="nmi")
    prob.run_model()

    # Finally, set the descent rate we want
    prob.set_val("mission.descent.fltcond|vs", np.linspace(-1300, -800, nn), units="ft/min")


def plot_results(prob, filename=None):
    """
    Make a plot with the results of the mission analysis.

    Parameters
    ----------
    prob : OpenMDAO Problem
        Problem with B738SizingMissionAnalysis model that has been run
    filename : str (optional)
        Filename to save to, by default will show plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    axs = axs.flatten()

    for phase in ["climb", "cruise", "descent", "reserve_climb", "reserve_cruise", "reserve_descent", "loiter"]:
        dist = prob.get_val(f"mission.{phase}.range", units="nmi")

        axs[0].plot(dist, prob.get_val(f"mission.{phase}.fltcond|h", units="ft"), color="tab:blue")
        axs[1].plot(dist, prob.get_val(f"mission.{phase}.fltcond|M"), color="tab:blue")
        axs[2].plot(dist, prob.get_val(f"mission.{phase}.fltcond|vs", units="ft/min"), color="tab:blue")
        axs[3].plot(dist, prob.get_val(f"mission.{phase}.weight", units="lb"), color="tab:blue")
        axs[4].plot(dist, prob.get_val(f"mission.{phase}.drag", units="lbf"), color="tab:blue")
        axs[4].plot(dist, prob.get_val(f"mission.{phase}.thrust", units="lbf"), color="tab:orange")
        axs[5].plot(dist, prob.get_val(f"mission.{phase}.throttle") * 100, color="tab:blue")

    axs[0].set_ylabel("Altitude (ft)")
    axs[1].set_ylabel("Mach number")
    axs[2].set_ylabel("Vertical speed (ft/min)")
    axs[3].set_ylabel("Weight (lb)")
    axs[4].set_ylabel("Longitudinal force (lb)")
    axs[5].set_ylabel("Throttle (%)")
    axs[4].legend(["Drag", "Thrust"])
    for i in range(6):
        axs[i].set_xlabel("Distance flown (nmi)")
        axs[i].spines[["right", "top"]].set_visible(False)
        if i != 1:
            axs[i].get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ",")))
        axs[i].get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ",")))

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


def run_738_sizing_analysis(num_nodes=21):
    p = om.Problem()
    p.model = B738SizingMissionAnalysis(num_nodes=num_nodes)

    # -------------- Add solvers --------------
    p.model.nonlinear_solver = om.NewtonSolver()
    p.model.nonlinear_solver.options["iprint"] = 2
    p.model.nonlinear_solver.options["solve_subsystems"] = True
    p.model.nonlinear_solver.options["maxiter"] = 20
    p.model.nonlinear_solver.options["atol"] = 1e-9
    p.model.nonlinear_solver.options["rtol"] = 1e-9

    p.model.linear_solver = om.DirectSolver()

    p.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
    p.model.nonlinear_solver.linesearch.options["print_bound_enforce"] = False

    p.setup()

    set_mission_profile(p)

    p.run_model()

    return p


if __name__ == "__main__":
    p = run_738_sizing_analysis()
    om.n2(p, show_browser=False)

    # Print some useful numbers
    print("\n\n================= Computed values =================")
    print(f"MTOW: {p.get_val('ac|weights|MTOW', units='lb').item():.1f} lb")
    print(f"Payload weight: {p.get_val('ac|weights|W_payload', units='lb').item()} lb")
    print(f"OEW: {p.get_val('ac|weights|OEW', units='lb').item():.1f} lb")
    print(f"Fuel burned: {p.get_val('mission.descent.fuel_burn_integ.fuel_burn_final', units='lb').item():.1f} lb")
    print(f"CL max cruise: {p.get_val('ac|aero|CLmax_cruise').item():.3f}")
    print(f"CL max takeoff: {p.get_val('ac|aero|CLmax_TO').item():.3f}")
    print(f"Balanced field length (continue): {p.get_val('mission.bfl.distance_continue', units='ft').item():.1f} ft")
    print(
        f"Balanced field length (abort): {p.get_val('mission.bfl.distance_abort', units='ft').item():.1f} ft (this should be the same as continue)"
    )

    plot_results(p, filename="plot.pdf")
