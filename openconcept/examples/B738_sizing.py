"""
@File    :   B738_sizing.py
@Date    :   2023/03/25
@Author  :   Eytan Adler
@Description : Boeing 737-800 estimate using empirical weight and drag buildups
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================
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
            drag_buildup_promotes += ["ac|takeoff_flap_deg", "ac|geom|wing|c4sweep"]
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

        # -------------- Compute CL max in cruise and takeoff --------------
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
                ("flap_extension", "ac|takeoff_flap_deg"),
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                ("CL_max_clean", "ac|aero|CLmax_cruise"),
            ],
            promotes_outputs=[("CL_max_flap", "ac|aero|CLmax_TO")],
        )

        # ==============================================================================
        # Propulsion
        # ==============================================================================
        # -------------- CFM56 engine surrogate model --------------
        self.add_subsystem(
            "CFM56",
            RubberizedTurbofan(num_nodes=nn, engine="CFM56"),
            promotes_inputs=["throttle", "fltcond|h", "fltcond|M", "ac|propulsion|engine|rating"],
            promotes_outputs=["thrust"],
        )

        # -------------- Multiply fuel flow by the number of engines --------------
        self.add_subsystem(
            "fuel_flow_mult",
            ElementMultiplyDivideComp(
                "fuel_flow",
                input_names=["fuel_flow_per_engine", "ac|propulsion|num_engines"],
                vec_size=[nn, 1],
                input_units=["kg/s", None],
            ),
            promotes_inputs=["ac|propulsion|num_engines"],
        )
        self.connect("CFM56.fuel_flow", "fuel_flow_mult.fuel_flow_per_engine")

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
        self.connect("fuel_flow_mult.fuel_flow", "fuel_burn_integ.fuel_flow")

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


class B738MissionAnalysis(om.Group):
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
        # self.add_subsystem("ac_vars", DictIndepVarComp)

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
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|AR"],
            promotes_outputs=["ac|geom|vstab|S_ref"],
        )
        self.add_subsystem(
            "hstab_area",
            HStabVolumeCoefficientSizing(),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|MAC"],
            promotes_outputs=["ac|geom|hstab|S_ref"],
        )
        self.connect(
            "tail_lever_arm_estimate.c4_to_wing_c4", ["ac|geom|hstab|c4_to_wing_c4", "ac|geom|vstab|c4_to_wing_c4"]
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
        # Mission analysis
        # ==============================================================================
        self.add_subsystem(
            "mission",
            FullMissionWithReserve(num_nodes=nn, aircraft_model=B738AircraftModel),
            promotes_inputs=["ac|*"],
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
            ),
            promotes_inputs=[("OEW", "ac|weights|OEW"), ("W_payload", "ac|weights|W_payload")],
            promotes_outputs=[("MTOW", "ac|weights|MTOW")],
        )
        self.connect(
            "mission.loiter.fuel_burn_integ.fuel_burn_final",
            ["MTOW_calc.W_fuel", "empty_weight.ac|weights|W_fuel_max"],
        )

        # -------------- Initial guesses for feedback variables for better solver performance --------------
        self.set_input_defaults("ac|weights|MTOW", 100e3, units="kg")
        for fuel_burn_var in ["MTOW_calc.W_fuel", "empty_weight.ac|weights|W_fuel_max"]:
            self.set_input_defaults(fuel_burn_var, 30e3, units="kg")

        # TODO: remove these once data file exists
        self.set_input_defaults("ac|geom|fuselage|length", 50, units="m")
        self.set_input_defaults("ac|geom|fuselage|height", 3, units="m")
        self.set_input_defaults("ac|geom|wing|S_ref", 100, units="m**2")


if __name__=="__main__":
    nn = 11
    p = om.Problem()
    p.model = B738MissionAnalysis(num_nodes=nn)
    p.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, maxiter=20, atol=1e-8, rtol=1e-8)
    p.model.linear_solver = om.DirectSolver()
    p.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(print_bound_enforce=False)
    p.setup()

    om.n2(p)
