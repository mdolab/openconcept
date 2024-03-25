import numpy as np
import openmdao.api as om
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp


class JetTransportEmptyWeight(om.Group):
    """
    Estimate of a jet transport aircraft's operating empty weight using a combination
    of weight estimation methods from Raymer, Roskam, and others. See the docstrings
    for individual weight components for more details on the models used.

    Inputs
    ------
    ac|num_passengers_max : float
        Maximum number of passengers (scalar, dimensionless)
    ac|num_flight_deck_crew : float
        Number of flight crew members (scalar, dimensionless)
    ac|num_cabin_crew : float
        Number of flight attendants (scalar, dimensionless)
    ac|cabin_pressure : float
        Cabin pressure (scalar, psi)
    ac|aero|Mach_max : float
        Maximum aircraft Mach number (scalar, dimensionless)
    ac|aero|Vstall_land : float
        Landing stall speed (scalar, knots)
    ac|geom|wing|S_ref : float
        Wing planform reference area (scalar, sq ft)
    ac|geom|wing|AR : float
        Wing aspect ratio (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Wing sweep at 25% mean aerodynamic chord (scalar, radians)
    ac|geom|wing|taper : float
        Wing taper ratio (scalar, dimensionless)
    ac|geom|wing|toverc : float
        Wing root thickness-to-chord ratio (scalar, dimensionless)
    ac|geom|hstab|S_ref : float
        Horizontal stabilizer reference area (scalar, sq ft)
    ac|geom|hstab|AR : float
        Horizontal stabilizer aspect ratio (scalar, dimensionless)
    ac|geom|hstab|c4sweep : float
        Horizontal stabilizer sweep at 25% mean aerodynamic chord (scalar, radians)
    ac|geom|hstab|c4_to_wing_c4 : float
        Distance from the horizontal stabilizer's quarter chord (of the MAC) to the wing's quarter chord (scalar, ft)
    ac|geom|vstab|S_ref : float
        Vertical stabilizer wing area (scalar, sq ft)
    ac|geom|vstab|AR : float
        Vertical stabilizer aspect ratio (scalar, dimensionless)
    ac|geom|vstab|c4sweep : float
        Vertical stabilizer sweep at 25% mean aerodynamic chord (scalar, radians)
    ac|geom|vstab|toverc : float
        Vertical stabilizer thickness-to-chord ratio (scalar, dimensionless)
    ac|geom|vstab|c4_to_wing_c4 : float
        Distance from the vertical stabilizer's quarter chord (of the MAC) to the wing's quarter chord (scalar, ft)
    ac|geom|fuselage|height : float
        Fuselage height (scalar, ft)
    ac|geom|fuselage|length : float
        Fuselage length, used to compute distance between quarter chord of wing and horizontal stabilizer (scalar, ft)
    ac|geom|fuselage|S_wet : float
        Fuselage wetted area (scalar, sq ft)
    ac|geom|maingear|length : float
        Length of the main landing gear (scalar, inch)
    ac|geom|maingear|num_wheels : float
        Total number of main landing gear wheels (scalar, dimensionless)
    ac|geom|maingear|num_shock_struts : float
        Total number of main landing gear shock struts (scalar, dimensionless)
    ac|geom|nosegear|length : float
        Length of the nose landing gear (scalar, inch)
    ac|geom|nosegear|num_wheels : float
        Total number of nose landing gear wheels (scalar, dimensionless)
    ac|propulsion|engine|rating : float
        Rated thrust of each engine (scalar, lbf)
    ac|propulsion|num_engines : float
        Number of engines (scalar, dimensionless)
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)
    ac|weights|MLW : float
        Maximum landing weight (scalar, lb)
    ac|weights|W_fuel_max : float
        Maximum fuel weight (scalar, lb)

    Outputs
    -------
    OEW : float
        Total operating empty weight (scalar, lb)
    W_wing : float
        Estimated wing weight without structural fudge factor multiplier (scalar, lb)
    W_hstab : float
        Weight of the horizontal stabilizer without structural fudge factor multiplier (scalar, lb)
    W_vstab : float
        Weight of the vertical stabilizer without structural fudge factor multiplier (scalar, lb)
    W_fuselage : float
        Fuselage weight without structural fudge factor multiplier (scalar, lb)
    W_mlg : float
        Main landing gear weight without structural fudge factor multiplier (scalar, lb)
    W_nlg : float
        Nose landing gear weight without structural fudge factor multiplier (scalar, lb)
    W_nacelle : float
        Weight of the nacelles (scalar, lb)
    W_structure : float
        Total structural weight = fudge factor * (W_wing + W_hstab + W_vstab + W_fuselage + W_mlg + W_nlg + W_nacelle) (scalar, lb)
    W_engines : float
        Total dry engine weight (scalar, lb)
    W_thrust_rev : float
        Total thrust reverser weight (scalar, lb)
    W_eng_control : float
        Total engine control weight (scalar, lb)
    W_fuelsystem : float
        Total fuel system weight including tanks and plumbing (scalar, lb)
    W_eng_start : float
        Total engine starter weight (scalar, lb)
    W_furnishings : float
        Weight estimate of seats, galleys, lavatories, and other furnishings (scalar, lb)
    W_flight_controls : float
        Flight control system weight (scalar, lb)
    W_avionics : float
        Intrumentation, avionics, and electronics weight (scalar, lb)
    W_electrical : float
        Electrical system weight (scalar, lb)
    W_ac_pressurize_antiice : float
        Air conditioning, pressurization, and anti-icing system weight (scalar, lb)
    W_oxygen : float
        Oxygen system weight (scalar, lb)
    W_APU : float
        Auxiliary power unit weight (scalar, lb)

    Options
    -------
    structural_fudge : float
        Multiplier on the structural weight to allow the user to account for miscellaneous items and
        advanced materials. Structural weight includes wing, horizontal stabilizer, vertical stabilizer,
        fuselage, landing gear, and nacelle weights. By default 1.2 (scalar, dimensionless)
    total_fudge : float
        Multiplier on the final operating empty weight estimate. Structural components have both the
        structural fudge and total fudge factors applied. By default 1.15 (scalar, dimensionless)
    wing_weight_multiplier : float
        Multiplier on wing weight. This can be used as a very rough way of increasing wing weight
        due to lack of inertial load relief from the fuel. By default 1.0 (scalar, dimensionless)
    n_ult : float
        Ultimate load factor, 1.5 x limit load factor, by default 1.5 x 2.5 (scalar, dimensionless)
    n_land_ult : float
        Ultimate landing load factor, which is 1.5 times the gear load factor (defined
        in equation 11.11). Table 11.5 gives reasonable gear load factor values for
        different aircraft types, with commercial aircraft in the 2.7-3 range. Default
        is taken at 2.8, thus the ultimate landing load factor is 2.8 x 1.5 (scalar, dimensionless)
    control_surface_area_frac : float
        Fraction of the total wing area covered by control surfaces and flaps, by default 0.1 (scalar, dimensionless)
    fuselage_width_frac : float
        Fuselage width at horizontal tail intersection divided by fuselage diameter, by default 0.5 (scalar, dimensionless)
    K_uht : float
        Correction for all-moving tail; set to 1.143 for all-moving tail or 1.0 otherwise, by default 1.0 (scalar, dimensionless)
    elevator_area_frac : float
        Fraction of horizontal stabilizer area covered by elevators, by default 0.2 (scalar, dimensionless)
    T_tail : bool
        True if the tail is a T-tail, False otherwise
    K_door : float
        Fuselage door parameter; 1 if no cargo door, 1.06 if one side cargo door, 1.12 if two side
        cargo doors, 1.12 if aft clamshell door, 1.25 if two side cargo doors, and aft clamshell door,
        by default 1 (scalar, dimensionless)
    K_lg : float
        Fuselage-mounted landing gear parameter; 1.12 if fuselage-mounted main landing gear and 1
        otherwise, by default 1 (scalar, dimensionless)
    kneeling_main_gear_parameter : float
        Set to 1.126 for kneeling main gear and 1 otherwise, by default 1 (scalar, dimensionless)
    kneeling_nose_gear_parameter : float
        Set to 1.15 for kneeling nose gear and 1 otherwise, by default 1 (scalar, dimensionless)
    K_lav : float
        Lavatory coefficient; 0.31 for short ranges and 1.11 for long ranges, by default 0.7
    K_buf : float
        Food provisions coefficient; 1.02 for short range and 5.68 for very long range, by default 4
    coeff_fc : float
        K_fc in Roskam times any additional coefficient. The book says take K_fc as 0.44 for un-powered
        flight controls and 0.64 for powered flight controls. Multiply this coefficient by 1.2 if leading
        edge devices are employed. If lift dumpers are employed, use a factor of 1.15. By default 1.2 * 0.64.
    coeff_avionics : float
        Roskam notes that the avionics weight estimates are probably conservative for modern computer-based
        flight management and navigation systems. This coefficient is multiplied by the Roskam estimate to
        account for this. By default 0.5.
    cabin_length_frac : float
        The length of the passenger cabin divided by the total fuselage length, by default 0.75.
    APU_weight_frac : float
        Auxiliary power unit weight divided by maximum takeoff weight, by deafult 0.0085.
    """

    def initialize(self):
        self.options.declare("structural_fudge", default=1.2, desc="Fudge factor on structural weights")
        self.options.declare("total_fudge", default=1.15, desc="Fudge factor applied to the final OEW value")
        self.options.declare("wing_weight_multiplier", default=1.0, desc="Multiplier on wing weight")
        self.options.declare("n_ult", default=2.5 * 1.5, desc="Ultimate load factor (dimensionless)")
        self.options.declare("n_land_ult", default=2.8 * 1.5, desc="ultimate landing load factor")
        self.options.declare(
            "control_surface_area_frac", default=0.1, desc="Fraction of wing area covered by control surfaces and flaps"
        )
        self.options.declare(
            "fuselage_width_frac", default=0.5, desc="Fuselage width at tail intersection divided by fuselage diameter"
        )
        self.options.declare("K_uht", default=1.0, desc="Scaling for all moving stabilizer")
        self.options.declare(
            "elevator_area_frac", default=0.2, desc="Fraction of horizontal stabilizer covered by elevators"
        )
        self.options.declare("T_tail", default=False, types=bool, desc="True if T-tail, False otherwise")
        self.options.declare("K_door", default=1, desc="Number of doors parameter")
        self.options.declare("K_lg", default=1, desc="Fuselage-mounted landing gear parameter")
        self.options.declare("kneeling_main_gear_parameter", default=1.0, desc="Kneeling main landing gear parameter")
        self.options.declare("kneeling_nose_gear_parameter", default=1.0, desc="Kneeling nose landing gear parameter")
        self.options.declare("K_lav", default=0.7, desc="Lavatory weight coefficient")
        self.options.declare("K_buf", default=4.0, desc="Food weight coefficient")
        self.options.declare("coeff_fc", default=1.2 * 0.64, desc="Coefficient on flight control system weight")
        self.options.declare("coeff_avionics", default=0.5, desc="Coefficient on avionics weight")
        self.options.declare("cabin_length_frac", default=0.75, desc="Cabin length / fuselage length")
        self.options.declare("APU_weight_frac", default=0.0085, desc="APU weight / MTOW")

    def setup(self):
        n_ult = self.options["n_ult"]

        # ==============================================================================
        # Lifting surface weights (wing and stabilizers)
        # ==============================================================================
        # -------------- Wing --------------
        self.add_subsystem(
            "wing",
            WingWeight_JetTransport(n_ult=n_ult, control_surface_area_frac=self.options["control_surface_area_frac"]),
            promotes_inputs=[
                "ac|weights|MTOW",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|taper",
                "ac|geom|wing|toverc",
            ],
            promotes_outputs=["W_wing"],
        )

        # -------------- Horizontal stabilizer --------------
        hstab = om.Group()
        hstab.add_subsystem(
            "hstab_const",
            HstabConst_JetTransport(fuselage_width_frac=self.options["fuselage_width_frac"]),
            promotes_inputs=[
                "ac|geom|hstab|S_ref",
                "ac|geom|hstab|AR",
                "ac|geom|fuselage|height",
            ],
        )
        hstab.add_subsystem(
            "hstab_calc",
            HstabWeight_JetTransport(
                n_ult=n_ult,
                K_uht=self.options["K_uht"],
                elevator_area_frac=self.options["elevator_area_frac"],
            ),
            promotes_inputs=[
                "ac|weights|MTOW",
                "ac|geom|hstab|S_ref",
                "ac|geom|hstab|AR",
                "ac|geom|hstab|c4sweep",
                "ac|geom|hstab|c4_to_wing_c4",
            ],
            promotes_outputs=["W_hstab"],
        )
        hstab.connect("hstab_const.HstabConst", "hstab_calc.HstabConst")
        self.add_subsystem("hstab", hstab, promotes_inputs=["*"], promotes_outputs=["W_hstab"])

        # -------------- Vertical stabilizer --------------
        self.add_subsystem(
            "vstab",
            VstabWeight_JetTransport(
                n_ult=n_ult,
                T_tail=self.options["T_tail"],
            ),
            promotes_inputs=[
                "ac|weights|MTOW",
                "ac|geom|vstab|S_ref",
                "ac|geom|vstab|AR",
                "ac|geom|vstab|c4sweep",
                "ac|geom|vstab|toverc",
                "ac|geom|vstab|c4_to_wing_c4",
            ],
            promotes_outputs=["W_vstab"],
        )

        # ==============================================================================
        # Fuselage
        # ==============================================================================
        fuselage = om.Group()
        fuselage.add_subsystem(
            "K_ws_term",
            FuselageKws_JetTransport(),
            promotes_inputs=[
                "ac|geom|wing|taper",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                "ac|geom|wing|c4sweep",
                "ac|geom|fuselage|length",
            ],
        )
        fuselage.add_subsystem(
            "fuselage_calc",
            FuselageWeight_JetTransport(n_ult=n_ult, K_door=self.options["K_door"], K_lg=self.options["K_lg"]),
            promotes_inputs=[
                "ac|weights|MTOW",
                "ac|geom|fuselage|length",
                "ac|geom|fuselage|S_wet",
                "ac|geom|fuselage|height",
            ],
            promotes_outputs=["W_fuselage"],
        )
        fuselage.connect("K_ws_term.K_ws", "fuselage_calc.K_ws")
        self.add_subsystem("fuselage", fuselage, promotes_inputs=["*"], promotes_outputs=["W_fuselage"])

        # ==============================================================================
        # Landing gear
        # ==============================================================================
        # -------------- Main gear --------------
        self.add_subsystem(
            "main_gear",
            MainLandingGearWeight_JetTransport(
                n_land_ult=self.options["n_land_ult"],
                kneeling_gear_parameter=self.options["kneeling_main_gear_parameter"],
            ),
            promotes_inputs=[
                "ac|weights|MLW",
                "ac|geom|maingear|length",
                "ac|geom|maingear|num_wheels",
                "ac|geom|maingear|num_shock_struts",
                "ac|aero|Vstall_land",
            ],
            promotes_outputs=["W_mlg"],
        )

        # -------------- Nose gear --------------
        self.add_subsystem(
            "nose_gear",
            NoseLandingGearWeight_JetTransport(
                n_land_ult=self.options["n_land_ult"],
                kneeling_gear_parameter=self.options["kneeling_nose_gear_parameter"],
            ),
            promotes_inputs=[
                "ac|weights|MLW",
                "ac|geom|nosegear|length",
                "ac|geom|nosegear|num_wheels",
            ],
            promotes_outputs=["W_nlg"],
        )

        # ==============================================================================
        # Propulsion system
        # ==============================================================================
        # -------------- Dry engine --------------
        # Engine weight computes a single engine, so it must be multiplied by the number of engines
        self.add_subsystem(
            "single_engine",
            EngineWeight_JetTransport(),
            promotes_inputs=["ac|propulsion|engine|rating"],
        )
        self.add_subsystem(
            "engine_multiplier",
            ElementMultiplyDivideComp(
                output_name="W_engines", input_names=["W_engine", "ac|propulsion|num_engines"], input_units=["lb", None]
            ),
            promotes_inputs=["ac|propulsion|num_engines"],
            promotes_outputs=["W_engines"],
        )
        self.connect("single_engine.W_engine", "engine_multiplier.W_engine")

        # -------------- Engine systems --------------
        self.add_subsystem(
            "engine_systems",
            EngineSystemsWeight_JetTransport(),
            promotes_inputs=[
                "ac|propulsion|engine|rating",
                "ac|propulsion|num_engines",
                "ac|aero|Mach_max",
                "ac|weights|W_fuel_max",
            ],
            promotes_outputs=[
                "W_thrust_rev",
                "W_eng_control",
                "W_fuelsystem",
                "W_eng_start",
            ],
        )
        self.connect("single_engine.W_engine", "engine_systems.W_engine")

        # -------------- Nacelle --------------
        self.add_subsystem(
            "nacelles",
            NacelleWeight_JetTransport(),
            promotes_inputs=["ac|propulsion|engine|rating", "ac|propulsion|num_engines"],
            promotes_outputs=["W_nacelle"],
        )

        # ==============================================================================
        # Furnishings for passengers
        # ==============================================================================
        self.add_subsystem(
            "furnishings",
            FurnishingWeight_JetTransport(K_lav=self.options["K_lav"], K_buf=self.options["K_buf"]),
            promotes_inputs=[
                "ac|num_passengers_max",
                "ac|num_flight_deck_crew",
                "ac|num_cabin_crew",
                "ac|cabin_pressure",
                "ac|weights|MTOW",
            ],
            promotes_outputs=["W_furnishings"],
        )

        # ==============================================================================
        # Other equipment
        # ==============================================================================
        # Estimate the volume of the passenger cabin by treating it as a cylinder with
        # the fuselage diameter and length of fuselage length times a constant factor
        # (by default 0.75)
        self.add_subsystem(
            "cabin_volume",
            om.ExecComp(
                "V_pressurized = pi * 0.25 * fus_height * fus_height * fus_length * cabin_frac",
                V_pressurized={"units": "ft**3", "val": 1},
                fus_height={"units": "ft", "val": 1},
                fus_length={"units": "ft", "val": 1},
                cabin_frac={"val": self.options["cabin_length_frac"], "constant": True},
            ),
            promotes_inputs=[("fus_height", "ac|geom|fuselage|height"), ("fus_length", "ac|geom|fuselage|length")],
        )
        self.add_subsystem(
            "equipment",
            EquipmentWeight_JetTransport(
                coeff_fc=self.options["coeff_fc"],
                coeff_avionics=self.options["coeff_avionics"],
                APU_weight_frac=self.options["APU_weight_frac"],
            ),
            promotes_inputs=[
                "ac|weights|MTOW",
                "ac|num_passengers_max",
                "ac|num_cabin_crew",
                "ac|num_flight_deck_crew",
                "ac|propulsion|num_engines",
                "W_fuelsystem",
            ],
            promotes_outputs=[
                "W_flight_controls",
                "W_avionics",
                "W_electrical",
                "W_ac_pressurize_antiice",
                "W_oxygen",
                "W_APU",
            ],
        )
        self.connect("cabin_volume.V_pressurized", "equipment.ac|geom|V_pressurized")

        # ==============================================================================
        # Multiply structural weights by fudge factor
        # ==============================================================================
        structure_weight_outputs = ["W_wing", "W_hstab", "W_vstab", "W_fuselage", "W_mlg", "W_nlg", "W_nacelle"]
        scaling_factors = [self.options["structural_fudge"]] * len(structure_weight_outputs)
        scaling_factors[0] *= self.options["wing_weight_multiplier"]
        self.add_subsystem(
            "structural_adjustment",
            AddSubtractComp(
                output_name="W_structure",
                input_names=structure_weight_outputs,
                scaling_factors=scaling_factors,
                units="lb",
            ),
            promotes_inputs=["*"],
            promotes_outputs=["W_structure"],
        )

        # ==============================================================================
        # Sum all weights to compute total operating empty weight
        # ==============================================================================
        final_weight_components = [
            "W_structure",
            "W_engines",
            "W_thrust_rev",
            "W_eng_control",
            "W_fuelsystem",
            "W_eng_start",
            "W_furnishings",
            "W_flight_controls",
            "W_avionics",
            "W_electrical",
            "W_ac_pressurize_antiice",
            "W_oxygen",
            "W_APU",
        ]
        self.add_subsystem(
            "sum_weights",
            AddSubtractComp(
                output_name="OEW",
                input_names=final_weight_components,
                scaling_factors=[self.options["total_fudge"]] * len(final_weight_components),
                units="lb",
            ),
            promotes_inputs=["*"],
            promotes_outputs=["OEW"],
        )


class WingWeight_JetTransport(om.ExplicitComponent):
    """
    Transport aircraft wing weight estimated from Raymer (eqn 15.25 in 1992 edition).

    Inputs
    ------
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)
    ac|geom|wing|S_ref : float
        Wing planform reference area (scalar, sq ft)
    ac|geom|wing|AR : float
        Wing aspect ratio (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Wing sweep at 25% mean aerodynamic chord (scalar, radians)
    ac|geom|wing|taper : float
        Wing taper ratio (scalar, dimensionless)
    ac|geom|wing|toverc : float
        Wing root thickness-to-chord ratio (scalar, dimensionless)

    Outputs
    -------
    W_wing : float
        Estimated wing weight (scalar, lb)

    Options
    -------
    n_ult : float
        Ultimate load factor, 1.5 x limit load factor, by default 1.5 x 2.5 (scalar, dimensionless)
    control_surface_area_frac : float
        Fraction of the total wing area covered by control surfaces and flaps, by default 0.1 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("n_ult", default=2.5 * 1.5, desc="Ultimate load factor (dimensionless)")
        self.options.declare(
            "control_surface_area_frac", default=0.1, desc="Fraction of wing area covered by control surfaces and flaps"
        )

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input("ac|geom|wing|c4sweep", units="rad", desc="Quarter-chord sweep angle")
        self.add_input("ac|geom|wing|taper", desc="Wing taper ratio")
        self.add_input("ac|geom|wing|toverc", desc="Wing max thickness to chord ratio")

        self.add_output("W_wing", units="lb", desc="Wing weight")
        self.declare_partials(["W_wing"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        W_wing_Raymer = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * ((self.options["control_surface_area_frac"] * inputs["ac|geom|wing|S_ref"]) ** 0.1)
        )

        outputs["W_wing"] = W_wing_Raymer

    def compute_partials(self, inputs, J):  # TO DO
        n_ult = self.options["n_ult"]
        J["W_wing", "ac|weights|MTOW"] = (
            (0.0051 * 0.557)
            * (inputs["ac|weights|MTOW"] ** (0.557 - 1))
            * n_ult**0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (self.options["control_surface_area_frac"] * inputs["ac|geom|wing|S_ref"]) ** 0.1
        )
        J["W_wing", "ac|geom|wing|S_ref"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (0.649 + 0.1)
            * (inputs["ac|geom|wing|S_ref"]) ** (0.649 + 0.1 - 1)
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (self.options["control_surface_area_frac"] ** 0.1)
        )
        J["W_wing", "ac|geom|wing|AR"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * 0.5
            * (inputs["ac|geom|wing|AR"]) ** (0.5 - 1)
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (self.options["control_surface_area_frac"] ** 0.1)
            * (inputs["ac|geom|wing|S_ref"]) ** 0.1
        )
        J["W_wing", "ac|geom|wing|c4sweep"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** (0.649 + 0.1)
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * -1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -2
            * (-1 * np.sin(inputs["ac|geom|wing|c4sweep"]))
            * (self.options["control_surface_area_frac"] ** 0.1)
        )
        J["W_wing", "ac|geom|wing|taper"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * 0.1
            * (1 + inputs["ac|geom|wing|taper"]) ** (0.1 - 1)
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (self.options["control_surface_area_frac"] ** 0.1)
            * (inputs["ac|geom|wing|S_ref"]) ** 0.1
        )
        J["W_wing", "ac|geom|wing|toverc"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * -0.4
            * (inputs["ac|geom|wing|toverc"]) ** (-0.4 - 1)
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (self.options["control_surface_area_frac"] ** 0.1)
            * (inputs["ac|geom|wing|S_ref"]) ** 0.1
        )


class HstabConst_JetTransport(om.ExplicitComponent):
    """
    The 1 + Fw/Bh term in Raymer's horizontal tail weight estimate (in eqn 15.26 in 1992 edition).

    Inputs
    ------
    ac|geom|hstab|S_ref : float
        Horizontal stabilizer reference area (scalar, sq ft)
    ac|geom|hstab|AR : float
        Horizontal stabilizer aspect ratio (scalar, dimensionless)
    ac|geom|fuselage|height : float
        Fuselage height (scalar, ft)

    Outputs
    -------
    HstasbConst : float
        The 1 + Fw/Bh term in the weight estimate (scalar, dimensionless)

    Options
    -------
    fuselage_width_frac : float
        Fuselage width at horizontal tail intersection divided by fuselage diameter, by default 0.5 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare(
            "fuselage_width_frac", default=0.5, desc="Fuselage width at tail intersection divided by fuselage diameter"
        )

    def setup(self):
        self.add_input("ac|geom|hstab|S_ref", units="ft**2", desc="Horizontal stabizer reference area")
        self.add_input("ac|geom|hstab|AR", desc="Horizontal stabilizer aspect ratio")
        self.add_input("ac|geom|fuselage|height", units="ft", desc="Fuselage height")
        self.add_output("HstabConst")
        self.declare_partials(["HstabConst"], ["*"])

    def compute(self, inputs, outputs):
        Fw = inputs["ac|geom|fuselage|height"] * self.options["fuselage_width_frac"]
        Bh = np.sqrt(inputs["ac|geom|hstab|S_ref"] * inputs["ac|geom|hstab|AR"])
        outputs["HstabConst"] = 1 + Fw / Bh

    def compute_partials(self, inputs, J):
        Fw = inputs["ac|geom|fuselage|height"] * self.options["fuselage_width_frac"]
        Bh = np.sqrt(inputs["ac|geom|hstab|S_ref"] * inputs["ac|geom|hstab|AR"])
        J["HstabConst", "ac|geom|hstab|S_ref"] = -Fw / Bh**2 * (0.5 / Bh) * inputs["ac|geom|hstab|AR"]
        J["HstabConst", "ac|geom|hstab|AR"] = -Fw / Bh**2 * (0.5 / Bh) * inputs["ac|geom|hstab|S_ref"]
        J["HstabConst", "ac|geom|fuselage|height"] = 1 / Bh * self.options["fuselage_width_frac"]


class HstabWeight_JetTransport(om.ExplicitComponent):
    """
    Horizontal stabilizer weight estimation from Raymer (eqn 15.26 in 1992 edition).
    This component makes the additional assumption that the distance between the wing
    quarter chord and horizontal stabilizer quarter chord is a constant fraction of
    the fuselage length (by default half).

    Inputs
    ------
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)
    ac|geom|hstab|S_ref : float
        Horizontal stabilizer wing area (scalar, sq ft)
    ac|geom|hstab|AR : float
        Horizontal stabilizer aspect ratio (scalar, dimensionless)
    ac|geom|hstab|c4sweep : float
        Horizontal stabilizer sweep at 25% mean aerodynamic chord (scalar, radians)
    ac|geom|fuselage|length : float
        Fuselage length, used to compute distance between quarter chord of wing and horizontal stabilizer (scalar, ft)
    ac|geom|hstab|c4_to_wing_c4 : float
        Distance from the horizontal stabilizer's quarter chord (of the MAC) to the wing's quarter chord (scalar, ft)
    HstasbConst : float
        The 1 + Fw/Bh term in the weight estimate (scalar, dimensionless)

    Outputs
    -------
    W_hstab : float
        Weight of the horizontal stabilizer (scalar, lb)

    Options
    -------
    n_ult : float
        Ultimate load factor, 1.5 x limit load factor, by default 1.5 x 2.5 (scalar, dimensionless)
    K_uht : float
        Correction for all-moving tail; set to 1.143 for all-moving tail or 1.0 otherwise, by default 1.0 (scalar, dimensionless)
    elevator_area_frac : float
        Fraction of horizontal stabilizer area covered by elevators, by default 0.2 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("n_ult", default=2.5 * 1.5, desc="Ultimate load factor (dimensionless)")
        self.options.declare("K_uht", default=1.0, desc="Scaling for all moving stabilizer")
        self.options.declare(
            "elevator_area_frac", default=0.2, desc="Fraction of horizontal stabilizer covered by elevators"
        )

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|hstab|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|hstab|AR", desc="Wing aspect ratio")
        self.add_input("ac|geom|hstab|c4sweep", units="rad", desc="Quarter-chord sweep angle")
        self.add_input("ac|geom|hstab|c4_to_wing_c4", units="ft", desc="Distance from wing to tail quarter chord")
        self.add_input("HstabConst", desc="1 + Fw/Bh term in equation")

        self.add_output("W_hstab", units="lb", desc="Hstab weight")
        self.declare_partials(["W_hstab"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        K_uht = self.options["K_uht"]
        Se_Sht = self.options["elevator_area_frac"]
        c4_wing_c4_tail = inputs["ac|geom|hstab|c4_to_wing_c4"]

        outputs["W_hstab"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult**0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 * c4_wing_c4_tail) ** 0.704
            * c4_wing_c4_tail**-1
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + Se_Sht) ** 0.1
        )

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        K_uht = self.options["K_uht"]
        Se_Sht = self.options["elevator_area_frac"]
        c4_wing_c4_tail = inputs["ac|geom|hstab|c4_to_wing_c4"]

        J["W_hstab", "ac|weights|MTOW"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * 0.639
            * inputs["ac|weights|MTOW"] ** (0.639 - 1)
            * n_ult**0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 * c4_wing_c4_tail) ** 0.704
            * c4_wing_c4_tail**-1
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + Se_Sht) ** 0.1
        )

        J["W_hstab", "ac|geom|hstab|S_ref"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult**0.10
            * 0.75
            * inputs["ac|geom|hstab|S_ref"] ** (0.75 - 1)
            * (0.3 * c4_wing_c4_tail) ** 0.704
            * c4_wing_c4_tail**-1
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + Se_Sht) ** 0.1
        )

        J["W_hstab", "ac|geom|hstab|AR"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult**0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 * c4_wing_c4_tail) ** 0.704
            * c4_wing_c4_tail**-1
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * 0.166
            * inputs["ac|geom|hstab|AR"] ** (0.166 - 1)
            * (1 + Se_Sht) ** 0.1
        )

        J["W_hstab", "ac|geom|hstab|c4sweep"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult**0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 * c4_wing_c4_tail) ** 0.704
            * c4_wing_c4_tail**-1
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -2
            * np.sin(inputs["ac|geom|hstab|c4sweep"])
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + Se_Sht) ** 0.1
        )

        J["W_hstab", "ac|geom|hstab|c4_to_wing_c4"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult**0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * 0.3**0.704
            * (-0.296)
            * c4_wing_c4_tail**-1.296
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + Se_Sht) ** 0.1
        )

        J["W_hstab", "HstabConst"] = (
            0.0379
            * K_uht
            * (-0.25)
            * inputs["HstabConst"] ** (-0.25 - 1)
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult**0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 * c4_wing_c4_tail) ** 0.704
            * c4_wing_c4_tail**-1
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + Se_Sht) ** 0.1
        )


class VstabWeight_JetTransport(om.ExplicitComponent):
    """
    Vertical stabilizer weight estimate from Raymer (eqn 15.27 in 1992 edition).
    This component makes the additional assumption that the distance between the wing
    quarter chord and vertical stabilizer quarter chord is a constant fraction of
    the fuselage length (by default half).

    Inputs
    ------
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)
    ac|geom|vstab|S_ref : float
        vertical stabilizer wing area (scalar, sq ft)
    ac|geom|vstab|AR : float
        vertical stabilizer aspect ratio (scalar, dimensionless)
    ac|geom|vstab|toverc : float
        vertical stabilizer thickness-to-chord ratio (scalar, dimensionless)
    ac|geom|vstab|c4sweep : float
        vertical stabilizer sweep at 25% mean aerodynamic chord (scalar, radians)
    ac|geom|vstab|c4_to_wing_c4 : float
        Distance from the vertical stabilizer's quarter chord (of the MAC) to the wing's quarter chord (scalar, ft)

    Outputs
    -------
    W_vstab : float
        Weight of the vertical stabilizer (scalar, lb)

    Options
    -------
    n_ult : float
        Ultimate load factor, 1.5 x limit load factor, by default 1.5 x 2.5 (scalar, dimensionless)
    T_tail : bool
        True if the tail is a T-tail, False otherwise
    """

    def initialize(self):
        self.options.declare("n_ult", default=2.5 * 1.5, desc="Ultimate load factor (dimensionless)")
        self.options.declare("T_tail", default=False, types=bool, desc="True if T-tail, False otherwise")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|vstab|S_ref", units="ft**2", desc="Reference vtail area in sq ft")
        self.add_input("ac|geom|vstab|AR", desc="vtail aspect ratio")
        self.add_input("ac|geom|vstab|c4sweep", units="rad", desc="Quarter-chord sweep angle")
        self.add_input("ac|geom|vstab|c4_to_wing_c4", units="ft", desc="Distance from wing to tail quarter chord")
        self.add_input("ac|geom|vstab|toverc", desc="root t/c of v-tail, estimated same as wing")

        self.add_output("W_vstab", units="lb", desc="Vstab weight")
        self.declare_partials(["W_vstab"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        Ht_Hv = 1.0 if self.options["T_tail"] else 0.0
        c4_wing_c4_tail = inputs["ac|geom|vstab|c4_to_wing_c4"]

        outputs["W_vstab"] = (
            0.0026
            * (1 + Ht_Hv) ** 0.225
            * inputs["ac|weights|MTOW"] ** 0.556
            * n_ult**0.536
            * c4_wing_c4_tail ** (-0.5 + 0.875)
            * inputs["ac|geom|vstab|S_ref"] ** 0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -1
            * inputs["ac|geom|vstab|AR"] ** 0.35
            * inputs["ac|geom|vstab|toverc"] ** -0.5
        )

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        Ht_Hv = 1.0 if self.options["T_tail"] else 0.0
        c4_wing_c4_tail = inputs["ac|geom|vstab|c4_to_wing_c4"]

        J["W_vstab", "ac|weights|MTOW"] = (
            0.0026
            * (1 + Ht_Hv) ** 0.225
            * 0.556
            * inputs["ac|weights|MTOW"] ** (0.556 - 1)
            * n_ult**0.536
            * c4_wing_c4_tail ** (-0.5 + 0.875)
            * inputs["ac|geom|vstab|S_ref"] ** 0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -1
            * inputs["ac|geom|vstab|AR"] ** 0.35
            * inputs["ac|geom|vstab|toverc"] ** -0.5
        )
        J["W_vstab", "ac|geom|vstab|S_ref"] = (
            0.0026
            * (1 + Ht_Hv) ** 0.225
            * inputs["ac|weights|MTOW"] ** 0.556
            * n_ult**0.536
            * c4_wing_c4_tail ** (-0.5 + 0.875)
            * 0.5
            * inputs["ac|geom|vstab|S_ref"] ** -0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -1
            * inputs["ac|geom|vstab|AR"] ** 0.35
            * inputs["ac|geom|vstab|toverc"] ** -0.5
        )
        J["W_vstab", "ac|geom|vstab|AR"] = (
            0.0026
            * (1 + Ht_Hv) ** 0.225
            * inputs["ac|weights|MTOW"] ** 0.556
            * n_ult**0.536
            * c4_wing_c4_tail ** (-0.5 + 0.875)
            * inputs["ac|geom|vstab|S_ref"] ** 0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -1
            * 0.35
            * inputs["ac|geom|vstab|AR"] ** (0.35 - 1)
            * inputs["ac|geom|vstab|toverc"] ** -0.5
        )
        J["W_vstab", "ac|geom|vstab|c4sweep"] = (
            0.0026
            * (1 + Ht_Hv) ** 0.225
            * inputs["ac|weights|MTOW"] ** 0.556
            * n_ult**0.536
            * c4_wing_c4_tail ** (-0.5 + 0.875)
            * inputs["ac|geom|vstab|S_ref"] ** 0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -2
            * np.sin(inputs["ac|geom|vstab|c4sweep"])
            * inputs["ac|geom|vstab|AR"] ** 0.35
            * inputs["ac|geom|vstab|toverc"] ** -0.5
        )
        J["W_vstab", "ac|geom|vstab|c4_to_wing_c4"] = (
            0.0026
            * (1 + Ht_Hv) ** 0.225
            * inputs["ac|weights|MTOW"] ** 0.556
            * n_ult**0.536
            * (-0.5 + 0.875)
            * c4_wing_c4_tail ** (-0.5 + 0.875 - 1)
            * inputs["ac|geom|vstab|S_ref"] ** 0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -1
            * inputs["ac|geom|vstab|AR"] ** 0.35
            * inputs["ac|geom|vstab|toverc"] ** -0.5
        )
        J["W_vstab", "ac|geom|vstab|toverc"] = (
            0.0026
            * (1 + Ht_Hv) ** 0.225
            * inputs["ac|weights|MTOW"] ** 0.556
            * n_ult**0.536
            * c4_wing_c4_tail ** (-0.5 + 0.875)
            * inputs["ac|geom|vstab|S_ref"] ** 0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -1
            * inputs["ac|geom|vstab|AR"] ** 0.35
            * (-0.5)
            * inputs["ac|geom|vstab|toverc"] ** -1.5
        )


class FuselageKws_JetTransport(om.ExplicitComponent):
    """
    Compute Raymer's Kws term for the fuselage weight estimation (in eqn 15.28 in the 1992 edition).

    Inputs
    ------
    ac|geom|wing|taper : float
        Main wing taper ratio (scalar, dimensionless)
    ac|geom|wing|S_ref : float
        Main wing reference area (scalar, sq ft)
    ac|geom|wing|AR : float
        Main wing aspect ratio (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Main wing quarter chord sweep angle (scalar, radians)
    ac|geom|fuselage|length : float
        Fuselage length (scalar, ft)

    Outputs
    -------
    K_ws : float
        K_ws term in Raymer's fuselage weight approximation (scalar, dimensionless)
    """

    def setup(self):
        self.add_input("ac|geom|wing|taper", desc="Wing taper ratio")
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Wing reference area")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input("ac|geom|wing|c4sweep", units="rad", desc="Wing aspect Ratio")
        self.add_input("ac|geom|fuselage|length", units="ft", desc="Fuselage structural length")
        self.add_output("K_ws", desc="Fuselage constant Kws defined in Raymer")
        self.declare_partials(["K_ws"], ["*"])

    def compute(self, inputs, outputs):
        Kws_raymer = (
            0.75
            * (1 + 2 * inputs["ac|geom|wing|taper"])
            / (1 + inputs["ac|geom|wing|taper"])
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * inputs["ac|geom|wing|AR"] ** 0.5
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        outputs["K_ws"] = Kws_raymer

    def compute_partials(self, inputs, J):
        J["K_ws", "ac|geom|wing|taper"] = (
            0.75
            * (1 + inputs["ac|geom|wing|taper"]) ** -2
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * inputs["ac|geom|wing|AR"] ** 0.5
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        J["K_ws", "ac|geom|wing|S_ref"] = (
            0.75
            * (1 + 2 * inputs["ac|geom|wing|taper"])
            / (1 + inputs["ac|geom|wing|taper"])
            * 0.5
            * inputs["ac|geom|wing|S_ref"] ** (0.5 - 1)
            * inputs["ac|geom|wing|AR"] ** 0.5
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        J["K_ws", "ac|geom|wing|AR"] = (
            0.75
            * (1 + 2 * inputs["ac|geom|wing|taper"])
            / (1 + inputs["ac|geom|wing|taper"])
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * 0.5
            * inputs["ac|geom|wing|AR"] ** (0.5 - 1)
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        J["K_ws", "ac|geom|wing|c4sweep"] = (
            0.75
            * (1 + 2 * inputs["ac|geom|wing|taper"])
            / (1 + inputs["ac|geom|wing|taper"])
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * inputs["ac|geom|wing|AR"] ** 0.5
            * (1 / np.cos(inputs["ac|geom|wing|c4sweep"])) ** 2
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        J["K_ws", "ac|geom|fuselage|length"] = (
            0.75
            * (1 + 2 * inputs["ac|geom|wing|taper"])
            / (1 + inputs["ac|geom|wing|taper"])
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * inputs["ac|geom|wing|AR"] ** 0.5
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * -1
            * inputs["ac|geom|fuselage|length"] ** (-1 - 1)
        )


class FuselageWeight_JetTransport(om.ExplicitComponent):
    """
    Fuselage weight estimation from Raymer (eqn 15.28 in 1992 edition).

    Inputs
    ------
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)
    ac|geom|fuselage|length : float
        Fuselage structural length (scalar, ft)
    ac|geom|fuselage|S_wet : float
        Fuselage wetted area (scalar, sq ft)
    ac|geom|fuselage|height : float
        Fuselage height (scalar, ft)
    K_ws : float
        Fuselage parameter computed in FuselageKws_JetTransport (scalar, dimensionless)

    Outputs
    -------
    W_fuselage : float
        Fuselage weight (scalar, lb)

    Options
    -------
    n_ult : float
        Ultimate load factor, 1.5 x limit load factor, by default 1.5 x 2.5 (scalar, dimensionless)
    K_door : float
        Fuselage door parameter; 1 if no cargo door, 1.06 if one side cargo door, 1.12 if two side
        cargo doors, 1.12 if aft clamshell door, 1.25 if two side cargo doors, and aft clamshell door,
        by default 1 (scalar, dimensionless)
    K_lg : float
        Fuselage-mounted landing gear parameter; 1.12 if fuselage-mounted main landing gear and 1
        otherwise, by default 1 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("n_ult", default=2.5 * 1.5, desc="Ultimate load factor (dimensionless)")
        self.options.declare("K_door", default=1, desc="Number of doors parameter")
        self.options.declare("K_lg", default=1, desc="Fuselage-mounted landing gear parameter")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|fuselage|length", units="ft", desc="fuselage structural length")
        self.add_input("ac|geom|fuselage|S_wet", units="ft**2", desc="fuselage wetted area")
        self.add_input("ac|geom|fuselage|height", units="ft", desc="Fuselage height")
        self.add_input("K_ws")

        self.add_output("W_fuselage", units="lb", desc="fuselage weight")
        self.declare_partials(["W_fuselage"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        K_door = self.options["K_door"]
        K_lg = self.options["K_lg"]

        outputs["W_fuselage"] = (
            0.3280
            * K_door
            * K_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult**0.5
            * inputs["ac|geom|fuselage|length"] ** (0.25 + 0.1)
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * (1 + inputs["K_ws"]) ** 0.04
            * inputs["ac|geom|fuselage|height"] ** -0.10
        )

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        K_door = self.options["K_door"]
        K_lg = self.options["K_lg"]

        J["W_fuselage", "ac|weights|MTOW"] = (
            0.3280
            * K_door
            * K_lg
            * 0.5
            * inputs["ac|weights|MTOW"] ** -0.5
            * n_ult**0.5
            * inputs["ac|geom|fuselage|length"] ** (0.25 + 0.1)
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * (1 + inputs["K_ws"]) ** 0.04
            * inputs["ac|geom|fuselage|height"] ** -0.10
        )
        J["W_fuselage", "ac|geom|fuselage|length"] = (
            0.3280
            * K_door
            * K_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult**0.5
            * 0.35
            * inputs["ac|geom|fuselage|length"] ** (0.25 + 0.1 - 1)
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * (1 + inputs["K_ws"]) ** 0.04
            * inputs["ac|geom|fuselage|height"] ** -0.10
        )
        J["W_fuselage", "ac|geom|fuselage|S_wet"] = (
            0.3280
            * K_door
            * K_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult**0.5
            * inputs["ac|geom|fuselage|length"] ** (0.25 + 0.1)
            * 0.302
            * inputs["ac|geom|fuselage|S_wet"] ** (0.302 - 1)
            * (1 + inputs["K_ws"]) ** 0.04
            * inputs["ac|geom|fuselage|height"] ** -0.10
        )
        J["W_fuselage", "ac|geom|fuselage|height"] = (
            0.3280
            * K_door
            * K_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult**0.5
            * inputs["ac|geom|fuselage|length"] ** (0.25 + 0.1)
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * (1 + inputs["K_ws"]) ** 0.04
            * (-0.1)
            * inputs["ac|geom|fuselage|height"] ** -1.10
        )
        J["W_fuselage", "K_ws"] = (
            0.3280
            * K_door
            * K_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult**0.5
            * inputs["ac|geom|fuselage|length"] ** (0.25 + 0.1)
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * 0.04
            * (1 + inputs["K_ws"]) ** (0.04 - 1)
            * inputs["ac|geom|fuselage|height"] ** -0.10
        )


class MainLandingGearWeight_JetTransport(om.ExplicitComponent):
    """
    Main landing gear weight estimate from Raymer (eqn 15.29 in 1992 edition).

    Inputs
    ------
    ac|weights|MLW : float
        Maximum landing weight (scalar, lb)
    ac|geom|maingear|length : float
        Length of the main landing gear (scalar, inch)
    ac|geom|maingear|num_wheels : float
        Total number of main landing gear wheels (scalar, dimensionless)
    ac|geom|maingear|num_shock_struts : float
        Total number of main landing gear shock struts (scalar, dimensionless)
    ac|aero|Vstall_land : float
        Landing stall speed (scalar, knots)

    Outputs
    -------
    W_mlg : float
        Main landing gear weight (scalar, lb)

    Options
    -------
    n_land_ult : float
        Ultimate landing load factor, which is 1.5 times the gear load factor (defined
        in equation 11.11). Table 11.5 gives reasonable gear load factor values for
        different aircraft types, with commercial aircraft in the 2.7-3 range. Default
        is taken at 2.8, thus the ultimate landing load factor is 2.8 x 1.5 (scalar, dimensionless)
    kneeling_gear_parameter : float
        Set to 1.126 for kneeling gear and 1 otherwise, by default 1 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("n_land_ult", default=2.8 * 1.5, desc="ultimate landing load factor")
        self.options.declare("kneeling_gear_parameter", default=1.0, desc="Kneeling landing gear parameter")

    def setup(self):
        self.add_input("ac|geom|maingear|length", units="inch", desc="main landing gear length")
        self.add_input("ac|weights|MLW", units="lb", desc="max landing weight")
        self.add_input("ac|geom|maingear|num_wheels", desc="numer of main landing gear wheels")
        self.add_input("ac|geom|maingear|num_shock_struts", desc="numer of main landing gear shock struts")
        self.add_input("ac|aero|Vstall_land", units="kn", desc="stall speed in max landing configuration")

        self.add_output("W_mlg", units="lb", desc="Main gear weight")
        self.declare_partials(["W_mlg"], ["*"])

    def compute(self, inputs, outputs):
        n_land_ult = self.options["n_land_ult"]

        outputs["W_mlg"] = (
            0.0106
            * self.options["kneeling_gear_parameter"]
            * inputs["ac|weights|MLW"] ** 0.888
            * n_land_ult**0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * inputs["ac|geom|maingear|num_wheels"] ** 0.321
            * inputs["ac|geom|maingear|num_shock_struts"] ** -0.5
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )

    def compute_partials(self, inputs, J):
        n_land_ult = self.options["n_land_ult"]

        J["W_mlg", "ac|weights|MLW"] = (
            0.0106
            * self.options["kneeling_gear_parameter"]
            * 0.888
            * inputs["ac|weights|MLW"] ** (0.888 - 1)
            * n_land_ult**0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * inputs["ac|geom|maingear|num_wheels"] ** 0.321
            * inputs["ac|geom|maingear|num_shock_struts"] ** -0.5
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )
        J["W_mlg", "ac|geom|maingear|length"] = (
            0.0106
            * self.options["kneeling_gear_parameter"]
            * inputs["ac|weights|MLW"] ** 0.888
            * n_land_ult**0.25
            * 0.4
            * inputs["ac|geom|maingear|length"] ** (0.4 - 1)
            * inputs["ac|geom|maingear|num_wheels"] ** 0.321
            * inputs["ac|geom|maingear|num_shock_struts"] ** -0.5
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )
        J["W_mlg", "ac|geom|maingear|num_wheels"] = (
            0.0106
            * self.options["kneeling_gear_parameter"]
            * inputs["ac|weights|MLW"] ** 0.888
            * n_land_ult**0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * 0.321
            * inputs["ac|geom|maingear|num_wheels"] ** (0.321 - 1)
            * inputs["ac|geom|maingear|num_shock_struts"] ** -0.5
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )
        J["W_mlg", "ac|geom|maingear|num_shock_struts"] = (
            0.0106
            * self.options["kneeling_gear_parameter"]
            * inputs["ac|weights|MLW"] ** 0.888
            * n_land_ult**0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * inputs["ac|geom|maingear|num_wheels"] ** 0.321
            * (-0.5)
            * inputs["ac|geom|maingear|num_shock_struts"] ** -1.5
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )
        J["W_mlg", "ac|aero|Vstall_land"] = (
            0.0106
            * self.options["kneeling_gear_parameter"]
            * inputs["ac|weights|MLW"] ** 0.888
            * n_land_ult**0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * inputs["ac|geom|maingear|num_wheels"] ** 0.321
            * inputs["ac|geom|maingear|num_shock_struts"] ** -0.5
            * 0.1
            * inputs["ac|aero|Vstall_land"] ** (0.1 - 1)
        )


class NoseLandingGearWeight_JetTransport(om.ExplicitComponent):
    """
    Nose landing gear weight estimate from Raymer (eqn 15.30 in 1992 edition).

    Inputs
    ------
    ac|weights|MLW : float
        Maximum landing weight (scalar, lb)
    ac|geom|nosegear|length : float
        Length of the nose landing gear (scalar, inch)
    ac|geom|nosegear|num_wheels : float
        Total number of nose landing gear wheels (scalar, dimensionless)

    Outputs
    -------
    W_nlg : float
        Nose landing gear weight (scalar, lb)

    Options
    -------
    n_land_ult : float
        Ultimate landing load factor, which is 1.5 times the gear load factor (defined
        in equation 11.11). Table 11.5 gives reasonable gear load factor values for
        different aircraft types, with commercial aircraft in the 2.7-3 range. Default
        is taken at 2.8, thus the ultimate landing load factor is 2.8 x 1.5 (scalar, dimensionless)
    kneeling_gear_parameter : float
        Set to 1.15 for kneeling gear and 1 otherwise, by default 1 (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("n_land_ult", default=2.8 * 1.5, desc="ultimate landing load factor")
        self.options.declare("kneeling_gear_parameter", default=1.0, desc="Kneeling landing gear parameter")

    def setup(self):
        self.add_input("ac|geom|nosegear|length", units="inch", desc="nose landing gear length")
        self.add_input("ac|weights|MLW", units="lb", desc="max landing weight")
        self.add_input("ac|geom|nosegear|num_wheels", desc="numer of nose landing gear wheels")

        self.add_output("W_nlg", units="lb", desc="nosegear weight")
        self.declare_partials(["W_nlg"], ["*"])

    def compute(self, inputs, outputs):
        n_land_ult = self.options["n_land_ult"]
        outputs["W_nlg"] = (
            0.032
            * self.options["kneeling_gear_parameter"]
            * inputs["ac|weights|MLW"] ** 0.646
            * n_land_ult**0.2
            * inputs["ac|geom|nosegear|length"] ** 0.5
            * inputs["ac|geom|nosegear|num_wheels"] ** 0.45
        )

    def compute_partials(self, inputs, J):
        n_land_ult = self.options["n_land_ult"]
        J["W_nlg", "ac|weights|MLW"] = (
            0.032
            * self.options["kneeling_gear_parameter"]
            * 0.646
            * inputs["ac|weights|MLW"] ** (0.646 - 1)
            * n_land_ult**0.2
            * inputs["ac|geom|nosegear|length"] ** 0.5
            * inputs["ac|geom|nosegear|num_wheels"] ** 0.45
        )
        J["W_nlg", "ac|geom|nosegear|length"] = (
            0.032
            * self.options["kneeling_gear_parameter"]
            * inputs["ac|weights|MLW"] ** 0.646
            * n_land_ult**0.2
            * 0.5
            * inputs["ac|geom|nosegear|length"] ** -0.5
            * inputs["ac|geom|nosegear|num_wheels"] ** 0.45
        )
        J["W_nlg", "ac|geom|nosegear|num_wheels"] = (
            0.032
            * self.options["kneeling_gear_parameter"]
            * inputs["ac|weights|MLW"] ** 0.646
            * n_land_ult**0.2
            * inputs["ac|geom|nosegear|length"] ** 0.5
            * 0.45
            * inputs["ac|geom|nosegear|num_wheels"] ** (0.45 - 1)
        )


class EngineWeight_JetTransport(om.ExplicitComponent):
    """
    Turbofan weight as estimated by the FLOPS weight estimation method (https://ntrs.nasa.gov/citations/20170005851).
    This approach adopts equation 76's transport and HWB weight estimation method. The computed engine weight
    is per engine (must be multiplied by number of engines to get total engine weight).

    Inputs
    ------
    ac|propulsion|engine|rating : float
        Rated thrust of each engine (scalar, lbf)

    Outputs
    -------
    W_engine : float
        Engine weight (scalar, lb)
    """

    def setup(self):
        self.add_input("ac|propulsion|engine|rating", units="lbf", desc="Rated thrust per engine")
        self.add_output("W_engine", units="lb")
        self.declare_partials("W_engine", "ac|propulsion|engine|rating", val=1 / 5.5)

    def compute(self, inputs, outputs):
        outputs["W_engine"] = inputs["ac|propulsion|engine|rating"] / 5.5


class EngineSystemsWeight_JetTransport(om.ExplicitComponent):
    """
    Engine system weight as estimated by the FLOPS weight estimation method
    (https://ntrs.nasa.gov/citations/20170005851). The computed weight is for
    all engines (does not need to be multiplied by number of engines). The
    equations are from sections 5.3.3 to 5.3.5 of the linked paper. This
    assumes that all engines have thrust reversers and there are no
    center-mounted engines.

    Roskam is used to estimate the engine starting system weight, assuming
    a pneumatic starting system and one or two get engines (eqn 6.27, Part V, 1989)

    Inputs
    ------
    ac|propulsion|engine|rating : float
        Rated thrust of each engine (scalar, lbf)
    ac|propulsion|num_engines : float
        Number of engines (scalar, dimensionless)
    ac|aero|Mach_max : float
        Maximum aircraft Mach number (scalar, dimensionless)
    ac|weights|W_fuel_max : float
        Maximum fuel weight (scalar, lb)
    W_engine : float
        Engine weight (scalar, lb)

    Outputs
    -------
    W_thrust_rev : float
        Total thrust reverser weight (scalar, lb)
    W_eng_control : float
        Total engine control weight (scalar, lb)
    W_fuelsystem : float
        Total fuel system weight including tanks and plumbing (scalar, lb)
    W_eng_start : float
        Total engine starter weight (scalar, lb)
    """

    def setup(self):
        self.add_input("ac|propulsion|engine|rating", units="lbf", desc="rated thrust per engine")
        self.add_input("ac|propulsion|num_engines", desc="number of engines")
        self.add_input("W_engine", units="lb", desc="engine weight")
        self.add_input("ac|aero|Mach_max", desc="maximum aircraft Mach number")
        self.add_input("ac|weights|W_fuel_max", units="lb", desc="maximum fuel weight")

        self.add_output("W_thrust_rev", units="lb")
        self.add_output("W_eng_control", units="lb")
        self.add_output("W_fuelsystem", units="lb")
        self.add_output("W_eng_start", units="lb")

        self.declare_partials("W_thrust_rev", ["ac|propulsion|engine|rating", "ac|propulsion|num_engines"])
        self.declare_partials("W_eng_control", ["ac|propulsion|engine|rating", "ac|propulsion|num_engines"])
        self.declare_partials(
            "W_fuelsystem", ["ac|weights|W_fuel_max", "ac|propulsion|num_engines", "ac|aero|Mach_max"]
        )
        self.declare_partials("W_eng_start", ["W_engine", "ac|propulsion|num_engines"])

    def compute(self, inputs, outputs):
        N_eng = inputs["ac|propulsion|num_engines"]
        T_rated = inputs["ac|propulsion|engine|rating"]
        M_max = inputs["ac|aero|Mach_max"]

        outputs["W_thrust_rev"] = 0.034 * T_rated * N_eng
        outputs["W_eng_control"] = 0.26 * N_eng * T_rated**0.5
        outputs["W_fuelsystem"] = 1.07 * inputs["ac|weights|W_fuel_max"] ** 0.58 * N_eng**0.43 * M_max**0.34

        # Roskam 1989, Part V, Equation 6.27
        outputs["W_eng_start"] = 9.33 * (inputs["W_engine"] / 1e3) ** 1.078 * N_eng

    def compute_partials(self, inputs, J):
        N_eng = inputs["ac|propulsion|num_engines"]
        T_rated = inputs["ac|propulsion|engine|rating"]
        M_max = inputs["ac|aero|Mach_max"]

        J["W_thrust_rev", "ac|propulsion|engine|rating"] = 0.034 * N_eng
        J["W_thrust_rev", "ac|propulsion|num_engines"] = 0.034 * T_rated
        J["W_eng_control", "ac|propulsion|engine|rating"] = 0.26 * N_eng * 0.5 / T_rated**0.5
        J["W_eng_control", "ac|propulsion|num_engines"] = 0.26 * T_rated**0.5
        J["W_fuelsystem", "ac|weights|W_fuel_max"] = (
            1.07 * 0.58 * inputs["ac|weights|W_fuel_max"] ** (0.58 - 1) * N_eng**0.43 * M_max**0.34
        )
        J["W_fuelsystem", "ac|propulsion|num_engines"] = (
            1.07 * inputs["ac|weights|W_fuel_max"] ** 0.58 * 0.43 * N_eng ** (0.43 - 1) * M_max**0.34
        )
        J["W_fuelsystem", "ac|aero|Mach_max"] = (
            1.07 * inputs["ac|weights|W_fuel_max"] ** 0.58 * N_eng**0.43 * 0.34 * M_max ** (0.34 - 1)
        )
        J["W_eng_start", "W_engine"] = 9.33 * 1.078 * inputs["W_engine"] ** 0.078 / 1e3**1.078 * N_eng
        J["W_eng_start", "ac|propulsion|num_engines"] = 9.33 * (inputs["W_engine"] / 1e3) ** 1.078


class NacelleWeight_JetTransport(om.ExplicitComponent):
    """
    Nacelle weight estimate from Roskam (eqn 5.37, Chapter 5, Part V, 1989).

    Inputs
    ------
    ac|propulsion|engine|rating : float
        Rated thrust of each engine (scalar, lbf)
    ac|propulsion|num_engines : float
        Number of engines (scalar, dimensionless)

    Outputs
    -------
    W_nacelle : float
        Nacelle weight (scalar, lb)
    """

    def setup(self):
        self.add_input("ac|propulsion|engine|rating", units="lbf", desc="rated thrust per engine")
        self.add_input("ac|propulsion|num_engines", desc="number of engines")
        self.add_output("W_nacelle", units="lb", desc="nacelle weight")

        self.declare_partials("W_nacelle", ["*"])

    def compute(self, inputs, outputs):
        outputs["W_nacelle"] = 0.065 * inputs["ac|propulsion|engine|rating"] * inputs["ac|propulsion|num_engines"]

    def compute_partials(self, inputs, J):
        J["W_nacelle", "ac|propulsion|engine|rating"] = 0.065 * inputs["ac|propulsion|num_engines"]
        J["W_nacelle", "ac|propulsion|num_engines"] = 0.065 * inputs["ac|propulsion|engine|rating"]


class FurnishingWeight_JetTransport(om.ExplicitComponent):
    """
    Weight estimate of seats, insulation, trim panels, sound proofing, instrument panels, control stands,
    lighting, wiring, galleys, lavatories, overhead luggage containers, escape provisions, and fire fighting
    equipment. Estimated using the General Dynamics method in Roskam (eqn 7.44, Chapter 7, Part V, 1989).

    Inputs
    ------
    ac|num_passengers_max : float
        Maximum number of passengers (scalar, dimensionless)
    ac|num_flight_deck_crew : float
        Number of flight crew members (scalar, dimensionless)
    ac|num_cabin_crew : float
        Number of flight attendants (scalar, dimensionless)
    ac|cabin_pressure : float
        Cabin pressure (scalar, psi)
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)

    Outputs
    -------
    W_furnishings : float
        Weight estimate of seats, galleys, lavatories, and other furnishings (scalar, lb)

    Options
    -------
    K_lav : float
        Lavatory coefficient; 0.31 for short ranges and 1.11 for long ranges, by default 0.7
    K_buf : float
        Food provisions coefficient; 1.02 for short range and 5.68 for very long range, by default 4
    """

    def initialize(self):
        self.options.declare("K_lav", default=0.7, desc="Lavatory weight coefficient")
        self.options.declare("K_buf", default=4.0, desc="Food weight coefficient")

    def setup(self):
        self.add_input("ac|num_passengers_max")
        self.add_input("ac|num_flight_deck_crew")
        self.add_input("ac|num_cabin_crew")
        self.add_input("ac|cabin_pressure", units="psi")
        self.add_input("ac|weights|MTOW", units="lb")
        self.add_output("W_furnishings", units="lb")

        self.declare_partials("W_furnishings", ["ac|num_passengers_max", "ac|cabin_pressure"])
        self.declare_partials("W_furnishings", "ac|num_flight_deck_crew", val=55.0)
        self.declare_partials("W_furnishings", "ac|num_cabin_crew", val=15.0)
        self.declare_partials("W_furnishings", "ac|weights|MTOW", val=0.771 / 1e3)

    def compute(self, inputs, outputs):
        n_pax = inputs["ac|num_passengers_max"]
        outputs["W_furnishings"] = (
            55 * inputs["ac|num_flight_deck_crew"]  # flight deck seats
            + 32 * n_pax  # passenger seats
            + 15 * inputs["ac|num_cabin_crew"]  # cabin crew seats
            + self.options["K_lav"] * n_pax**1.33  # lavatories and water
            + self.options["K_buf"] * n_pax**1.12  # food provisions
            + 109 * (n_pax * (1 + inputs["ac|cabin_pressure"]) / 100) ** 0.505  # cabin windows
            + 0.771 * (inputs["ac|weights|MTOW"] / 1e3)  # misc
        )

    def compute_partials(self, inputs, J):
        n_pax = inputs["ac|num_passengers_max"]
        J["W_furnishings", "ac|num_passengers_max"] = (
            32
            + 1.33 * self.options["K_lav"] * n_pax**0.33  # lavatories and water
            + 1.12 * self.options["K_buf"] * n_pax**0.12  # food provisions
            + 109 * 0.505 * n_pax ** (0.505 - 1) * ((1 + inputs["ac|cabin_pressure"]) / 100) ** 0.505  # cabin windows
        )
        J["W_furnishings", "ac|cabin_pressure"] = (
            109 * 0.505 * (n_pax / 100) ** 0.505 * (1 + inputs["ac|cabin_pressure"]) ** (0.505 - 1)
        )


class EquipmentWeight_JetTransport(om.ExplicitComponent):
    """
    Weight estimate of the flight control system, electrical system, avionics, air conditioning,
    pressurization system, anti-icing system, oxygen system, and APU. The estimates are all from
    Roskam 1989 Part V.

    Inputs
    ------
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)
    ac|num_passengers_max : float
        Maximum number of passengers (scalar, dimensionless)
    ac|num_cabin_crew : float
        Number of flight attendants (scalar, dimensionless)
    ac|num_flight_deck_crew : float
        Number of flight crew members; the Roskam equation uses number of pilots, but this is
        the same value for modern aircraft (scalar, dimensionless)
    ac|propulsion|num_engines : float
        Number of engines (scalar, dimensionless)
    ac|geom|V_pressurized : float
        Pressurized cabin volume (scalar, cubic ft)
    W_fuelsystem : float
        Fuel system weight (scalar, lb)

    Outputs
    -------
    W_flight_controls : float
        Flight control system weight (scalar, lb)
    W_avionics : float
        Intrumentation, avionics, and electronics weight (scalar, lb)
    W_electrical : float
        Electrical system weight (scalar, lb)
    W_ac_pressurize_antiice : float
        Air conditioning, pressurization, and anti-icing system weight (scalar, lb)
    W_oxygen : float
        Oxygen system weight (scalar, lb)
    W_APU : float
        Auxiliary power unit weight (scalar, lb)

    Options
    -------
    coeff_fc : float
        K_fc in Roskam times any additional coefficient. The book says take K_fc as 0.44 for un-powered
        flight controls and 0.64 for powered flight controls. Multiply this coefficient by 1.2 if leading
        edge devices are employed. If lift dumpers are employed, use a factor of 1.15. By default 1.2 * 0.64.
    coeff_avionics : float
        Roskam notes that the avionics weight estimates are probably conservative for modern computer-based
        flight management and navigation systems. This coefficient is multiplied by the Roskam estimate to
        account for this. By default 0.5.
    APU_weight_frac : float
        APU weight divided by maximum takeoff weight, by deafult 0.0085.
    """

    def initialize(self):
        self.options.declare("coeff_fc", default=1.2 * 0.64, desc="Coefficient on flight control system weight")
        self.options.declare("coeff_avionics", default=0.5, desc="Coefficient on avionics weight")
        self.options.declare("APU_weight_frac", default=0.0085, desc="APU weight / MTOW")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb")
        self.add_input("ac|num_passengers_max")
        self.add_input("ac|num_cabin_crew")
        self.add_input("ac|num_flight_deck_crew")
        self.add_input("ac|propulsion|num_engines")
        self.add_input("ac|geom|V_pressurized", units="ft**3")
        self.add_input("W_fuelsystem", units="lb")

        self.add_output("W_flight_controls", units="lb")
        self.add_output("W_avionics", units="lb")
        self.add_output("W_electrical", units="lb")
        self.add_output("W_ac_pressurize_antiice", units="lb")
        self.add_output("W_oxygen", units="lb")
        self.add_output("W_APU", units="lb")

        self.declare_partials("W_flight_controls", "ac|weights|MTOW")
        self.declare_partials("W_avionics", ["ac|num_flight_deck_crew", "ac|weights|MTOW", "ac|propulsion|num_engines"])
        self.declare_partials(
            "W_electrical", ["W_fuelsystem", "ac|num_flight_deck_crew", "ac|weights|MTOW", "ac|propulsion|num_engines"]
        )
        self.declare_partials(
            "W_ac_pressurize_antiice",
            [
                "ac|num_flight_deck_crew",
                "ac|num_cabin_crew",
                "ac|num_passengers_max",
                "ac|geom|V_pressurized",
            ],
        )
        self.declare_partials("W_oxygen", ["ac|num_flight_deck_crew", "ac|num_cabin_crew", "ac|num_passengers_max"])
        self.declare_partials("W_APU", "ac|weights|MTOW", val=self.options["APU_weight_frac"])

    def compute(self, inputs, outputs):
        MTOW = inputs["ac|weights|MTOW"]

        # Torenbeek method from Roskam Part V 1989 Chapter 7 Equation 7.6
        outputs["W_flight_controls"] = self.options["coeff_fc"] * MTOW ** (2 / 3)

        # General Dynamics method from Roskam Part V 1989 Chapter 7 Equation 7.23
        outputs["W_avionics"] = self.options["coeff_avionics"] * (
            inputs["ac|num_flight_deck_crew"] * (15 + 0.032e-3 * MTOW)  # flight instruments
            + inputs["ac|propulsion|num_engines"] * (5 + 0.006e-3 * MTOW)  # engine intruments
            + (0.15e-3 + 0.012) * MTOW  # other instruments
        )

        # General Dynamics method from Roskam Part V 1989 Chapter 7 Equation 7.15
        outputs["W_electrical"] = 1163 * (1e-3 * (inputs["W_fuelsystem"] + outputs["W_avionics"])) ** 0.506

        # Air conditioning, pressurization, and anti-icing systems from General Dynamics method from
        # Roskam Part V 1989 Chapter 7 Equation 7.29
        n_people = inputs["ac|num_flight_deck_crew"] + inputs["ac|num_cabin_crew"] + inputs["ac|num_passengers_max"]
        outputs["W_ac_pressurize_antiice"] = 469 * (1e-4 * inputs["ac|geom|V_pressurized"] * n_people) ** 0.419

        # General Dynamics method from Roskam Part V 1989 Chapter 7 Equation 7.35
        outputs["W_oxygen"] = 7 * n_people**0.702

        # Roskam Part V 1989 Chapter 7 Equation 7.40
        outputs["W_APU"] = self.options["APU_weight_frac"] * MTOW

    def compute_partials(self, inputs, J):
        MTOW = inputs["ac|weights|MTOW"]

        J["W_flight_controls", "ac|weights|MTOW"] = 2 / 3 * self.options["coeff_fc"] * MTOW ** (-1 / 3)

        J["W_avionics", "ac|num_flight_deck_crew"] = self.options["coeff_avionics"] * (15 + 0.032e-3 * MTOW)
        J["W_avionics", "ac|propulsion|num_engines"] = self.options["coeff_avionics"] * (5 + 0.006e-3 * MTOW)
        J["W_avionics", "ac|weights|MTOW"] = self.options["coeff_avionics"] * (
            inputs["ac|num_flight_deck_crew"] * 0.032e-3
            + inputs["ac|propulsion|num_engines"] * 0.006e-3
            + (0.15e-3 + 0.012)
        )

        W_avionics = self.options["coeff_avionics"] * (
            inputs["ac|num_flight_deck_crew"] * (15 + 0.032e-3 * MTOW)  # flight instruments
            + inputs["ac|propulsion|num_engines"] * (5 + 0.006e-3 * MTOW)  # engine intruments
            + (0.15e-3 + 0.012) * MTOW  # other instruments
        )
        J["W_electrical", "W_fuelsystem"] = (
            1163 * 0.506 * (1e-3 * (inputs["W_fuelsystem"] + W_avionics)) ** (0.506 - 1) * 1e-3
        )
        delectrical_davionics = J["W_electrical", "W_fuelsystem"]
        J["W_electrical", "ac|num_flight_deck_crew"] = (
            delectrical_davionics * J["W_avionics", "ac|num_flight_deck_crew"]
        )
        J["W_electrical", "ac|propulsion|num_engines"] = (
            delectrical_davionics * J["W_avionics", "ac|propulsion|num_engines"]
        )
        J["W_electrical", "ac|weights|MTOW"] = delectrical_davionics * J["W_avionics", "ac|weights|MTOW"]

        n_people = inputs["ac|num_flight_deck_crew"] + inputs["ac|num_cabin_crew"] + inputs["ac|num_passengers_max"]
        J["W_ac_pressurize_antiice", "ac|num_flight_deck_crew"] = (
            469
            * 0.419
            * (1e-4 * inputs["ac|geom|V_pressurized"] * n_people) ** (0.419 - 1)
            * 1e-4
            * inputs["ac|geom|V_pressurized"]
        )
        J["W_ac_pressurize_antiice", "ac|num_cabin_crew"][:] = J["W_ac_pressurize_antiice", "ac|num_flight_deck_crew"]
        J["W_ac_pressurize_antiice", "ac|num_passengers_max"][:] = J[
            "W_ac_pressurize_antiice", "ac|num_flight_deck_crew"
        ]
        J["W_ac_pressurize_antiice", "ac|geom|V_pressurized"] = (
            469 * 0.419 * (1e-4 * inputs["ac|geom|V_pressurized"] * n_people) ** (0.419 - 1) * 1e-4 * n_people
        )

        J["W_oxygen", "ac|num_flight_deck_crew"] = 7 * 0.702 * n_people ** (0.702 - 1)
        J["W_oxygen", "ac|num_cabin_crew"][:] = J["W_oxygen", "ac|num_flight_deck_crew"]
        J["W_oxygen", "ac|num_passengers_max"][:] = J["W_oxygen", "ac|num_flight_deck_crew"]
