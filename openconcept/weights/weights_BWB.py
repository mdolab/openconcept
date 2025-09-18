"""
@File    :   weights_BWB.py
@Date    :   2023/03/20
@Author  :   Eytan Adler
@Description : BWB weight estimation methods
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================
from openconcept.utilities import ElementMultiplyDivideComp, AddSubtractComp
from openconcept.weights.weights_jet_transport import (
    WingWeight_JetTransport,
    MainLandingGearWeight_JetTransport,
    NoseLandingGearWeight_JetTransport,
    EngineWeight_JetTransport,
    EngineSystemsWeight_JetTransport,
    NacelleWeight_JetTransport,
    FurnishingWeight_JetTransport,
    EquipmentWeight_JetTransport,
)


class BWBEmptyWeight(om.Group):
    """
    Estimate the empty weight of a BWB.

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
    ac|geom|centerbody|S_cabin : float
        Planform area of the pressurized centerbody cabin area (scalar, sq ft)
    ac|geom|centerbody|S_aftbody : float
        Planform area of the centerbody aft of the cabin (scalar, sq ft)
    ac|geom|centerbody|taper_aftbody : float
        Taper ratio of the ceterbody region aft of the cabin (scalar, dimensionless)
    ac|geom|wing|S_ref : float
        Outboard wing planform reference area (scalar, sq ft)
    ac|geom|wing|AR : float
        Outboard wing aspect ratio (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Outboard wing sweep at 25% mean aerodynamic chord (scalar, radians)
    ac|geom|wing|taper : float
        Outboard wing taper ratio (scalar, dimensionless)
    ac|geom|wing|toverc : float
        Outboard wing root thickness-to-chord ratio (scalar, dimensionless)
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
    ac|geom|V_pressurized : float
        Volume of the pressurized cabin (scalar, cubic ft)
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
    W_cabin : float
        Weight of the pressurized cabin region of the BWB without structural fudge factor multiplier (scalar, lb)
    W_aftbody : float
        Weight of the centerbody region aft of the pressurized cabin without structural fudge factor multiplier (scalar, lb)
    W_wing : float
        Estimated outboard wing weight without structural fudge factor multiplier (scalar, lb)
    W_mlg : float
        Main landing gear weight without structural fudge factor multiplier (scalar, lb)
    W_nlg : float
        Nose landing gear weight without structural fudge factor multiplier (scalar, lb)
    W_nacelle : float
        Weight of the nacelles (scalar, lb)
    W_structure : float
        Total structural weight = fudge factor * (W_cabin + W_aftbody + W_wing + W_mlg + W_nlg + W_nacelle) (scalar, lb)
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
        fuselage, landing gear, and nacelle weights. By default 1.0 (scalar, dimensionless)
    total_fudge : float
        Multiplier on the final operating empty weight estimate. Structural components have both the
        structural fudge and total fudge factors applied. By default 1.0 (scalar, dimensionless)
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
    APU_weight_frac : float
        Auxiliary power unit weight divided by maximum takeoff weight, by deafult 0.0085.
    """

    def initialize(self):
        self.options.declare("structural_fudge", default=1.0, desc="Fudge factor on structural weights")
        self.options.declare("total_fudge", default=1.0, desc="Fudge factor applied to the final OEW value")
        self.options.declare("wing_weight_multiplier", default=1.0, desc="Multiplier on wing weight")
        self.options.declare("n_ult", default=2.5 * 1.5, desc="Ultimate load factor (dimensionless)")
        self.options.declare("n_land_ult", default=2.8 * 1.5, desc="ultimate landing load factor")
        self.options.declare(
            "control_surface_area_frac", default=0.1, desc="Fraction of wing area covered by control surfaces and flaps"
        )
        self.options.declare("kneeling_main_gear_parameter", default=1.0, desc="Kneeling main landing gear parameter")
        self.options.declare("kneeling_nose_gear_parameter", default=1.0, desc="Kneeling nose landing gear parameter")
        self.options.declare("K_lav", default=0.7, desc="Lavatory weight coefficient")
        self.options.declare("K_buf", default=4.0, desc="Food weight coefficient")
        self.options.declare("coeff_fc", default=1.2 * 0.64, desc="Coefficient on flight control system weight")
        self.options.declare("coeff_avionics", default=0.5, desc="Coefficient on avionics weight")
        self.options.declare("APU_weight_frac", default=0.0085, desc="APU weight / MTOW")

    def setup(self):
        n_ult = self.options["n_ult"]

        # ==============================================================================
        # BWB structure weights
        # ==============================================================================
        # -------------- Pressurized cabin --------------
        self.add_subsystem(
            "cabin",
            CabinWeight_BWB(),
            promotes_inputs=["ac|geom|centerbody|S_cabin", "ac|weights|MTOW"],
            promotes_outputs=["W_cabin"],
        )

        # -------------- Unpressurized portion of the centerbody --------------
        self.add_subsystem(
            "aftbody",
            AftbodyWeight_BWB(),
            promotes_inputs=[
                "ac|geom|centerbody|S_aftbody",
                "ac|geom|centerbody|taper_aftbody",
                "ac|propulsion|num_engines",
                "ac|weights|MTOW",
            ],
            promotes_outputs=["W_aftbody"],
        )

        # -------------- Outboard wing modeled with conventional wing weight estimate --------------
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
                "ac|geom|V_pressurized",
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

        # ==============================================================================
        # Multiply structural weights by fudge factor
        # ==============================================================================
        structure_weight_outputs = ["W_wing", "W_cabin", "W_aftbody", "W_mlg", "W_nlg", "W_nacelle"]
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


class CabinWeight_BWB(om.ExplicitComponent):
    """
    Compute the weight of the pressurized cabin portion of a BWB centerbody. Estimate
    is based on a curve fit of FEA models. Details described in "A Sizing Methodology
    for the Conceptual Design of Blended-Wing-Body Transports" by Kevin R. Bradley.

    Inputs
    ------
    ac|geom|centerbody|S_cabin : float
        Planform area of the pressurized centerbody cabin area (scalar, sq ft)
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)

    Outputs
    -------
    W_cabin : float
        Weight of the pressurized cabin region of the BWB (scalar, lb)
    """

    def setup(self):
        self.add_input("ac|geom|centerbody|S_cabin", units="ft**2")
        self.add_input("ac|weights|MTOW", units="lb")
        self.add_output("W_cabin", val=100e3, units="lb")
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        outputs["W_cabin"] = (
            1.803246 * inputs["ac|weights|MTOW"] ** 0.166552 * inputs["ac|geom|centerbody|S_cabin"] ** 1.061158
        )

    def compute_partials(self, inputs, J):
        J["W_cabin", "ac|geom|centerbody|S_cabin"] = (
            1.803246
            * inputs["ac|weights|MTOW"] ** 0.166552
            * 1.061158
            * inputs["ac|geom|centerbody|S_cabin"] ** 0.061158
        )
        J["W_cabin", "ac|weights|MTOW"] = (
            1.803246
            * 0.166552
            * inputs["ac|weights|MTOW"] ** (0.166552 - 1)
            * inputs["ac|geom|centerbody|S_cabin"] ** 1.061158
        )


class AftbodyWeight_BWB(om.ExplicitComponent):
    """
    Compute the weight of the portion of the centerbody aft of the pressurized region
    (behind the rear spar). Estimate is based on a curve fit of FEA models. Details
    described in "A Sizing Methodology for the Conceptual Design of Blended-Wing-Body
    Transports" by Kevin R. Bradley.

    Inputs
    ------
    ac|geom|centerbody|S_aftbody : float
        Planform area of the centerbody aft of the cabin (scalar, sq ft)
    ac|geom|centerbody|taper_aftbody : float
        Taper ratio of the ceterbody region aft of the cabin (scalar, dimensionless)
    ac|propulsion|num_engines : float
        Number of engines (scalar, dimensionless)
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, lb)

    Outputs
    -------
    W_aftbody : float
        Weight of the centerbody region aft of the pressurized cabin (scalar, lb)
    """

    def setup(self):
        self.add_input("ac|geom|centerbody|S_aftbody", units="ft**2")
        self.add_input("ac|geom|centerbody|taper_aftbody")
        self.add_input("ac|propulsion|num_engines")
        self.add_input("ac|weights|MTOW", units="lb")
        self.add_output("W_aftbody", units="lb")
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        outputs["W_aftbody"] = (
            0.53
            * (1 + 0.05 * inputs["ac|propulsion|num_engines"])
            * inputs["ac|geom|centerbody|S_aftbody"]
            * inputs["ac|weights|MTOW"] ** 0.2
            * (inputs["ac|geom|centerbody|taper_aftbody"] + 0.5)
        )

    def compute_partials(self, inputs, J):
        J["W_aftbody", "ac|geom|centerbody|S_aftbody"] = (
            0.53
            * (1 + 0.05 * inputs["ac|propulsion|num_engines"])
            * inputs["ac|weights|MTOW"] ** 0.2
            * (inputs["ac|geom|centerbody|taper_aftbody"] + 0.5)
        )
        J["W_aftbody", "ac|geom|centerbody|taper_aftbody"] = (
            0.53
            * (1 + 0.05 * inputs["ac|propulsion|num_engines"])
            * inputs["ac|geom|centerbody|S_aftbody"]
            * inputs["ac|weights|MTOW"] ** 0.2
        )
        J["W_aftbody", "ac|propulsion|num_engines"] = (
            0.53
            * 0.05
            * inputs["ac|geom|centerbody|S_aftbody"]
            * inputs["ac|weights|MTOW"] ** 0.2
            * (inputs["ac|geom|centerbody|taper_aftbody"] + 0.5)
        )
        J["W_aftbody", "ac|weights|MTOW"] = (
            0.53
            * (1 + 0.05 * inputs["ac|propulsion|num_engines"])
            * inputs["ac|geom|centerbody|S_aftbody"]
            * 0.2
            * inputs["ac|weights|MTOW"] ** (0.2 - 1)
            * (inputs["ac|geom|centerbody|taper_aftbody"] + 0.5)
        )
