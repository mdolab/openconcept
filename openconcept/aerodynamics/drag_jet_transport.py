"""
@File    :   drag_jet_transport.py
@Date    :   2023/03/23
@Author  :   Eytan Adler
@Description : Zero lift drag coefficient buildup for tube-and-wing jet transport aircraft
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
from openconcept.utilities import AddSubtractComp, ElementMultiplyDivideComp
from openconcept.geometry import WingMACTrapezoidal


class ParasiteDragCoefficient_JetTransport(om.Group):
    """
    Zero-lift drag coefficient buildup for jet transport aircraft based on
    a combination of methods from Roskam, Raymer, and the approach presented by
    OpenVSP (https://openvsp.org/wiki/doku.php?id=parasitedrag). See the individual
    component docstrings for a more detailed explanation.

    Inputs
    ------
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|rho : float
        Air density (vector, kg/m^3)
    fltcond|T : float
        Air temperature (vector, K)
    ac|geom|fuselage|length : float
        Fuselage length (scalar, m)
    ac|geom|fuselage|height : float
        Fuselage height (scalar, m)
    ac|geom|fuselage|S_wet : float
        Fuselage wetted area (scalar, sq m)
    ac|geom|hstab|S_ref : float
        Horizontal stabilizer planform area (scalar, sq m)
    ac|geom|hstab|AR : float
        Horizontal stabilizer aspect ratio (scalar, dimensionless)
    ac|geom|hstab|taper : float
        Horizontal stabilizer taper ratio (scalar, dimensionless)
    ac|geom|hstab|toverc : float
        Horizontal stabilizer thickness-to-chord ratio (scalar, dimensionless)
    ac|geom|vstab|S_ref : float
        Vertical stabilizer planform area (scalar, sq m)
    ac|geom|vstab|AR : float
        Vertical stabilizer aspect ratio (scalar, dimensionless)
    ac|geom|vstab|taper : float
        Vertical stabilizer taper ratio (scalar, dimensionless)
    ac|geom|vstab|toverc : float
        Vertical stabilizer thickness-to-chord ratio (scalar, dimensionless)
    ac|geom|wing|S_ref : float
        Wing planform area (scalar, sq m)
    ac|geom|wing|AR : float
        If include wing (otherwise not an input), wing aspect ratio (scalar, dimensionless)
    ac|geom|wing|taper : float
        If include wing (otherwise not an input), wing taper ratio (scalar, dimensionless)
    ac|geom|wing|toverc : float
        If include wing (otherwise not an input), wing thickness-to-chord ratio (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        If configuration is \"takeoff\" (otherwise not an input), wing quarter chord sweep (scalar, rad)
    ac|geom|nacelle|length : float
        Nacelle length (scalar, m)
    ac|geom|nacelle|S_wet : float
        Nacelle wetted area (scalar, sq m)
    ac|propulsion|num_engines : float
        Number of engines, multiplier on nacelle drag (scalar, dimensionless)
    ac|aero|takeoff_flap_deg : float
        If configuration is \"takeoff\" (otherwise not an input), flap setting on takeoff (scalar, deg)

    Outputs
    -------
    CD0 : float
        Zero-lift drag coefficient (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points per phase, by default 1
    include_wing : bool
        Include an estimate of the drag of the wing in the output drag estimate, by default True.
    configuration : str
        Aircraft configuration, either \"takeoff\" or \"clean\". Takeoff includes drag
        from landing gear and extended flaps. Clean assumes gear and flaps are retracted.
    FF_nacelle : float
        Nacelle form factor. By default 1.25 * 1.2, which is taken from the Jenkinson wing nacelle
        specified in the OpenVSP documentation (https://openvsp.org/wiki/doku.php?id=parasitedrag)
        multiplied by a rough estimate of the interference factor of 1.2, appx taken from Raymer.
        It was originally published in Civil Jet Aircraft by Jenkinson, Simpkin, and Rhodes (1999).
        Include any desired interference factor in the value provided to this option.
    Q_fuselage : float
        Interference factor for fuselage to multiply the form factor estimate. By
        default 1.0 from Raymer.
    Q_tail : float
        Interference factor for horizontal and vertical stabilizers to multiply the form factor
        estimate. By default 1.05 from Raymer for conventional tail configuration.
    Q_wing : float
        Interference factor for wing to multiply the form factor estimate. By
        default 1.0.
    flap_chord_frac : float
        Flap chord divided by wing chord, by default 0.2
    Q_flap : float
        Interference drag of flap. By default 1.25, from Roskam Equation 4.75 for Fowler flaps.
    wing_area_flapped_frac : float
        Flapped wing area divided by total wing area. Flapped wing area integrates the chord
        over any portions of the span that contain flaps (not just the area of the flap itself).
        By default 0.9.
    drag_fudge_factor : float
        Multiplier on the resulting zero-lift drag coefficient estimate, by default 1.0
    fuselage_laminar_frac : float
        Fraction of the total fuselage length that has a laminary boundary layer, by default 0.0
    hstab_laminar_frac : float
        Fraction of the total horizontal stabilizer that has a laminary boundary layer, by default 0.15
    vstab_laminar_frac : float
        Fraction of the total vertical stabilizer that has a laminary boundary layer, by default 0.15
    wing_laminar_frac : float
        Fraction of the total wing that has a laminary boundary layer, by default 0.15
    nacelle_laminar_frac : float
        Fraction of the total engine nacelle length that has a laminary boundary layer, by default 0.0
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")
        self.options.declare("include_wing", default=True, types=bool, desc="Include the wing drag")
        self.options.declare("configuration", default="clean", values=["takeoff", "clean"])
        self.options.declare("drag_fudge_factor", default=1.0, desc="Multiplier on total drag coefficient")
        self.options.declare("FF_nacelle", default=1.25 * 1.2, desc="Nacelle form factor times interference factor")
        self.options.declare("Q_fuselage", default=1.0, desc="Fuselage interference factor")
        self.options.declare("Q_tail", default=1.05, desc="Tail interference factor")
        self.options.declare("Q_wing", default=1.0, desc="Wing interference factor")
        self.options.declare("flap_chord_frac", default=0.2, desc="Flap chord / wing chord")
        self.options.declare("Q_flap", default=1.25, desc="Flap interference factor")
        self.options.declare("wing_area_flapped_frac", default=0.9, desc="Flapped wing area / wing area")
        self.options.declare("fuselage_laminar_frac", default=0.0, desc="Fraction of fuselage with laminar flow")
        self.options.declare("hstab_laminar_frac", default=0.15, desc="Fraction of horizontal tail with laminar flow")
        self.options.declare("vstab_laminar_frac", default=0.15, desc="Fraction of vertical tail with laminar flow")
        self.options.declare("wing_laminar_frac", default=0.15, desc="Fraction of wing with laminar flow")
        self.options.declare("nacelle_laminar_frac", default=0.0, desc="Fraction of engine nacelle with laminar flow")

    def setup(self):
        is_clean = self.options["configuration"] == "clean"
        include_wing = self.options["include_wing"]
        nn = self.options["num_nodes"]

        iv = self.add_subsystem("iv", om.IndepVarComp())

        # ==============================================================================
        # Compute form factors
        # ==============================================================================
        # -------------- Fuselage --------------
        self.add_subsystem(
            "fuselage_form",
            FuselageFormFactor_JetTransport(Q_fuselage=self.options["Q_fuselage"]),
            promotes_inputs=["ac|geom|fuselage|length", "ac|geom|fuselage|height"],
        )

        # -------------- Horizontal stabilizer --------------
        self.add_subsystem(
            "hstab_form",
            WingFormFactor_JetTransport(Q=self.options["Q_tail"]),
            promotes_inputs=[("toverc", "ac|geom|hstab|toverc")],
        )

        # -------------- Vertical stabilizer --------------
        self.add_subsystem(
            "vstab_form",
            WingFormFactor_JetTransport(Q=self.options["Q_tail"]),
            promotes_inputs=[("toverc", "ac|geom|vstab|toverc")],
        )

        # -------------- Wing --------------
        if include_wing:
            self.add_subsystem(
                "wing_form",
                WingFormFactor_JetTransport(Q=self.options["Q_wing"]),
                promotes_inputs=[("toverc", "ac|geom|wing|toverc")],
            )

        # -------------- Nacelle --------------
        iv.add_output("FF_nacelle", val=self.options["FF_nacelle"])

        # ==============================================================================
        # Skin friction coefficients for each component
        # ==============================================================================
        # -------------- Fuselage --------------
        self.add_subsystem(
            "fuselage_friction",
            SkinFrictionCoefficient_JetTransport(num_nodes=nn, laminar_frac=self.options["fuselage_laminar_frac"]),
            promotes_inputs=["fltcond|Utrue", "fltcond|rho", "fltcond|T", ("L", "ac|geom|fuselage|length")],
        )

        # -------------- Horizontal stabilizer, vertical stabilizer, and wing (if included) --------------
        wing_surfs = ["hstab", "vstab"]
        if include_wing:
            wing_surfs.append("wing")
        for surf in wing_surfs:
            self.add_subsystem(
                f"{surf}_MAC_calc",
                WingMACTrapezoidal(),
                promotes_inputs=[
                    ("S_ref", f"ac|geom|{surf}|S_ref"),
                    ("AR", f"ac|geom|{surf}|AR"),
                    ("taper", f"ac|geom|{surf}|taper"),
                ],
            )
            self.add_subsystem(
                f"{surf}_friction",
                SkinFrictionCoefficient_JetTransport(num_nodes=nn, laminar_frac=self.options[f"{surf}_laminar_frac"]),
                promotes_inputs=["fltcond|Utrue", "fltcond|rho", "fltcond|T"],
            )
            self.connect(f"{surf}_MAC_calc.MAC", f"{surf}_friction.L")

        # -------------- Nacelle --------------
        self.add_subsystem(
            "nacelle_friction",
            SkinFrictionCoefficient_JetTransport(num_nodes=nn, laminar_frac=self.options["nacelle_laminar_frac"]),
            promotes_inputs=["fltcond|Utrue", "fltcond|rho", "fltcond|T", ("L", "ac|geom|nacelle|length")],
        )

        # ==============================================================================
        # Compute the parasitic drag coefficient
        # ==============================================================================
        mult = self.add_subsystem(
            "drag_coeffs",
            ElementMultiplyDivideComp(),
            promotes_inputs=[
                "ac|geom|fuselage|S_wet",
                "ac|geom|hstab|S_ref",
                "ac|geom|vstab|S_ref",
                "ac|geom|nacelle|S_wet",
                "ac|propulsion|num_engines",
                ("S_wing_1", "ac|geom|wing|S_ref"),  # This is a bit of a hack needed to
                ("S_wing_2", "ac|geom|wing|S_ref"),  # allow multiple equations to share
                ("S_wing_3", "ac|geom|wing|S_ref"),  # the same input
                ("S_wing_4", "ac|geom|wing|S_ref"),
            ],
        )
        mult.add_equation(
            output_name="CD_fuselage",
            input_names=["ac|geom|fuselage|S_wet", "FF_fuselage", "Cf_fuselage", "S_wing_1"],
            vec_size=[1, 1, nn, 1],
            input_units=["m**2", None, None, "m**2"],
            divide=[False, False, False, True],
        )
        mult.add_equation(
            output_name="CD_hstab",
            input_names=["ac|geom|hstab|S_ref", "FF_hstab", "Cf_hstab", "S_wing_2"],
            vec_size=[1, 1, nn, 1],
            input_units=["m**2", None, None, "m**2"],
            divide=[False, False, False, True],
            scaling_factor=2,
        )  # scaling factor of two is since wetted area is ~2x reference area
        mult.add_equation(
            output_name="CD_vstab",
            input_names=["ac|geom|vstab|S_ref", "FF_vstab", "Cf_vstab", "S_wing_3"],
            vec_size=[1, 1, nn, 1],
            input_units=["m**2", None, None, "m**2"],
            divide=[False, False, False, True],
            scaling_factor=2,
        )  # scaling factor of two is since wetted area is ~2x reference area
        mult.add_equation(
            output_name="CD_nacelle",
            input_names=["ac|geom|nacelle|S_wet", "FF_nacelle", "Cf_nacelle", "ac|propulsion|num_engines", "S_wing_4"],
            vec_size=[1, 1, nn, 1, 1],
            input_units=["m**2", None, None, None, "m**2"],
            divide=[False, False, False, False, True],
        )
        if include_wing:
            mult.add_equation(
                output_name="CD_wing",
                input_names=["FF_wing", "Cf_wing"],
                vec_size=[1, nn],
                input_units=[None, None],
                divide=[False, False],
                scaling_factor=2,
            )  # scaling factor of two is since wetted area is ~2x reference area

        # -------------- Internal connections --------------
        self.connect("fuselage_form.FF_fuselage", "drag_coeffs.FF_fuselage")
        self.connect("hstab_form.FF_wing", "drag_coeffs.FF_hstab")
        self.connect("vstab_form.FF_wing", "drag_coeffs.FF_vstab")
        self.connect("iv.FF_nacelle", "drag_coeffs.FF_nacelle")

        for surf in ["fuselage", "hstab", "vstab", "nacelle"]:
            self.connect(f"{surf}_friction.Cf", f"drag_coeffs.Cf_{surf}")

        if include_wing:
            self.connect("wing_form.FF_wing", "drag_coeffs.FF_wing")
            self.connect("wing_friction.Cf", "drag_coeffs.Cf_wing")

        # ==============================================================================
        # Any addition drag sources in the takeoff configuration
        # ==============================================================================
        # -------------- Flaps --------------
        if is_clean:
            CD_flap_source = "iv.CD_flap"
            iv.add_output("CD_flap", val=0.0)
        else:
            CD_flap_source = "flaps.CD_flap"
            self.add_subsystem(
                "flaps",
                FlapDrag_JetTransport(
                    flap_chord_frac=self.options["flap_chord_frac"],
                    Q_flap=self.options["Q_flap"],
                    wing_area_flapped_frac=self.options["wing_area_flapped_frac"],
                ),
                promotes_inputs=[("flap_extension", "ac|aero|takeoff_flap_deg"), "ac|geom|wing|c4sweep"],
            )

        # -------------- Landing gear --------------
        # Raymer suggests adding 0.02 to the zero-lift drag coefficient when retractable
        # landing gear are in the down position. See Section 5.3, page 99 in the 1992 edition.
        iv.add_output("CD_landing_gear", val=0.0 if is_clean else 0.02)

        # ==============================================================================
        # Sum the total drag coefficients
        # ==============================================================================
        drag_coeff_inputs = ["CD_fuselage", "CD_hstab", "CD_vstab", "CD_nacelle"]
        if include_wing:
            drag_coeff_inputs.append("CD_wing")
        drag_coeff_inputs += ["CD_flap", "CD_landing_gear"]
        self.add_subsystem(
            "sum_CD0",
            AddSubtractComp(
                output_name="CD0",
                input_names=drag_coeff_inputs,
                vec_size=[nn] * (len(drag_coeff_inputs) - 2) + [1, 1],
                scaling_factors=[self.options["drag_fudge_factor"]] * len(drag_coeff_inputs),
            ),
            promotes_outputs=["CD0"],
        )

        self.connect("drag_coeffs.CD_fuselage", "sum_CD0.CD_fuselage")
        self.connect("drag_coeffs.CD_hstab", "sum_CD0.CD_hstab")
        self.connect("drag_coeffs.CD_vstab", "sum_CD0.CD_vstab")
        self.connect("drag_coeffs.CD_nacelle", "sum_CD0.CD_nacelle")
        if include_wing:
            self.connect("drag_coeffs.CD_wing", "sum_CD0.CD_wing")
        self.connect(CD_flap_source, "sum_CD0.CD_flap")
        self.connect("iv.CD_landing_gear", "sum_CD0.CD_landing_gear")


class FuselageFormFactor_JetTransport(om.ExplicitComponent):
    """
    Form factor of fuselage based on slender body form factor equation from Torenbeek
    1982 (Synthesis of Subsonic Aircraft Design), taken from OpenVSP documentation:
    https://openvsp.org/wiki/doku.php?id=parasitedrag (accessed March 23, 2023)

    Inputs
    ------
    ac|geom|fuselage|length : float
        Fuselage structural length (scalar, m)
    ac|geom|fuselage|height : float
        Fuselage height (scalar, m)

    Outputs
    -------
    FF_fuselage : float
        Fuselage form factor (scalar, dimensionless)

    Options
    -------
    Q_fuselage : float
        Interference factor for fuselage to multiply the form factor estimate. By
        default 1.0 from Raymer.
    """

    def initialize(self):
        self.options.declare("Q_fuselage", default=1.0, desc="Fuselage interference factor")

    def setup(self):
        self.add_input("ac|geom|fuselage|length", units="m")
        self.add_input("ac|geom|fuselage|height", units="m")
        self.add_output("FF_fuselage")
        self.declare_partials("FF_fuselage", ["ac|geom|fuselage|length", "ac|geom|fuselage|height"])

    def compute(self, inputs, outputs):
        fineness_ratio = inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"]
        outputs["FF_fuselage"] = self.options["Q_fuselage"] * (1 + 2.2 / fineness_ratio + 3.8 / fineness_ratio**3)

    def compute_partials(self, inputs, J):
        L = inputs["ac|geom|fuselage|length"]
        D = inputs["ac|geom|fuselage|height"]
        Q = self.options["Q_fuselage"]
        fineness_ratio = L / D
        df_dL = 1 / D
        df_dD = -L / D**2
        J["FF_fuselage", "ac|geom|fuselage|length"] = (
            Q * (-2.2 / fineness_ratio**2 - 3 * 3.8 / fineness_ratio**4) * df_dL
        )
        J["FF_fuselage", "ac|geom|fuselage|height"] = (
            Q * (-2.2 / fineness_ratio**2 - 3 * 3.8 / fineness_ratio**4) * df_dD
        )


class WingFormFactor_JetTransport(om.ExplicitComponent):
    """
    Form factor of wing times the optional interference factor based on wing form factor equation from
    Torenbeek 1982 (Synthesis of Subsonic Aircraft Design), but can be applied to other lifting surfaces.
    Taken from OpenVSP documentation: https://openvsp.org/wiki/doku.php?id=parasitedrag (accessed March 23, 2023)

    Inputs
    ------
    toverc : float
        Thickness-to-chord ratio of the wing surface

    Outputs
    -------
    FF_wing : float
        Wing form factor (scalar, dimensionless)

    Options
    -------
    Q : float
        Interference factor for fuselage to multiply the form factor estimate, by default 1.0
    """

    def initialize(self):
        self.options.declare("Q", default=1.0, desc="Wing interference factor")

    def setup(self):
        self.add_input("toverc")
        self.add_output("FF_wing")
        self.declare_partials("FF_wing", "toverc")

    def compute(self, inputs, outputs):
        tc = inputs["toverc"]
        outputs["FF_wing"] = self.options["Q"] * (1 + 2.7 * tc + 100 * tc**4)

    def compute_partials(self, inputs, J):
        tc = inputs["toverc"]
        J["FF_wing", "toverc"] = self.options["Q"] * (2.7 + 400 * tc**3)


class FlapDrag_JetTransport(om.ExplicitComponent):
    """
    Estimates the additional drag from extending the flaps using Roskam 1989 Part VI
    Chapter 4 Equation 4.71 and 4.75. This assumes Fowler flaps.

    Inputs
    ------
    flap_extension : float
        Flap extension amount (scalar, deg)
    ac|geom|wing|c4sweep : float
        Wing sweep at 25% mean aerodynamic chord (scalar, radians)

    Outputs
    -------
    CD_flap : float
        Increment to drag coefficient from flap profile drag (scalar, dimensionless)

    Options
    -------
    flap_chord_frac : float
        Flap chord divided by wing chord, by default 0.2
    Q_flap : float
        Interference drag of flap. By default 1.25, from Roskam Equation 4.75 for Fowler flaps.
    wing_area_flapped_frac : float
        Flapped wing area divided by total wing area. Flapped wing area integrates the chord
        over any portions of the span that contain flaps (not just the area of the flap itself).
        By default 0.9.
    """

    def initialize(self):
        self.options.declare("flap_chord_frac", default=0.2, desc="Flap chord / wing chord")
        self.options.declare("Q_flap", default=1.25, desc="Flap interference factor")
        self.options.declare("wing_area_flapped_frac", default=0.9, desc="Flapped wing area / wing area")

    def setup(self):
        self.add_input("flap_extension", units="deg")
        self.add_input("ac|geom|wing|c4sweep", units="rad")
        self.add_output("CD_flap")
        self.declare_partials("CD_flap", "*")

        # -------------- 2D profile drag increment due to flaps --------------
        # This approximation is a very rough curve of Figure 4.48 in Roskam 1989 Part VI
        # Chapter 4, which shows the 2D drag increment of Fowler flaps as a function of
        # flap extension angle and cf/c, which I presume is flap chord over wing chord
        cf_c = self.options["flap_chord_frac"]
        self.quadratic_coeff = 5.75e-4 * cf_c**2 - 7.45e-5 * cf_c + 1.23e-5

    def compute(self, inputs, outputs):
        delta = inputs["flap_extension"]
        sweep = inputs["ac|geom|wing|c4sweep"]

        # 2D profile drag increment
        airfoil_drag_incr = self.quadratic_coeff * delta**2

        outputs["CD_flap"] = (
            airfoil_drag_incr * np.cos(sweep) * self.options["wing_area_flapped_frac"] * self.options["Q_flap"]
        )

    def compute_partials(self, inputs, J):
        delta = inputs["flap_extension"]
        sweep = inputs["ac|geom|wing|c4sweep"]

        # 2D profile drag increment
        airfoil_drag_incr = self.quadratic_coeff * delta**2

        J["CD_flap", "flap_extension"] = (
            2
            * self.quadratic_coeff
            * delta
            * np.cos(sweep)
            * self.options["wing_area_flapped_frac"]
            * self.options["Q_flap"]
        )
        J["CD_flap", "ac|geom|wing|c4sweep"] = (
            -airfoil_drag_incr * np.sin(sweep) * self.options["wing_area_flapped_frac"] * self.options["Q_flap"]
        )


class SkinFrictionCoefficient_JetTransport(om.ExplicitComponent):
    """
    Compute the average coefficient of friction using the methodology described here:
    https://openvsp.org/wiki/doku.php?id=parasitedrag. It uses the Blasius and explicit
    fit of Spalding correlations.

    Inputs
    ------
    L : float
        Characteristic length to use in the Reynolds number computation (scalar, m)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|rho : float
        Air density (vector, kg/m^3)
    fltcond|T : float
        Air temperature (vector, K)

    Outputs
    -------
    Cf : float
        Skin friction coefficient (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points per phase, by default 1
    laminar_frac : float
        Fraction of total length that has a laminar boundary layer, by default 0.0
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")
        self.options.declare("laminar_frac", default=0.0, desc="Fraction of length that is laminar")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("L", units="m")
        self.add_input("fltcond|Utrue", units="m/s", shape=(nn,))
        self.add_input("fltcond|rho", units="kg/m**3", shape=(nn,))
        self.add_input("fltcond|T", units="K", shape=(nn,))

        self.add_output("Cf", shape=(nn,))

        arng = np.arange(nn)
        self.declare_partials("Cf", "fltcond|*", rows=arng, cols=arng)
        self.declare_partials("Cf", "L", rows=arng, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        # Compute the kinematic viscosity
        beta = 1.458e-6
        sutherlands_constant = 100.4
        visc_dyn = beta * inputs["fltcond|T"] ** (3 / 2) / (inputs["fltcond|T"] + sutherlands_constant)
        visc_kin = visc_dyn / inputs["fltcond|rho"]

        # Reynolds number
        Re = inputs["fltcond|Utrue"] * inputs["L"] / visc_kin

        # Skin friction coefficient assuming fully turbulent
        Cf = 0.523 / np.log(0.06 * Re) ** 2  # explicit fit of Spalding

        # Case where some is laminar
        lam_frac = self.options["laminar_frac"]
        if lam_frac > 0.0:
            Re_lam = Re * lam_frac
            Cf_lam = 1.32824 / np.sqrt(Re_lam)  # Blasius
            Cf_turb = 0.523 / np.log(0.06 * Re_lam) ** 2  # explicit fit of Spalding
            Cf = Cf + (Cf_lam - Cf_turb) * lam_frac

        outputs["Cf"] = Cf

    def compute_partials(self, inputs, J):
        # Compute the kinematic viscosity
        beta = 1.458e-6
        S = 100.4
        T = inputs["fltcond|T"]
        visc_dyn = beta * T ** (3 / 2) / (T + S)
        visc_kin = visc_dyn / inputs["fltcond|rho"]

        dvdyn_dT = beta * np.sqrt(T) * (3 * S + T) / (2 * (S + T) ** 2)
        dvkin_dT = dvdyn_dT / inputs["fltcond|rho"]
        dvkin_drho = -visc_dyn / inputs["fltcond|rho"] ** 2

        # Reynolds number
        U = inputs["fltcond|Utrue"]
        L = inputs["L"]
        Re = U * L / visc_kin
        dRe_dT = -U * L / visc_kin**2 * dvkin_dT
        dRe_drho = -U * L / visc_kin**2 * dvkin_drho
        dRe_dU = L / visc_kin
        dRe_dL = U / visc_kin

        # Skin friction coefficient assuming fully turbulent
        Cf = 0.523 / np.log(0.06 * Re) ** 2  # explicit fit of Spalding
        dCf_dRe = -1.046 / (Re * (np.log(Re) - 2.81341) ** 3)

        J["Cf", "fltcond|T"] = dCf_dRe * dRe_dT
        J["Cf", "fltcond|rho"] = dCf_dRe * dRe_drho
        J["Cf", "fltcond|Utrue"] = dCf_dRe * dRe_dU
        J["Cf", "L"] = dCf_dRe * dRe_dL

        # Case where some is laminar
        lam_frac = self.options["laminar_frac"]
        if lam_frac > 0.0:
            Re_lam = Re * lam_frac
            Cf_lam = 1.32824 / np.sqrt(Re_lam)  # Blasius
            Cf_turb = 0.523 / np.log(0.06 * Re_lam) ** 2  # explicit fit of Spalding
            dCflam_dRelam = -0.5 * 1.32824 / Re_lam**1.5
            dCfturb_dRelam = -1.046 / (Re_lam * (np.log(Re_lam) - 2.81341) ** 3)
            Cf = Cf + (Cf_lam - Cf_turb) * lam_frac

            J["Cf", "fltcond|T"] += lam_frac * (dCflam_dRelam - dCfturb_dRelam) * lam_frac * dRe_dT
            J["Cf", "fltcond|rho"] += lam_frac * (dCflam_dRelam - dCfturb_dRelam) * lam_frac * dRe_drho
            J["Cf", "fltcond|Utrue"] += lam_frac * (dCflam_dRelam - dCfturb_dRelam) * lam_frac * dRe_dU
            J["Cf", "L"] += lam_frac * (dCflam_dRelam - dCfturb_dRelam) * lam_frac * dRe_dL
