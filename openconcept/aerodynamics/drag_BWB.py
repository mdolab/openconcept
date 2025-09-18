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
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================
from openconcept.utilities import AddSubtractComp, ElementMultiplyDivideComp
from openconcept.aerodynamics.drag_jet_transport import SkinFrictionCoefficient_JetTransport, FlapDrag_JetTransport


class ParasiteDragCoefficient_BWB(om.Group):
    """
    Zero-lift drag coefficient buildup for BWB based on
    a combination of methods from Roskam, Raymer, and the approach presented by
    OpenVSP (https://openvsp.org/wiki/doku.php?id=parasitedrag). See the individual
    component docstrings for a more detailed explanation.

    NOTE: This component does not include drag from the wing, because it assumes
          it is computed by OpenAeroStruct. Add it in!

    Inputs
    ------
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|rho : float
        Air density (vector, kg/m^3)
    fltcond|T : float
        Air temperature (vector, K)
    ac|geom|wing|S_ref : float
        Wing planform area (scalar, sq m)
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
    configuration : str
        Aircraft configuration, either \"takeoff\" or \"clean\". Takeoff includes drag
        from landing gear and extended flaps. Clean assumes gear and flaps are retracted.
    FF_nacelle : float
        Nacelle form factor. By default 1.25 * 1.2, which is taken from the Jenkinson wing nacelle
        specified in the OpenVSP documentation (https://openvsp.org/wiki/doku.php?id=parasitedrag)
        multiplied by a rough estimate of the interference factor of 1.2, appx taken from Raymer.
        It was originally published in Civil Jet Aircraft by Jenkinson, Simpkin, and Rhodes (1999).
        Include any desired interference factor in the value provided to this option.
    flap_chord_frac : float
        Flap chord divided by wing chord, by default 0.2
    Q_flap : float
        Interference drag of flap. By default 1.25, from Roskam Equation 4.75 for Fowler flaps.
    wing_area_flapped_frac : float
        Flapped wing area divided by total wing area. Flapped wing area integrates the chord
        over any portions of the span that contain flaps (not just the area of the flap itself).
        Set this to zero to neglect drag from flaps, by default 0.
    drag_fudge_factor : float
        Multiplier on the resulting zero-lift drag coefficient estimate, by default 1.0
    nacelle_laminar_frac : float
        Fraction of the total engine nacelle length that has a laminary boundary layer, by default 0.0
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")
        self.options.declare("configuration", default="clean", values=["takeoff", "clean"])
        self.options.declare("drag_fudge_factor", default=1.0, desc="Multiplier on total drag coefficient")
        self.options.declare("FF_nacelle", default=1.25 * 1.2, desc="Nacelle form factor times interference factor")
        self.options.declare("flap_chord_frac", default=0.2, desc="Flap chord / wing chord")
        self.options.declare("Q_flap", default=1.25, desc="Flap interference factor")
        self.options.declare("wing_area_flapped_frac", default=0.0, desc="Flapped wing area / wing area")
        self.options.declare("nacelle_laminar_frac", default=0.0, desc="Fraction of engine nacelle with laminar flow")

    def setup(self):
        is_clean = self.options["configuration"] == "clean"
        nn = self.options["num_nodes"]

        iv = self.add_subsystem("iv", om.IndepVarComp())

        # ==============================================================================
        # Compute form factors
        # ==============================================================================
        # -------------- Nacelle --------------
        iv.add_output("FF_nacelle", val=self.options["FF_nacelle"])

        # ==============================================================================
        # Skin friction coefficients for each component
        # ==============================================================================
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
                "ac|geom|nacelle|S_wet",
                "ac|propulsion|num_engines",
                "ac|geom|wing|S_ref",
            ],
        )
        mult.add_equation(
            output_name="CD_nacelle",
            input_names=[
                "ac|geom|nacelle|S_wet",
                "FF_nacelle",
                "Cf_nacelle",
                "ac|propulsion|num_engines",
                "ac|geom|wing|S_ref",
            ],
            vec_size=[1, 1, nn, 1, 1],
            input_units=["m**2", None, None, None, "m**2"],
            divide=[False, False, False, False, True],
        )

        # -------------- Internal connections --------------
        self.connect("iv.FF_nacelle", "drag_coeffs.FF_nacelle")
        self.connect("nacelle_friction.Cf", "drag_coeffs.Cf_nacelle")

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
        # For BWBs, this value has been halved because the wing area has roughly doubled from
        # a comparable tube-and-wing configuration (assuming same drag force for gear).
        iv.add_output("CD_landing_gear", val=0.0 if is_clean else 0.01)

        # ==============================================================================
        # Sum the total drag coefficients
        # ==============================================================================
        drag_coeff_inputs = ["CD_nacelle", "CD_flap", "CD_landing_gear"]
        self.add_subsystem(
            "sum_CD0",
            AddSubtractComp(
                output_name="CD0",
                input_names=drag_coeff_inputs,
                vec_size=[nn, 1, 1],
                scaling_factors=[self.options["drag_fudge_factor"]] * len(drag_coeff_inputs),
            ),
            promotes_outputs=["CD0"],
        )

        self.connect("drag_coeffs.CD_nacelle", "sum_CD0.CD_nacelle")
        self.connect(CD_flap_source, "sum_CD0.CD_flap")
        self.connect("iv.CD_landing_gear", "sum_CD0.CD_landing_gear")
