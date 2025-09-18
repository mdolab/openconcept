import numpy as np
import openmdao.api as om


class WingMACTrapezoidal(om.ExplicitComponent):
    """
    Compute the mean aerodynamic chord of a trapezoidal planform.

    Inputs
    ------
    S_ref : float
        Wing planform area (scalar, sq m)
    AR : float
        Wing aspect ratio (scalar, dimensionless)
    taper : float
        Wing taper ratio (scalar, dimensionless)

    Outputs
    -------
    MAC : float
        Mean aerodynamic chord of the trapezoidal planform (scalar, m)
    """

    def setup(self):
        self.add_input("S_ref", units="m**2")
        self.add_input("AR")
        self.add_input("taper")
        self.add_output("MAC", lower=1e-6, units="m")
        self.declare_partials("MAC", "*")

    def compute(self, inputs, outputs):
        S = inputs["S_ref"]
        AR = inputs["AR"]
        taper = inputs["taper"]

        c_root = np.sqrt(S / AR) * 2 / (1 + taper)
        c_tip = taper * c_root
        outputs["MAC"] = 2 / 3 * (c_root + c_tip - c_root * c_tip / (c_root + c_tip))

    def compute_partials(self, inputs, J):
        S = inputs["S_ref"]
        AR = inputs["AR"]
        taper = inputs["taper"]

        c_root = np.sqrt(S / AR) * 2 / (1 + taper)
        dcr_dS = 0.5 / np.sqrt(S * AR) * 2 / (1 + taper)
        dcr_dAR = -0.5 * S**0.5 / AR**1.5 * 2 / (1 + taper)
        dcr_dtaper = -np.sqrt(S / AR) * 2 / (1 + taper) ** 2

        c_tip = taper * c_root

        dMAC_dcr = 2 / 3 * (1 - c_tip**2 / (c_root + c_tip) ** 2)
        dMAC_dct = 2 / 3 * (1 - c_root**2 / (c_root + c_tip) ** 2)

        J["MAC", "S_ref"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dS
        J["MAC", "AR"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dAR
        J["MAC", "taper"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dtaper + dMAC_dct * c_root


class WingSpan(om.ExplicitComponent):
    """
    Compute the wing span as the square root of wing area times aspect ratio.

    Inputs
    ------
    S_ref : float
        Wing planform area (scalar, sq m)
    AR : float
        Wing aspect ratio (scalar, dimensionless)

    Outputs
    -------
    span : float
        Wing span (scalar, m)
    """

    def setup(self):
        self.add_input("S_ref", units="m**2")
        self.add_input("AR")

        self.add_output("span", units="m")
        self.declare_partials(["span"], ["*"])

    def compute(self, inputs, outputs):
        b = inputs["S_ref"] ** 0.5 * inputs["AR"] ** 0.5
        outputs["span"] = b

    def compute_partials(self, inputs, J):
        J["span", "S_ref"] = 0.5 * inputs["S_ref"] ** (0.5 - 1) * inputs["AR"] ** 0.5
        J["span", "AR"] = inputs["S_ref"] ** 0.5 * 0.5 * inputs["AR"] ** (0.5 - 1)


class WingAspectRatio(om.ExplicitComponent):
    """
    Compute the aspect ratio from span and wing area.

    Inputs
    ------
    S_ref : float
        Planform area (scalar, sq m)
    span : float
        Wing span (scalar, m)

    Outputs
    -------
    AR : float
        Aspect ratio, weighted by section areas (scalar, deg)
    """

    def setup(self):
        self.add_input("S_ref", units="m**2")
        self.add_input("span", units="m")
        self.add_output("AR", val=10.0, lower=1e-6)
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        outputs["AR"] = inputs["span"] ** 2 / inputs["S_ref"]

    def compute_partials(self, inputs, J):
        J["AR", "span"] = 2 * inputs["span"] / inputs["S_ref"]
        J["AR", "S_ref"] = -(inputs["span"] ** 2) / inputs["S_ref"] ** 2


class WingSweepFromSections(om.ExplicitComponent):
    """
    Compute the average quarter chord sweep angle weighted by section areas
    by taking in sectional parameters as they would be defined for a
    sectional OpenAeroStruct mesh. The actual average is of the cosine of
    the sweep angle, rather than the angle itself. This means that it will
    always return a positive sweep angle (because it does an arccos), even
    if the wing is forward swept.

    Inputs
    ------
    x_LE_sec : float
        Streamwise offset of the section's leading edge, starting with the outboard
        section (wing tip) and moving inboard toward the root (vector of length
        num_sections, m)
    y_sec : float
        Spanwise location of each section, starting with the outboard section (wing
        tip) at the MOST NEGATIVE y value and moving inboard (increasing y value)
        toward the root; the user does not provide a value for the root because it
        is always 0.0 (vector of length num_sections - 1, m)
    chord_sec : float
        Chord of each section, starting with the outboard section (wing tip) and
        moving inboard toward the root (vector of length num_sections, m)

    Outputs
    -------
    c4sweep : float
        Average quarter chord sweep, computed as the weighted average of
        cos(section sweep angle) by section areas and then arccos of the
        resulting quantity. This means it does not discriminate between
        forward and backward sweep angles (scalar, deg)

    Options
    -------
    num_sections : int
        Number of spanwise sections to define planform shape (scalar, dimensionless)
    idx_sec_start : int
        Index in the inputs to begin the average sweep calculation (negative indices not
        accepted), inclusive, by default 0
    idx_sec_end : int
        Index in the inputs to end the average sweep calculation (negative indices not
        accepted), inclusive, by default num_sections - 1
    """

    def initialize(self):
        self.options.declare(
            "num_sections", default=2, types=int, desc="Number of sections along the half span to define"
        )
        self.options.declare("idx_sec_start", default=0)
        self.options.declare("idx_sec_end", default=None)

    def setup(self):
        self.n_sec = self.options["num_sections"]
        self.i_start = self.options["idx_sec_start"]
        self.i_end = self.options["idx_sec_end"]
        if self.i_end is None:
            self.i_end = self.n_sec
        else:
            self.i_end += 1  # make it exclusive

        self.add_input("x_LE_sec", shape=(self.n_sec,), units="m")
        self.add_input("y_sec", shape=(self.n_sec - 1,), units="m")
        self.add_input("chord_sec", shape=(self.n_sec,), units="m")

        self.add_output("c4sweep", units="deg")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        # Extract out the ones we care about
        LE_sec = inputs["x_LE_sec"][self.i_start : self.i_end]
        chord_sec = inputs["chord_sec"][self.i_start : self.i_end]
        y_sec = np.hstack((inputs["y_sec"], [0.0]))[self.i_start : self.i_end]

        # Compute the c4sweep for each section
        x_c4 = LE_sec + chord_sec * 0.25
        widths = y_sec[1:] - y_sec[:-1]  # section width in y direction
        setback = x_c4[:-1] - x_c4[1:]  # relative offset of sections in streamwise direction
        to_rad = np.pi / 180
        c4sweep_sec = np.arctan(setback / widths) / to_rad

        # Perform a weighted average with panel areas as weights. Do the weighted average
        # on the cosine of the sweep angle rather than the angle itself.
        # This is consistent with OpenAeroStruct.
        A_sec = 0.5 * (chord_sec[:-1] + chord_sec[1:]) * widths
        outputs["c4sweep"] = np.arccos(np.sum(np.cos(c4sweep_sec * to_rad) * A_sec) / np.sum(A_sec)) / to_rad


class WingAreaFromSections(om.ExplicitComponent):
    """
    Compute the planform area of a specified portion of the wing
    by taking in sectional parameters as they would be defined for a
    sectional OpenAeroStruct mesh.

    NOTE: The area from this component is valid only if the scale_area
          option of mesh_gen is set to False! Otherwise, the area computed
          here will be off by a factor.

    Inputs
    ------
    y_sec : float
        Spanwise location of each section, starting with the outboard section (wing
        tip) at the MOST NEGATIVE y value and moving inboard (increasing y value)
        toward the root; the user does not provide a value for the root because it
        is always 0.0 (vector of length num_sections - 1, m)
    chord_sec : float
        Chord of each section, starting with the outboard section (wing tip) and
        moving inboard toward the root (vector of length num_sections, m)

    Outputs
    -------
    S : float
        Planform area of the specified region (scalar, sq m)

    Options
    -------
    num_sections : int
        Number of spanwise sections to define planform shape (scalar, dimensionless)
    idx_sec_start : int
        Index in the inputs to begin the average sweep calculation (negative indices not
        accepted), inclusive, by default 0
    idx_sec_end : int
        Index in the inputs to end the average sweep calculation (negative indices not
        accepted), inclusive, by default num_sections - 1
    chord_frac_start : float
        Fraction of the chord (streamwise direction) at which to begin the computed area, by default 0.0
    chord_frac_end : float
        Fraction of the chord (streamwise direction) at which to end the computed area, by default 1.0
    """

    def initialize(self):
        self.options.declare(
            "num_sections", default=2, types=int, desc="Number of sections along the half span to define"
        )
        self.options.declare("idx_sec_start", default=0)
        self.options.declare("idx_sec_end", default=None)
        self.options.declare("chord_frac_start", default=0.0, desc="Fraction of chord to begin area computation")
        self.options.declare("chord_frac_end", default=1.0, desc="Fraction of chord to end area computation")

    def setup(self):
        self.n_sec = self.options["num_sections"]
        self.i_start = self.options["idx_sec_start"]
        self.i_end = self.options["idx_sec_end"]
        if self.i_end is None:
            self.i_end = self.n_sec
        else:
            self.i_end += 1  # make it exclusive

        self.add_input("y_sec", shape=(self.n_sec - 1,), units="m")
        self.add_input("chord_sec", shape=(self.n_sec,), units="m")

        self.add_output("S", units="m**2")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        # Extract out the ones we care about
        chord_sec = inputs["chord_sec"][self.i_start : self.i_end]
        y_sec = np.hstack((inputs["y_sec"], [0.0]))[self.i_start : self.i_end]

        # Compute the area
        avg_chord = (
            0.5 * (chord_sec[:-1] + chord_sec[1:]) * (self.options["chord_frac_end"] - self.options["chord_frac_start"])
        )
        widths = y_sec[1:] - y_sec[:-1]
        outputs["S"] = 2 * np.sum(avg_chord * widths)


class WingMACFromSections(om.ExplicitComponent):
    """
    Compute the mean aerodynamic chord of an OpenAeroStruct section geoemtry.

    Inputs
    ------
    x_LE_sec : float
        Streamwise offset of the section's leading edge, starting with the outboard
        section (wing tip) and moving inboard toward the root (vector of length
        num_sections, m)
    y_sec : float
        Spanwise location of each section, starting with the outboard section (wing
        tip) at the MOST NEGATIVE y value and moving inboard (increasing y value)
        toward the root; the user does not provide a value for the root because it
        is always 0.0 (vector of length num_sections - 1, m)
    chord_sec : float
        Chord of each section, starting with the outboard section (wing tip) and
        moving inboard toward the root (vector of length num_sections, m)

    Outputs
    -------
    MAC : float
        Mean aerodynamic chord (scalar, m)
    x_c4MAC : float
        X location of the quarter chord of MAC in the same x coordinates as x_LE_sec (scalar, m)

    Options
    -------
    num_sections : int
        Number of spanwise sections to define planform shape (scalar, dimensionless)
    idx_sec_start : int
        Index in the inputs to begin the average sweep calculation (negative indices not
        accepted), inclusive, by default 0
    idx_sec_end : int
        Index in the inputs to end the average sweep calculation (negative indices not
        accepted), inclusive, by default num_sections - 1
    """

    def initialize(self):
        self.options.declare(
            "num_sections", default=2, types=int, desc="Number of sections along the half span to define"
        )
        self.options.declare("idx_sec_start", default=0)
        self.options.declare("idx_sec_end", default=None)

    def setup(self):
        self.n_sec = self.options["num_sections"]
        self.i_start = self.options["idx_sec_start"]
        self.i_end = self.options["idx_sec_end"]
        if self.i_end is None:
            self.i_end = self.n_sec
        else:
            self.i_end += 1  # make it exclusive

        self.add_input("x_LE_sec", shape=(self.n_sec,), units="m")
        self.add_input("y_sec", shape=(self.n_sec - 1,), units="m")
        self.add_input("chord_sec", shape=(self.n_sec,), units="m")

        self.add_output("MAC", lower=1e-6, units="m")
        self.add_output("x_c4MAC", units="m")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        # Extract out the ones we care about
        LE_sec = inputs["x_LE_sec"][self.i_start : self.i_end]
        chord_sec = inputs["chord_sec"][self.i_start : self.i_end]
        y_sec = np.hstack((inputs["y_sec"], [0.0]))[self.i_start : self.i_end]

        # Properties at the ouboard and inboard sections of each region
        x1 = LE_sec[:-1]
        c1 = chord_sec[:-1]
        y1 = y_sec[:-1]
        x2 = LE_sec[1:]
        c2 = chord_sec[1:]
        y2 = y_sec[1:]

        # Compute the planform area of the half wing
        widths = y2 - y1
        S = np.sum(0.5 * (c1 + c2) * widths)

        # Derived using the following MATLAB symbolic math code
        #
        #         syms y y1 y2 x1 x2 xq c1 c2 Sw

        #         % Shape functions
        #         N1 = (y2 - y) / (y2 - y1);
        #         N2 = (y - y1) / (y2 - y1);

        #         % Chord and quarter chord for the section as a function of y
        #         c = c1 * N1 + c2 * N2;
        #         xq = x1 * N1 + x2 * N2 + c / 4;

        #         % Mean aerodynamic chord
        #         cw = 1 / Sw * int(c^2, y, y1, y2)

        #         % Longitudinal position of MAC quarter chord from x = 0
        #         xqw = simplify(1 / Sw * int(c * xq, y, y1, y2))
        #
        outputs["MAC"] = np.sum((y2 - y1) * (c1**2 + c1 * c2 + c2**2) / (3 * S))
        outputs["x_c4MAC"] = np.sum(
            (y2 - y1) * (c1 * c2 + 4 * c1 * x1 + 2 * c1 * x2 + 2 * c2 * x1 + 4 * c2 * x2 + c1**2 + c2**2) / (12 * S)
        )
