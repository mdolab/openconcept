"""
@File    :   wave_drag.py
@Date    :   2023/04/17
@Author  :   Eytan Adler
@Description : Compute wave drag
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


class WaveDragFromSections(om.ExplicitComponent):
    """
    Compute the wave drag given a section-based geometry definition for an
    OpenAeroStruct mesh. This uses the same wave drag approximation as OpenAeroStruct,
    based on the Korn equation. Unlike OpenAeroStruct, it allows easy computation
    of wave drag for only a portion of the wing, which is useful for BWBs.

    Inputs
    ------
    fltcond|M : float
        Mach number (vector of lenght num_nodes, dimensionless)
    fltcond|CL : float
        Lift coefficient (vector of lenght num_nodes, dimensionless)
    y_sec : float
        Spanwise location of each section, starting with the outboard section (wing
        tip) at the MOST NEGATIVE y value and moving inboard (increasing y value)
        toward the root; the user does not provide a value for the root because it
        is always 0.0 (vector of length num_sections - 1, m)
    chord_sec : float
        Chord of each section, starting with the outboard section (wing tip) and
        moving inboard toward the root (vector of length num_sections, m)
    toverc_sec : float
        Thickness-to-chord ratio of each section, starting with the outboard section
        (wing tip) and moving inboard toward the root (vector of length num_sections, m)
    c4sweep : float
        Average quarter-chord sweep of the wing's region of interest (from idx_sec_start
        to idx_sec_end); can be computed using OpenConcept's WingSweepFromSections
        component in the geometry directory (scalar, deg)
    S_ref : float
        If specify_area_norm set to True, this input provides the area by which to normalize
        the drag coefficient (scalar, sq m)

    Outputs
    -------
    CD_wave : float
        Wave drag coefficient of the specified wing region normalized by the planform
        area of that region, unless specify_norm is set to True, in which case
        it is normalized by the planform area of the whole wing (scalar, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points per phase
    num_sections : int
        Number of spanwise sections to define planform shape (scalar, dimensionless)
    idx_sec_start : int
        Index in the inputs to begin the average sweep calculation (negative indices not
        accepted), inclusive, by default 0
    idx_sec_end : int
        Index in the inputs to end the average sweep calculation (negative indices not
        accepted), inclusive, by default num_sections - 1
    specify_area_norm : bool
        Add an input which determines the area by which to normalize the drag coefficient,
        otherwise will normalize by the area of the specified region, by default False
    airfoil_tech_level : float
        Airfoil technology level, by default 0.95
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int, desc="Number of analysis points")
        self.options.declare(
            "num_sections", default=2, types=int, desc="Number of sections along the half span to define"
        )
        self.options.declare("idx_sec_start", default=0, desc="Index of wing section at which to start")
        self.options.declare("idx_sec_end", default=None, desc="Index of wing section at which to end")
        self.options.declare(
            "specify_area_norm", default=False, types=bool, desc="Add area input by which to normalize"
        )
        self.options.declare("airfoil_tech_level", default=0.95, desc="Airfoil technology level")

    def setup(self):
        nn = self.options["num_nodes"]
        self.n_sec = self.options["num_sections"]
        self.i_start = self.options["idx_sec_start"]
        self.i_end = self.options["idx_sec_end"]
        if self.i_end is None:
            self.i_end = self.n_sec
        else:
            self.i_end += 1  # make it exclusive

        self.add_input("fltcond|M", shape=(nn,))
        self.add_input("fltcond|CL", shape=(nn,))
        self.add_input("toverc_sec", shape=(self.n_sec,))
        self.add_input("y_sec", shape=(self.n_sec - 1,), units="m")
        self.add_input("chord_sec", shape=(self.n_sec,), units="m")
        self.add_input("c4sweep", units="deg")
        if self.options["specify_area_norm"]:
            self.add_input("S_ref", units="m**2")

        self.add_output("CD_wave", val=0.0, shape=(nn,))

        # TODO: Add analytic derivatives
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        # Extract out the ones we care about
        chord_sec = inputs["chord_sec"][self.i_start : self.i_end]
        toverc_sec = inputs["toverc_sec"][self.i_start : self.i_end]
        y_sec = np.hstack((inputs["y_sec"], [0.0]))[self.i_start : self.i_end]
        cos_sweep = np.cos(inputs["c4sweep"] * np.pi / 180)
        M = inputs["fltcond|M"]
        CL = inputs["fltcond|CL"]

        panel_areas = 0.5 * (chord_sec[:-1] + chord_sec[1:]) * (y_sec[1:] - y_sec[:-1])

        # Numerically integrate to get area-averaged t/c (integrate t/c * c in y and divide result by region area)
        # TODO: Derive this analytically
        n = 500
        toverc_interp = np.linspace(toverc_sec[:-1], toverc_sec[1:], n)
        chord_interp = np.linspace(chord_sec[:-1], chord_sec[1:], n)
        y_interp = np.linspace(y_sec[:-1], y_sec[1:], n)
        panel_toverc = np.trapz((toverc_interp * chord_interp).T, x=y_interp.T) / panel_areas
        avg_toverc = np.sum(panel_toverc * panel_areas) / np.sum(panel_areas)

        MDD = self.options["airfoil_tech_level"] / cos_sweep - avg_toverc / cos_sweep**2 - CL / (10 * cos_sweep**3)
        M_crit = MDD - (0.1 / 80.0) ** (1 / 3)

        outputs["CD_wave"] = 0.0
        outputs["CD_wave"][M > M_crit] = 20 * (M[M > M_crit] - M_crit[M > M_crit]) ** 4
        outputs["CD_wave"] *= 2  # account for symmetry

        if self.options["specify_area_norm"]:
            outputs["CD_wave"] *= inputs["S_ref"] / (2 * np.sum(panel_areas))
