"""
@File    :   CLmax_jet_transport.py
@Date    :   2023/04/11
@Author  :   Eytan Adler
@Description : Max lift coefficient estimate using critical section method
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
from openconcept.aerodynamics.openaerostruct import VLM
from openconcept.utilities import AddSubtractComp


class CLmaxCriticalSectionVLM(om.Group):
    """
    Predict the maximum lift coefficient of the wing by solving for the
    angle of attack where the maximum sectional lift coefficient equals the
    provided maximum lift coefficient of the airfoil. The solution for the
    sectional lift coefficients is done using OpenAeroStruct.

    Inputs
    ------
    ac|aero|airfoil_Cl_max : float
        Maximum 2D lift coefficient of the wing airfoil (scalar, dimensionless)
    fltcond|M : float
        Mach number for maximum CL calculation (scalar, dimensionless)
    fltcond|h : float
        Altitude for maximum CL calculation (scalar, m)
    fltcond|TempIncrement : float
        Temperature increment for maximum CL calculation (scalar, degC)
    ac|geom|wing|OAS_mesh: float
        OpenAeroStruct 3D mesh (num_x + 1 x num_y + 1 x 3 vector, m)
    ac|geom|wing|toverc : float
        Thickness to chord ratio of each streamwise strip of panels ordered from wing
        tip to wing root; used for the viscous and wave drag calculations
        (vector of length num_y, dimensionless)

    Outputs
    -------
    CL_max : float
        Maximum lift coefficient estimated using the critical section method (scalar, dimensionless)

    Options
    -------
    num_x : int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of panels in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    surf_options : dict
        Dictionary of OpenAeroStruct surface options; any options provided here
        will override the default ones; see the OpenAeroStruct documentation for more information.
        Because the geometry transformations are excluded in this model (to simplify the interface),
        the <transformation>_cp options are not supported. The t_over_c option is also removed since
        it is provided instead as an input.
    rho : float or int
        Constraint aggregation factor for sectional lift coefficient aggregation, by default 200
    """

    def initialize(self):
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare("num_y", default=6, desc="Number of spanwise (half wing) mesh panels")
        self.options.declare("surf_options", default=None, desc="Dictionary of OpenAeroStruct surface options")
        self.options.declare("rho", default=200, types=(float, int), desc="Sectional CL aggregation factor")

    def setup(self):
        # TODO: Extend the method to accept different maximum sectional
        # lift coefficient methods across the span of the wing

        # -------------- Simulate the wing in OpenAeroStruct --------------
        self.add_subsystem(
            "VLM",
            VLM(num_x=self.options["num_x"], num_y=self.options["num_y"], surf_options=self.options["surf_options"]),
            promotes_inputs=[
                "fltcond|M",
                "fltcond|h",
                "fltcond|TempIncrement",
                "ac|geom|wing|OAS_mesh",
                "ac|geom|wing|toverc",
            ],
            promotes_outputs=[("fltcond|CL", "CL_max")],
        )

        # -------------- Aggregate the sectional lift coefficients to find the max --------------
        self.add_subsystem("max_sectional_CL", om.KSComp(width=self.options["num_y"], rho=self.options["rho"]))
        self.connect("VLM.sectional_CL", "max_sectional_CL.g")

        # -------------- Solve for the angle of attack --------------
        self.add_subsystem(
            "sectional_CL_balance",
            om.BalanceComp("alpha", lhs_name="sec_CL_max_VLM", rhs_name="airfoil_CL_max", val=5, units="deg"),
            promotes_inputs=[("airfoil_CL_max", "ac|aero|airfoil_Cl_max")],
        )
        self.connect("max_sectional_CL.KS", "sectional_CL_balance.sec_CL_max_VLM")
        self.connect("sectional_CL_balance.alpha", "VLM.fltcond|alpha")

