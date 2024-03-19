"""
@File    :   CLmax_jet_transport.py
@Date    :   2023/04/11
@Author  :   Eytan Adler
@Description : Max lift coefficient estimate using critical section method
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import warnings

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
from openconcept.utilities import AddSubtractComp

# ==============================================================================
# Extension modules
# ==============================================================================
from openconcept.aerodynamics.openaerostruct import VLM


class CLmaxCriticalSectionVLM(om.Group):
    """
    Predict the maximum lift coefficient of the wing by solving for the
    angle of attack where the maximum sectional lift coefficient equals the
    provided maximum lift coefficient of the airfoil. The solution for the
    sectional lift coefficients is done using OpenAeroStruct.

    Inputs
    ------
    ac|aero|airfoil_Cl_max : float
        Maximum 2D lift coefficient of the wing airfoil (scalar or, if vec_Cl_max is set
        to true, vector of length num_y, dimensionless)
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
    vec_Cl_max : bool
        Make the input ac|aero|airfoil_Cl_max a vector of length num_y where each item in the vector is
        a spanwise panel's local maximum lift coefficient, ordered from wing tip to root. This enables
        specification of a varying maximum sectional lift coefficient along the span, which can be used
        to model high lift devices on only a portion of the wing. If this option is False,
        ac|aero|airfoil_Cl_max is a scalar, which represents the maximum airfoil lift coefficient across
        the entire span of the wing, by default False
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
        self.options.declare("vec_Cl_max", default=False, types=bool, desc="Make ac|aero|airfoil_Cl_max input a vector")
        self.options.declare("surf_options", default=None, desc="Dictionary of OpenAeroStruct surface options")
        self.options.declare("rho", default=200, types=(float, int), desc="Sectional CL aggregation factor")

    def setup(self):
        Cl_max_shape = self.options["num_y"] if self.options["vec_Cl_max"] else 1

        # -------------- Simulate the wing in OpenAeroStruct --------------
        aero = om.Group()
        aero.add_subsystem(
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
        aero.set_input_defaults("VLM.fltcond|alpha", 5, units="deg")

        # -------------- Compute and aggregate Cl - Clmax across the span (Cl - Clmax should be <= 0) --------------
        aero.add_subsystem(
            "Cl_max_limit",
            AddSubtractComp(
                output_name="Cl_limit_vec",
                input_names=["Cl", "Cl_max"],
                vec_size=[self.options["num_y"], Cl_max_shape],
                scaling_factors=[1, -1],
            ),
            promotes_inputs=[("Cl_max", "ac|aero|airfoil_Cl_max")],
        )
        aero.connect("VLM.sectional_CL", "Cl_max_limit.Cl")

        aero.add_subsystem("max_limit", om.KSComp(width=self.options["num_y"], rho=self.options["rho"]))
        aero.connect("Cl_max_limit.Cl_limit_vec", "max_limit.g")

        self.add_subsystem("aero", aero, promotes=["*"])

        # -------------- Solve for the angle of attack that makes max(Cl - Clmax) = 0 --------------
        self.add_subsystem(
            "sectional_CL_balance",
            om.BalanceComp("alpha", lhs_name="max_Cl_limit_VLM", rhs_val=0.0, val=5, units="deg"),
        )
        self.connect("max_limit.KS", "sectional_CL_balance.max_Cl_limit_VLM")
        self.connect("sectional_CL_balance.alpha", "VLM.fltcond|alpha")

        # -------------- Solver setup --------------
        # Use the Schur solver if it's available, otherwise this will be very expensive
        try:
            self.nonlinear_solver = om.NonlinearSchurSolver(
                groupNames=["aero", "sectional_CL_balance"], iprint=2, solve_subsystems=True
            )
            self.linear_solver = om.LinearSchur()
        except AttributeError:
            warnings.warn(
                "OpenMDAO NonlinearSchurSolver is not available, CLmaxCriticalSectionVLM will be very slow!",
                stacklevel=2,
            )

            # Add the Newton solver
            self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, maxiter=10)
            self.linear_solver = om.DirectSolver()
