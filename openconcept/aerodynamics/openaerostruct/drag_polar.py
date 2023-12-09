import numpy as np
import openmdao.api as om
from time import time
from copy import copy, deepcopy
import multiprocessing as mp

# Atmospheric calculations
from openconcept.atmospherics import TemperatureComp, PressureComp, DensityComp, SpeedOfSoundComp
from openconcept.aerodynamics.openaerostruct import (
    TrapezoidalPlanformMesh,
    SectionPlanformMesh,
    ThicknessChordRatioInterp,
    SectionLinearInterp,
)

# Progress bar
progress_bar = True
try:
    import tqdm
except ImportError:
    print('Progress bar for training data can be enabled by installing the tqdm Python package with "pip install tqdm"')
    progress_bar = False

# OpenAeroStruct
try:
    from openaerostruct.geometry.geometry_mesh_transformations import Rotate
    from openaerostruct.aerodynamics.aero_groups import AeroPoint
except ImportError:
    raise ImportError("OpenAeroStruct must be installed to use the VLMDragPolar component")

CITATION = """
@article{Adler2022d,
    author = {Adler, Eytan J. and Martins, Joaquim R. R. A.},
    doi = {10.2514/1.c037096},
    issn = {1533-3868},
    journal = {Journal of Aircraft},
    month = {December},
    publisher = {American Institute of Aeronautics and Astronautics},
    title = {Efficient Aerostructural Wing Optimization Considering Mission Analysis},
    year = {2022}
}
"""


class VLMDragPolar(om.Group):
    """
    Drag polar generated using OpenAeroStruct's vortex lattice method and a surrogate
    model to decrease the computational cost. It allows for three planform definitions.
    The first is a trapezoidal planform defined by aspect ratio, taper, and sweep. The
    second builds a planform from sections definitions, each defined by a streamwise
    offset of the leading edge, spanwise position, and chord length. The mesh is linearly
    lofted between the sections. Finally, this component can take in a mesh directly to
    enable more flexibility. This component enables twisting of the mesh as well.

    Notes
    -----
    Twist is ordered starting at the tip and moving to the root; a twist
    of [-1, 0, 1] would have a tip twist of -1 deg and root twist of 1 deg.
    This is the same ordering as for the section definitions if the geometry
    option is set to \"section\".

    Set the OMP_NUM_THREADS environment variable to 1 for much better parallel training performance!

    Make sure that the wing reference area provided to this group is consistent both with the reference
    area used by OpenAeroStruct (projected area by default) AND the reference area used by OpenConcept
    to compute the lift coefficient it provides.

    Inputs
    ------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    fltcond|M : float
        Mach number (vector, dimensionless)
    fltcond|h : float
        Altitude (vector, m)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
    ac|geom|wing|S_ref : float
        Full planform area (scalar, m^2)
    ac|aero|CD_nonwing : float
        Drag coefficient of components other than the wing; e.g. fuselage,
        tail, interference drag, etc.; this value is simply added to the
        drag coefficient computed by OpenAeroStruct (scalar, dimensionless)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)
        NOTE: fltcond|TempIncrement is a scalar in this component but a vector in OC. \
            This is the case because of the way the VLMDataGen component is set up. To \
            make it work, TempIncrement would need to be an input to the surrogate, \
            which is not worth the extra training cost (at minimum a 2x increase).

        If geometry option is \"trapezoidal\" (the default)
            - **ac|geom|wing|AR** *(float)* - Aspect ratio (scalar, dimensionless)
            - **ac|geom|wing|taper** *(float)* - Taper ratio (must be >0 and <=1); tip chord / root chord (scalar, dimensionless)
            - **ac|geom|wing|c4sweep** *(float)* - Quarter chord sweep angle (scalar, degrees)
            - **ac|geom|wing|toverc** *(float)* - Wing tip and wing root airfoil thickness to chord ratio in that order; panel
              thickness to chord ratios are linearly interpolated between the tip and root (vector of length 2, dimensionless)
            - **ac|geom|wing|twist** *(float)* - Twist at spline control points, ordered from tip to root (vector of length num_twist, degrees)

        If geometry option is \"section\" (despite inputs in m, they are scaled to provided planform area)
            - **ac|geom|wing|x_LE_sec** *(float)* - Streamwise offset of the section's leading edge, starting with the outboard
              section (wing tip) and moving inboard toward the root (vector of length
              num_sections, m)
            - **ac|geom|wing|y_sec** *(float)* - Spanwise location of each section, starting with the outboard section (wing
              tip) at the MOST NEGATIVE y value and moving inboard (increasing y value)
              toward the root; the user does not provide a value for the root because it
              is always 0.0 (vector of length num_sections - 1, m)
            - **ac|geom|wing|chord_sec** *(float)* - Chord of each section, starting with the outboard section (wing tip) and
              moving inboard toward the root (vector of length num_sections, m)
            - **ac|geom|wing|toverc** *(float)* - Thickness to chord ratio of the airfoil at each defined section starting at the wing tip
              and moving wing root airfoil; panel thickness to chord ratios is linearly interpolated (vector of length num_sections, m)
            - **ac|geom|wing|twist** *(float)* - Twist at each section, ordered from tip to root (vector of length num_sections, degrees)

        If geometry option is \"mesh\"
            - **ac|geom|wing|OAS_mesh** *(float)* - OpenAeroStruct 3D mesh (num_x + 1 x num_y + 1 x 3 vector, m)
            - **ac|geom|wing|toverc** *(float)* - Thickness to chord ratio of each streamwise strip of panels ordered from
              wing tip to wing root (vector of length num_y, dimensionless)

    Outputs
    -------
    drag : float
        Drag force (vector, Newtons)
    twisted_mesh : float
        OpenAeroStruct mesh that has been twisted according to twist inputs; only an output if geometry options
        is set to \"trapezoidal\" or \"section\" (num_x + 1 x num_y + 1 x 3 vector, m)

    Options
    -------
    num_nodes : int
        Number of analysis points per mission segment (scalar, dimensionless)
    geometry : float
        Choose the geometry parameterization from the following options (by default trapezoidal):

            - "trapezoidal": Create a trapezoidal mesh based on apsect ratio, taper, and sweep
            - "section": Create a mesh lofted between defined sections
            - "mesh": Pass in a mesh directly to this component

    num_x : int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of panels in y (spanwise) direction for half wing. If geometry option
        is set to \"section\", this value represents the number of panels between each
        pair of sections. In that case, it can also be a vector with potentially a different
        number of panels between each pair of sections (scalar or vector, dimensionless)
    num_sections : int
        Only if geometry option is \"section\", this represents the number of spanwise sections
        to define planform shape. This parameter is ignored for other geometry options (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist, only used if geometry is set to \"trapezoidal\" because
        \"mesh\" linearly interpolates twist between sections and \"mesh\" does not provide twist
        functionality (scalar, dimensionless)
    alpha_train : list or ndarray
        List of angle of attack values at which to train the model (ndarray, degrees)
    Mach_train : list or ndarray
        List of Mach numbers at which to train the model (ndarray, dimensionless)
    alt_train : list or ndarray
        List of altitude values at which to train the model (ndarray, m)
    surf_options : dict
        Dictionary of OpenAeroStruct surface options; any options provided here
        will override the default ones; see the OpenAeroStruct documentation for more information.
        Because the geometry transformations are excluded in this model (to simplify the interface),
        the <transformation>_cp options are not supported. The t_over_c option is also removed since
        it is provided instead as an input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points to run")
        self.options.declare(
            "geometry",
            default="trapezoidal",
            values=["trapezoidal", "section", "mesh"],
            desc="OpenAeroStruct mesh parameterization",
        )
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare(
            "num_y", default=6, types=(int, list, tuple, np.ndarray), desc="Number of spanwise (half wing) mesh panels"
        )
        self.options.declare(
            "num_sections", default=2, types=int, desc="Number of sections along the half span to define"
        )
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare(
            "Mach_train",
            default=np.array([0.1, 0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]),
            types=(list, np.ndarray),
            desc="List of Mach number training values (dimensionless)",
        )
        self.options.declare(
            "alpha_train",
            default=np.linspace(-10, 15, 6),
            types=(list, np.ndarray),
            desc="List of angle of attack training values (degrees)",
        )
        self.options.declare(
            "alt_train",
            default=np.linspace(0, 12e3, 4),
            types=(list, np.ndarray),
            desc="List of altitude training values (meters)",
        )
        self.options.declare(
            "surf_options", default={}, types=(dict), desc="Dictionary of OpenAeroStruct surface options"
        )

    def setup(self):
        nn = self.options["num_nodes"]
        nx = self.options["num_x"]
        ny = self.options["num_y"]
        n_twist = self.options["num_twist"]
        geo = self.options["geometry"]
        n_alpha = self.options["alpha_train"].size
        n_Mach = self.options["Mach_train"].size
        n_alt = self.options["alt_train"].size

        # Get the total number of y coordinates if it's not a scalar
        if geo == "section" and isinstance(ny, int):
            ny_tot = (self.options["num_sections"] - 1) * ny + 1
        elif not isinstance(ny, int):
            ny_tot = np.sum(ny) + 1
        else:
            ny_tot = ny + 1

        # Generate the mesh if geometry parameterization says we should and twist it
        if geo == "trapezoidal":
            if not isinstance(ny, int):
                raise ValueError("The num_y option must be an integer if the geometry option is trapezoidal")

            self.add_subsystem(
                "gen_mesh",
                TrapezoidalPlanformMesh(num_x=nx, num_y=ny),
                promotes_inputs=[
                    ("S", "ac|geom|wing|S_ref"),
                    ("AR", "ac|geom|wing|AR"),
                    ("taper", "ac|geom|wing|taper"),
                    ("sweep", "ac|geom|wing|c4sweep"),
                ],
                promotes_outputs=[("mesh", "ac|geom|wing|OAS_mesh")],
            )

            # Compute the twist at mesh y-coordinates
            comp = self.add_subsystem(
                "twist_bsp",
                om.SplineComp(
                    method="bsplines",
                    x_interp_val=np.linspace(0, 1, ny_tot),
                    num_cp=n_twist,
                    interp_options={"order": min(n_twist, 4)},
                ),
                promotes_inputs=[("twist_cp", "ac|geom|wing|twist")],
            )
            comp.add_spline(y_cp_name="twist_cp", y_interp_name="twist", y_units="deg")
            self.set_input_defaults("ac|geom|wing|twist", np.zeros(n_twist), units="deg")

            # Apply twist spline to mesh
            self.add_subsystem(
                "twist_mesh",
                Rotate(val=np.zeros(ny_tot), mesh_shape=(nx + 1, ny_tot, 3), symmetry=True),
                promotes_inputs=[("in_mesh", "ac|geom|wing|OAS_mesh")],
                promotes_outputs=[("mesh", "twisted_mesh")],
            )
            self.connect("twist_bsp.twist", "twist_mesh.twist")
        elif geo == "section":
            self.add_subsystem(
                "gen_mesh",
                SectionPlanformMesh(num_x=nx, num_y=ny, num_sections=self.options["num_sections"], scale_area=True),
                promotes_inputs=[
                    ("S", "ac|geom|wing|S_ref"),
                    ("x_LE_sec", "ac|geom|wing|x_LE_sec"),
                    ("y_sec", "ac|geom|wing|y_sec"),
                    ("chord_sec", "ac|geom|wing|chord_sec"),
                ],
                promotes_outputs=[("mesh", "ac|geom|wing|OAS_mesh")],
            )

            # Compute the twist at mesh y-coordinates
            self.add_subsystem(
                "twist_sec",
                SectionLinearInterp(num_y=ny, num_sections=self.options["num_sections"], units="deg", cos_spacing=True),
                promotes_inputs=[("property_sec", "ac|geom|wing|twist")],
            )
            self.set_input_defaults("ac|geom|wing|twist", np.zeros(self.options["num_sections"]), units="deg")

            # Apply twist spline to mesh
            self.add_subsystem(
                "twist_mesh",
                Rotate(val=np.zeros(ny_tot), mesh_shape=(nx + 1, ny_tot, 3), symmetry=True),
                promotes_inputs=[("in_mesh", "ac|geom|wing|OAS_mesh")],
                promotes_outputs=[("mesh", "twisted_mesh")],
            )
            self.connect("twist_sec.property_node", "twist_mesh.twist")

        # Interpolate the thickness to chord ratios for each panel
        if geo in ["trapezoidal", "section"]:
            n_sections = self.options["num_sections"] if geo == "section" else 2
            self.add_subsystem(
                "t_over_c_interp",
                ThicknessChordRatioInterp(num_y=ny, num_sections=n_sections),
                promotes_inputs=[("section_toverc", "ac|geom|wing|toverc")],
            )

        # Inputs to promote from the calls to OpenAeroStruct (if geometry is "mesh",
        # promote the thickness-to-chord ratio directly)
        VLM_promote_inputs = ["ac|aero|CD_nonwing", "fltcond|TempIncrement"]
        if geo == "mesh":
            VLM_promote_inputs += ["ac|geom|wing|toverc", "ac|geom|wing|OAS_mesh"]
        else:
            self.connect("t_over_c_interp.panel_toverc", "training_data.ac|geom|wing|toverc")
            self.connect("twisted_mesh", "training_data.ac|geom|wing|OAS_mesh")

        # Training data
        self.add_subsystem(
            "training_data",
            VLMDataGen(
                num_x=nx,
                num_y=ny_tot - 1,
                alpha_train=self.options["alpha_train"],
                Mach_train=self.options["Mach_train"],
                alt_train=self.options["alt_train"],
                surf_options=self.options["surf_options"],
            ),
            promotes_inputs=VLM_promote_inputs,
        )

        # Surrogate model
        interp = om.MetaModelStructuredComp(
            method="scipy_cubic", training_data_gradients=True, vec_size=nn, extrapolate=True
        )
        interp.add_input("fltcond|M", 0.1, training_data=self.options["Mach_train"])
        interp.add_input("alpha", 0.0, units="deg", training_data=self.options["alpha_train"])
        interp.add_input("fltcond|h", 0.0, units="m", training_data=self.options["alt_train"])
        interp.add_output("CL", training_data=np.zeros((n_Mach, n_alpha, n_alt)))
        interp.add_output("CD", training_data=np.zeros((n_Mach, n_alpha, n_alt)))
        self.add_subsystem("aero_surrogate", interp, promotes_inputs=["fltcond|M", "fltcond|h"])
        self.connect("training_data.CL_train", "aero_surrogate.CL_train")
        self.connect("training_data.CD_train", "aero_surrogate.CD_train")

        # Solve for angle of attack that meets input lift coefficient
        self.add_subsystem(
            "alpha_bal",
            om.BalanceComp(
                "alpha", eq_units=None, lhs_name="CL_VLM", rhs_name="fltcond|CL", val=np.ones(nn), units="deg"
            ),
            promotes_inputs=["fltcond|CL"],
        )
        self.connect("alpha_bal.alpha", "aero_surrogate.alpha")
        self.connect("aero_surrogate.CL", "alpha_bal.CL_VLM")

        # Compute drag force from drag coefficient
        self.add_subsystem(
            "drag_calc",
            om.ExecComp(
                "drag = q * S * CD",
                drag={"units": "N", "shape": (nn,)},
                q={"units": "Pa", "shape": (nn,)},
                S={"units": "m**2"},
                CD={"shape": (nn,)},
            ),
            promotes_inputs=[("q", "fltcond|q"), ("S", "ac|geom|wing|S_ref")],
            promotes_outputs=["drag"],
        )
        self.connect("aero_surrogate.CD", "drag_calc.CD")

        # Set input defaults for inputs promoted from different places with different values
        self.set_input_defaults("ac|geom|wing|S_ref", 1.0, units="m**2")


class VLMDataGen(om.ExplicitComponent):
    """
    Generates a grid of OpenAeroStruct lift and drag data to train
    a surrogate model. The grid is defined by the options and the
    mesh geometry by the input. This component will only recalculate
    the lift and drag grid when the mesh changes. All VLMDataGen
    components in the model must use the same training points and surf_options.

    Inputs
    ------
    ac|geom|wing|OAS_mesh: float
        OpenAeroStruct 3D mesh (num_x + 1 x num_y + 1 x 3 vector, m)
    ac|geom|wing|toverc : float
        Thickness to chord ratio of each streamwise strip of panels ordered from wing
        tip to wing root; used for the viscous and wave drag calculations
        (vector of length num_y, dimensionless)
    ac|aero|CD_nonwing : float
        Drag coefficient of components other than the wing; e.g. fuselage,
        tail, interference drag, etc.; this value is simply added to the
        drag coefficient computed by OpenAeroStruct (scalar, dimensionless)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)

    Outputs
    -------
    CL_train : 3-dim ndarray
        Grid of lift coefficient data to train structured surrogate model
    CD_train : 3-dim ndarray
        Grid of drag coefficient data to train structured surrogate model

    Options
    -------
    num_x : int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of panels in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    Mach_train : list or ndarray
        List of Mach numbers at which to train the model (ndarray, dimensionless)
    alpha_train : list or ndarray
        List of angle of attack values at which to train the model (ndarray, degrees)
    alt_train : list or ndarray
        List of altitude values at which to train the model (ndarray, m)
    surf_options : dict
        Dictionary of OpenAeroStruct surface options; any options provided here
        will override the default ones; see the OpenAeroStruct documentation for more information.
        Because the geometry transformations are excluded in this model (to simplify the interface),
        the <transformation>_cp options are not supported. The t_over_c option is also removed since
        it is provided instead as an input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare("num_y", default=6, desc="Number of spanwise (half wing) mesh panels")
        self.options.declare(
            "Mach_train",
            default=np.array([0.1, 0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]),
            desc="List of Mach number training values (dimensionless)",
        )
        self.options.declare(
            "alpha_train", default=np.linspace(-10, 15, 6), desc="List of angle of attack training values (degrees)"
        )
        self.options.declare(
            "alt_train", default=np.linspace(0, 12e3, 4), desc="List of altitude training values (meters)"
        )
        self.options.declare("surf_options", default={}, desc="Dictionary of OpenAeroStruct surface options")

    def setup(self):
        nx = self.options["num_x"]
        ny = self.options["num_y"]

        self.add_input("ac|geom|wing|OAS_mesh", units="m", shape=(nx + 1, ny + 1, 3))
        self.add_input("ac|geom|wing|toverc", shape=(ny,), val=0.12)
        self.add_input("ac|aero|CD_nonwing", val=0.0)
        self.add_input("fltcond|TempIncrement", val=0.0, units="degC")

        n_Mach = self.options["Mach_train"].size
        n_alpha = self.options["alpha_train"].size
        n_alt = self.options["alt_train"].size
        self.add_output("CL_train", shape=(n_Mach, n_alpha, n_alt))
        self.add_output("CD_train", shape=(n_Mach, n_alpha, n_alt))

        self.declare_partials("*", "*")

        # Check that the surf_options dictionary does not differ
        # from other instances of the VLMDataGen object
        if hasattr(VLMDataGen, "surf_options"):
            error = False
            if VLMDataGen.surf_options.keys() != self.options["surf_options"].keys():
                error = True
            for key in VLMDataGen.surf_options.keys():
                if isinstance(VLMDataGen.surf_options[key], np.ndarray):
                    error = error or np.any(VLMDataGen.surf_options[key] != self.options["surf_options"][key])
                else:
                    error = error or VLMDataGen.surf_options[key] != self.options["surf_options"][key]
            if error:
                raise ValueError(
                    "The VLMDataGen and VLMDragPolar components do not support\n"
                    "differently-valued surf_options within an OpenMDAO model. Trying to replace:\n"
                    f"{VLMDataGen.surf_options}\n"
                    f"with new options:\n{self.options['surf_options']}"
                )
        else:
            VLMDataGen.surf_options = deepcopy(self.options["surf_options"])

        # Generate grids and default cached values for training inputs and outputs
        VLMDataGen.mesh = -np.ones((nx + 1, ny + 1, 3))
        VLMDataGen.toverc = -np.ones(ny)
        VLMDataGen.temp_incr = -42 * np.ones(1)
        VLMDataGen.Mach, VLMDataGen.alpha, VLMDataGen.alt = np.meshgrid(
            self.options["Mach_train"], self.options["alpha_train"], self.options["alt_train"], indexing="ij"
        )
        VLMDataGen.CL = np.zeros((n_Mach, n_alpha, n_alt))
        VLMDataGen.CD = np.zeros((n_Mach, n_alpha, n_alt))
        VLMDataGen.partials = None

    def compute(self, inputs, outputs):
        mesh = inputs["ac|geom|wing|OAS_mesh"]
        toverc = inputs["ac|geom|wing|toverc"]
        CD_nonwing = inputs["ac|aero|CD_nonwing"]
        temp_incr = inputs["fltcond|TempIncrement"]

        # If the inputs are unchaged, use the previously calculated values
        tol = 1e-13  # floating point comparison tolerance
        if (
            np.all(np.abs(mesh - VLMDataGen.mesh) < tol)
            and np.all(np.abs(toverc - VLMDataGen.toverc) < tol)
            and np.abs(temp_incr - VLMDataGen.temp_incr) < tol
        ):
            outputs["CL_train"] = VLMDataGen.CL
            outputs["CD_train"] = VLMDataGen.CD + CD_nonwing
            return

        # Copy new values to cached ones
        VLMDataGen.mesh[:, :, :] = mesh
        VLMDataGen.toverc[:] = toverc
        VLMDataGen.temp_incr[:] = temp_incr

        # Compute new training values
        train_in = {}
        train_in["Mach_number_grid"] = VLMDataGen.Mach
        train_in["alpha_grid"] = VLMDataGen.alpha
        train_in["alt_grid"] = VLMDataGen.alt
        train_in["TempIncrement"] = temp_incr
        train_in["mesh"] = mesh
        train_in["toverc"] = toverc
        train_in["num_x"] = self.options["num_x"]
        train_in["num_y"] = self.options["num_y"]

        data = compute_training_data(train_in, surf_dict=self.options["surf_options"])
        VLMDataGen.CL[:] = data["CL"]
        VLMDataGen.CD[:] = data["CD"]
        VLMDataGen.partials = copy(data["partials"])
        outputs["CL_train"] = VLMDataGen.CL
        outputs["CD_train"] = VLMDataGen.CD + CD_nonwing

    def compute_partials(self, inputs, partials):
        # Compute partials if they haven't been already and return them
        self.compute(inputs, {})
        for key, value in VLMDataGen.partials.items():
            partials[key][:] = value
        partials["CD_train", "ac|aero|CD_nonwing"] = np.ones(VLMDataGen.CD.shape)


"""
Generates training data and its total derivatives by
calling OpenAeroStruct at each training point.

Parameters
----------
inputs : dict
    A dictionary containing the following entries:
    Mach_number_grid : mdarray
        Mach number (3D meshgrid 'ij'-style ndarray, dimensionless)
    alpha_grid : ndarray
        Angle of attack (3D meshgrid 'ij'-style ndarray, degrees)
    alt_grid : ndarray
        Altitude (3D meshgrid 'ij'-style ndarray, m)
    TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)
    mesh: ndarray
        OpenAeroStruct 3D mesh (num_x + 1 x num_y + 1 x 3 ndarray, m)
    toverc : float
        Thickness to chord ratio of each streamwise strip of panels ordered from wing
        tip to wing root; used for the viscous and wave drag calculations
        (vector of length num_y, dimensionless)
    num_x: int
        number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of panels in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
surf_dict : dict
    Dictionary of OpenAeroStruct surface options; any options provided here
    will override the default ones; see the OpenAeroStruct documentation for more information.
    Because the geometry transformations are excluded in this model (to simplify the interface),
    the <transformation>_cp options are not supported. The t_over_c option is also removed since
    it is provided instead as an input.

Returns
-------
data : dict
    A dictionary containing the following entries:
    CL : ndarray
        Lift coefficients at training points (3D meshgrid 'ij'-style ndarray, dimensionless)
    CD : ndarray
        Drag coefficients at training points (3D meshgrid 'ij'-style ndarray, dimensionless)
    partials : dict
        Partial derivatives of the training data flattened in the proper OpenMDAO-style
        format for use as partial derivatives in the VLMDataGen component
"""


def compute_training_data(inputs, surf_dict=None):
    t_start = time()
    print("Generating OpenAeroStruct aerodynamic training data...")

    # Set up test points for use in parallelized map function ([Mach, alpha, altitude, inputs] for each point)
    test_points = np.array(
        [
            inputs["Mach_number_grid"].flatten(),
            inputs["alpha_grid"].flatten(),
            inputs["alt_grid"].flatten(),
            np.zeros(inputs["Mach_number_grid"].size),
        ]
    ).T.tolist()
    inputs_to_send = {"surf_dict": surf_dict}
    keys = ["TempIncrement", "mesh", "toverc", "num_x", "num_y"]
    for key in keys:
        inputs_to_send[key] = inputs[key]
    for row in test_points:
        row[-1] = inputs_to_send

    # Initialize the parallel pool and compute the OpenAeroStruct data
    with mp.Pool() as parallel_pool:
        if progress_bar:
            out = list(tqdm.tqdm(parallel_pool.imap(compute_aerodynamic_data, test_points), total=len(test_points)))
        else:
            out = list(parallel_pool.map(compute_aerodynamic_data, test_points))

    # Initialize output arrays
    CL = np.zeros(inputs["Mach_number_grid"].shape)
    CD = np.zeros(inputs["Mach_number_grid"].shape)
    jac_num_rows = inputs["Mach_number_grid"].size  # product of array dimensions
    mesh_jac_num_cols = inputs["mesh"].size
    num_y_panels = inputs["mesh"].shape[1] - 1
    of = ["CL_train", "CD_train"]
    partials = {}
    for f in of:
        partials[f, "ac|geom|wing|OAS_mesh"] = np.zeros((jac_num_rows, mesh_jac_num_cols))
        partials[f, "ac|geom|wing|toverc"] = np.zeros((jac_num_rows, num_y_panels))
        partials[f, "fltcond|TempIncrement"] = np.zeros((jac_num_rows, 1))
    data = {"CL": CL, "CD": CD, "partials": partials}

    # Transfer data into output data structure the proper format
    for i in range(len(out)):
        data["CL"][np.unravel_index(i, inputs["Mach_number_grid"].shape)] = out[i]["CL"]
        data["CD"][np.unravel_index(i, inputs["Mach_number_grid"].shape)] = out[i]["CD"]
        for key in data["partials"].keys():
            data["partials"][key][i] = out[i]["partials"][key]

    print(f"        ...done in {time() - t_start} sec\n")

    return data


# Function to compute CL, CD, and derivatives at a given test point. Used for
# the parallel mapping function in compute_training_data
# Input "point" is row in test_points array
def compute_aerodynamic_data(point):
    inputs = point[3]

    # Set up OpenAeroStruct problem
    p = om.Problem(reports=False)
    p.model.add_subsystem(
        "aero_analysis",
        VLM(
            num_x=inputs["num_x"],
            num_y=inputs["num_y"],
            surf_options=inputs["surf_dict"],
        ),
        promotes=["*"],
    )

    p.setup(mode="rev")

    # Set variables
    p.set_val("fltcond|TempIncrement", val=inputs["TempIncrement"], units="degC")
    p.set_val("ac|geom|wing|OAS_mesh", val=inputs["mesh"], units="m")
    p.set_val("ac|geom|wing|toverc", val=inputs["toverc"])
    p.set_val("fltcond|M", point[0])
    p.set_val("fltcond|alpha", point[1], units="deg")
    p.set_val("fltcond|h", point[2], units="m")

    p.run_model()

    output = {"CL": p.get_val("fltcond|CL"), "CD": p.get_val("fltcond|CD"), "partials": {}}

    # Compute derivatives
    of = ["fltcond|CL", "fltcond|CD"]
    of_out = ["CL_train", "CD_train"]
    wrt = ["ac|geom|wing|OAS_mesh", "ac|geom|wing|toverc", "fltcond|TempIncrement"]

    deriv = p.compute_totals(of, wrt)

    for n, f in enumerate(of):
        for u in wrt:
            output["partials"][of_out[n], u] = np.copy(deriv[f, u])

    return output


class VLM(om.Group):
    """
    Computes lift, drag, and other forces using OpenAeroStruct's vortex lattice implementation.
    This group contains a Reynolds number calculation that uses a linear approximation of dynamic
    viscosity. It is accurate up through ~33k ft and probably ok up to 40k ft, but not much further.

    Thickness to chord ratio of the airfoils is taken from the input, not the value in
    the surf_options dict (which is ignored).

    Inputs
    ------
    fltcond|alpha : float
        Angle of attack (scalar, degrees)
    fltcond|M : float
        Mach number (scalar, dimensionless)
    fltcond|h : float
        Altitude (scalar, m)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)
    ac|geom|wing|OAS_mesh: float
        OpenAeroStruct 3D mesh (num_x + 1 x num_y + 1 x 3 vector, m)
    ac|geom|wing|toverc : float
        Thickness to chord ratio of each streamwise strip of panels ordered from wing
        tip to wing root; used for the viscous and wave drag calculations
        (vector of length num_y, dimensionless)

    Outputs
    -------
    fltcond|CL : float
        Lift coefficient of wing (scalar, dimensionless)
    fltcond|CD : float
        Drag coefficient of wing (scalar, dimensionless)
    fltcond|CDi : float
        Induced drag component (scalar, dimensionless)
    fltcond|CDv : float
        Viscous drag component (scalar, dimensionless)
    fltcond|CDw : float
        Wave drag component (scalar, dimensionless)
    sectional_CL : float
        Sectional lift coefficient for each chordwise panel strip (vector of length num_y, dimensionless)
    panel_forces : float
        Force from VLM for each panel (vector of size (num_x x num_y x 3), N)

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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare("num_y", default=6, desc="Number of spanwise (half wing) mesh panels")
        self.options.declare("surf_options", default=None, desc="Dictionary of OpenAeroStruct surface options")

    def setup(self):
        # Number of coordinates is one more than the number of panels
        nx = int(self.options["num_x"]) + 1
        ny = int(self.options["num_y"]) + 1

        # =================================================================
        #              Compute atmospheric and fluid properties
        # =================================================================
        self.add_subsystem("temp", TemperatureComp(num_nodes=1), promotes_inputs=["fltcond|h", "fltcond|TempIncrement"])
        self.add_subsystem("pressure", PressureComp(num_nodes=1), promotes_inputs=["fltcond|h"])
        self.add_subsystem("density", DensityComp(num_nodes=1))
        self.connect("temp.fltcond|T", "density.fltcond|T")
        self.connect("pressure.fltcond|p", "density.fltcond|p")
        self.add_subsystem("sound_speed", SpeedOfSoundComp(num_nodes=1))
        self.connect("temp.fltcond|T", "sound_speed.fltcond|T")
        self.add_subsystem(
            "airspeed",
            om.ExecComp("Utrue = Mach * a", Utrue={"units": "m/s", "val": 200.0}, a={"units": "m/s", "val": 300.0}),
            promotes_inputs=[("Mach", "fltcond|M")],
        )
        self.connect("sound_speed.fltcond|a", "airspeed.a")

        # Compute dimensionalized Reynolds number (use linear interpolation from standard atmosphere up
        # to 35k ft to estimate dynamic viscosity)
        self.add_subsystem(
            "Re_calc",
            om.ExecComp(
                "re = rho * u / (-3.329134*10**(-10) * h + 1.792398*10**(-5))",
                re={"units": "1/m", "val": 1e6},
                rho={"units": "kg/m**3", "val": 1.0},
                u={"units": "m/s", "val": 100.0},
                h={"units": "m", "val": 1.0},
            ),
            promotes_inputs=[("h", "fltcond|h")],
        )
        self.connect("density.fltcond|rho", "Re_calc.rho")
        self.connect("airspeed.Utrue", "Re_calc.u")

        # =================================================================
        #                       Call OpenAeroStruct
        # =================================================================
        # This dummy mesh must be passed to the surface dict so OpenAeroStruct
        # knows the dimensions of the mesh and whether it is a left or right wing
        dummy_mesh = np.zeros((nx, ny, 3))
        dummy_mesh[:, :, 0], dummy_mesh[:, :, 1] = np.meshgrid(
            np.linspace(0, 1, nx), np.linspace(-1, 0, ny), indexing="ij"
        )

        self.surf_dict = {
            "name": "wing",
            "mesh": dummy_mesh,  # this must be defined
            # because the VLMGeometry component uses the shape of the mesh in this
            # dictionary to determine the size of the mesh; the values don't matter
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "projected",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            "CL0": 0.0,  # CL of the surface at alpha=0
            "CD0": 0.0,  # CD of the surface at alpha=0
            # Airfoil properties for viscous drag calculation
            "k_lam": 0.05,  # percentage of chord with laminar
            # flow, used for viscous drag
            "t_over_c": np.array([0.12]),  # thickness over chord ratio (NACA SC2-0612)
            "c_max_t": 0.37,  # chordwise location of maximum (NACA SC2-0612)
            # thickness
            "with_viscous": True,  # if true, compute viscous drag
            "with_wave": True,  # if true, compute wave drag
        }

        # Overwrite any options in the surface dict with those provided in the options
        if self.options["surf_options"] is not None:
            for key in self.options["surf_options"]:
                self.surf_dict[key] = self.options["surf_options"][key]

        self.add_subsystem(
            "aero_point",
            AeroPoint(surfaces=[self.surf_dict]),
            promotes_inputs=[
                ("Mach_number", "fltcond|M"),
                ("alpha", "fltcond|alpha"),
                (f"{self.surf_dict['name']}_perf.t_over_c", "ac|geom|wing|toverc"),
            ],
            promotes_outputs=[
                (f"{self.surf_dict['name']}_perf.CD", "fltcond|CD"),
                (f"{self.surf_dict['name']}_perf.CL", "fltcond|CL"),
                (f"{self.surf_dict['name']}_perf.CDi", "fltcond|CDi"),
                (f"{self.surf_dict['name']}_perf.CDv", "fltcond|CDv"),
                (f"{self.surf_dict['name']}_perf.CDw", "fltcond|CDw"),
                (f"{self.surf_dict['name']}_perf.Cl", "sectional_CL"),
                (f"aero_states.{self.surf_dict['name']}_sec_forces", "panel_forces"),
            ],
        )
        self.connect("airspeed.Utrue", "aero_point.v")
        self.connect("density.fltcond|rho", "aero_point.rho")
        self.connect("Re_calc.re", "aero_point.re")

        # Promote the mesh from within OpenAeroStruct
        self.promotes(
            subsys_name="aero_point",
            inputs=[(f"{self.surf_dict['name']}.def_mesh", "ac|geom|wing|OAS_mesh")],
        )
        self.promotes(
            subsys_name="aero_point",
            inputs=[(f"aero_states.{self.surf_dict['name']}_def_mesh", "ac|geom|wing|OAS_mesh")],
        )

        # Set input defaults for inputs that go to multiple locations
        self.set_input_defaults("fltcond|M", 0.1)
        self.set_input_defaults("fltcond|alpha", 0.0, units="deg")
        self.set_input_defaults("ac|geom|wing|OAS_mesh", dummy_mesh, units="m")
        self.set_input_defaults("ac|geom|wing|toverc", np.full(ny - 1, 0.12))  # 12% airfoil thickness by default


# Example usage of the drag polar that compares
# the surrogate to a direction OpenAeroStruct call
# with a very coarse mesh and training point distribution
def example_usage():
    # Define parameters
    nn = 1
    num_x = 2
    num_y = 4
    S = 427.8  # m^2
    AR = 9.82
    taper = 0.149
    sweep = 31.6
    twist = np.array([-1, 1])
    n_twist = twist.size

    M = 0.7
    CL = 0.35
    h = 0  # m

    p = om.Problem()
    p.model.add_subsystem(
        "drag_polar",
        VLMDragPolar(
            num_nodes=nn,
            num_x=num_x,
            num_y=num_y,
            num_twist=n_twist,
            Mach_train=np.linspace(0.1, 0.8, 3),
            alpha_train=np.linspace(-11, 15, 3),
            alt_train=np.linspace(0, 15e3, 2),
        ),
        promotes=["*"],
    )
    p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2)
    p.model.linear_solver = om.DirectSolver()
    p.setup()

    # Set values
    # Geometry
    p.set_val("ac|geom|wing|S_ref", S, units="m**2")
    p.set_val("ac|geom|wing|AR", AR)
    p.set_val("ac|geom|wing|taper", taper)
    p.set_val("ac|geom|wing|c4sweep", sweep, units="deg")
    p.set_val("ac|geom|wing|twist", twist, units="deg")
    p.set_val("fltcond|q", 6125.0 * np.ones(nn), units="Pa")
    p.set_val("fltcond|TempIncrement", 0, units="degC")

    # Flight condition
    p.set_val("fltcond|M", M)
    p.set_val("fltcond|CL", CL)
    p.set_val("fltcond|h", h, units="m")

    p.run_model()

    print("================== SURROGATE ==================")
    print(f"CL: {p.get_val('aero_surrogate.CL')}")
    print(f"CD: {p.get_val('aero_surrogate.CD')}")
    print(f"Alpha: {p.get_val('aero_surrogate.alpha', units='deg')} deg")

    # Call OpenAeroStruct at the same flight condition to compare
    prob = om.Problem()
    prob.model.add_subsystem("model", VLM(num_x=num_x, num_y=num_y), promotes=["*"])

    prob.setup()

    # Set mesh value
    prob.set_val("ac|geom|wing|OAS_mesh", p.get_val("twisted_mesh", units="m"), units="m")

    # Flight condition
    prob.set_val("fltcond|M", M)
    prob.set_val("fltcond|alpha", p.get_val("aero_surrogate.alpha", units="deg"), units="deg")
    prob.set_val("fltcond|h", h, units="m")

    prob.run_model()

    print("\n================== OpenAeroStruct ==================")
    print(f"CL: {prob.get_val('fltcond|CL')}")
    print(f"CD: {prob.get_val('fltcond|CD')}")
    print(f"Alpha: {prob.get_val('fltcond|alpha', units='deg')} deg")


if __name__ == "__main__":
    example_usage()
