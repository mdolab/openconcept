import numpy as np
import openmdao.api as om
from time import time
from copy import copy, deepcopy
import multiprocessing as mp
import warnings

# Atmospheric calculations
from openconcept.atmospherics import TemperatureComp, PressureComp, DensityComp, SpeedOfSoundComp

# Utitilty for vector manipulation
from openconcept.utilities import VectorConcatenateComp

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
    from openaerostruct.integration.aerostruct_groups import AerostructPoint
    from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
    from openaerostruct.structures.wingbox_group import WingboxGroup
    from openconcept.aerodynamics.openaerostruct.mesh_gen import TrapezoidalPlanformMesh
except ImportError:
    raise ImportError("OpenAeroStruct must be installed to use the AerostructDragPolar component")

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


class AerostructDragPolar(om.Group):
    """
    Drag polar and wing weight estimate generated using OpenAeroStruct's
    aerostructural analysis capabilities and a surrogate
    model to decrease the computational cost.

    This component cannot currently handle fuel loads on the wing,
    nor can it handle point loads applied to the structure.

    Notes
    -----
    The spanwise variables (twist, toverc, skin/spar thickness) are ordered
    starting at the tip and moving to the root; a twist of [-1, 0, 1] would
    have a tip twist of -1 deg and root twist of 1 deg

    Set the OMP_NUM_THREADS environment variable to 1 for much better parallel training performance!

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
    ac|geom|wing|AR : float
        Aspect ratio (scalar, dimensionless)
    ac|geom|wing|taper : float
        Taper ratio (must be >0 and <=1); tip chord / root chord (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Quarter chord sweep angle (scalar, degrees)
    ac|geom|wing|twist : float
        List of twist angles at control points of spline (vector, degrees)
        NOTE: length of vector is num_twist (set in options), NOT num_nodes
    ac|geom|wing|toverc : float
        List of thickness to chord ratios at control points of spline (vector, dimensionless)
        NOTE: length of vector is num_toverc (set in options)
    ac|geom|wing|skin_thickness : float
        List of skin thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_skin (set in options)
    ac|geom|wing|spar_thickness : float
        List of spar thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_spar (set in options)
    ac|aero|CD_nonwing : float
        Drag coefficient of components other than the wing; e.g. fuselage,
        tail, interference drag, etc.; this value is simply added to the
        drag coefficient computed by OpenAeroStruct (scalar, dimensionless)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)
        NOTE: fltcond|TempIncrement is a scalar in this component but a vector in OC. \
              This will be the case for the forseeable future because of the way the \
              OASDataGen component is set up. To make it work, TempIncrement would \
              need to be an input to the surrogate, which is not worth the extra \
              training cost (at minimum a 2x increase).

    Outputs
    -------
    drag : float
        Drag force (vector, Newtons)
    failure : float
        KS aggregation quantity obtained by combining the failure criteria
        for each FEM node. Must be < 0 to constrain wingboxes stresses to
        be less than yield stress. Used to simplify the optimization problem by
        reducing the number of constraints (vector, dimensionless)
    ac|weights|W_wing : float
        Weight of the wing (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points per mission segment (scalar, dimensionless)
    num_x : int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of panels in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist (scalar, dimensionless)
    num_toverc : int
        Number of spline control points for thickness to chord ratio (scalar, dimensionless)
    num_skin : int
        Number of spline control points for skin thickness (scalar, dimensionless)
    num_spar : int
        Number of spline control points for spar thickness (scalar, dimensionless)
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
        the <transformation>_cp options are not supported. The input ac|geom|wing|twist is the same
        as modifying the twist_cp option in the surface dictionary. The mesh geometry modification
        is limited to adjusting the input parameters to this component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points to run")
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare("num_y", default=6, desc="Number of spanwise (half wing) mesh panels")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare("num_toverc", default=4, desc="Number of thickness to chord ratio spline control points")
        self.options.declare("num_skin", default=4, desc="Number of skin thickness spline control points")
        self.options.declare("num_spar", default=4, desc="Number of spar thickness spline control points")
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
        nn = self.options["num_nodes"]
        n_alpha = self.options["alpha_train"].size
        n_Mach = self.options["Mach_train"].size
        n_alt = self.options["alt_train"].size

        # Training data
        self.add_subsystem(
            "training_data",
            OASDataGen(
                num_x=self.options["num_x"],
                num_y=self.options["num_y"],
                num_twist=self.options["num_twist"],
                num_toverc=self.options["num_toverc"],
                num_skin=self.options["num_skin"],
                num_spar=self.options["num_spar"],
                alpha_train=self.options["alpha_train"],
                Mach_train=self.options["Mach_train"],
                alt_train=self.options["alt_train"],
                surf_options=self.options["surf_options"],
            ),
            promotes_inputs=[
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                "ac|geom|wing|taper",
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|twist",
                "ac|geom|wing|toverc",
                "ac|geom|wing|skin_thickness",
                "ac|geom|wing|spar_thickness",
                "ac|aero|CD_nonwing",
                "fltcond|TempIncrement",
            ],
            promotes_outputs=[("W_wing", "ac|weights|W_wing")],
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
        interp.add_output("failure", training_data=np.zeros((n_Mach, n_alpha, n_alt)))
        self.add_subsystem(
            "aero_surrogate", interp, promotes_inputs=["fltcond|M", "fltcond|h"], promotes_outputs=["failure"]
        )
        self.connect("training_data.CL_train", "aero_surrogate.CL_train")
        self.connect("training_data.CD_train", "aero_surrogate.CD_train")
        self.connect("training_data.failure_train", "aero_surrogate.failure_train")

        # Solve for angle of attack that meets input lift coefficient
        self.add_subsystem(
            "alpha_bal",
            om.BalanceComp(
                "alpha", eq_units=None, lhs_name="CL_OAS", rhs_name="fltcond|CL", val=np.ones(nn), units="deg"
            ),
            promotes_inputs=["fltcond|CL"],
        )
        self.connect("alpha_bal.alpha", "aero_surrogate.alpha")
        self.connect("aero_surrogate.CL", "alpha_bal.CL_OAS")

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


class OASDataGen(om.ExplicitComponent):
    """
    Generates a grid of OpenAeroStruct lift and drag data to train
    a surrogate model. The grid is defined by the options and the
    planform geometry by the inputs. This component will only recalculate
    the lift and drag grid when the planform shape changes.

    Notes
    -----
    The spanwise variables (twist, toverc, skin/spar thickness) are ordered
    starting at the tip and moving to the root; a twist of [-1, 0, 1] would
    have a tip twist of -1 deg and root twist of 1 deg

    Inputs
    ------
    ac|geom|wing|S_ref : float
        Full planform area (scalar, m^2)
    ac|geom|wing|AR : float
        Aspect ratio (scalar, dimensionless)
    ac|geom|wing|taper : float
        Taper ratio (must be >0 and <=1); tip chord / root chord (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Quarter chord sweep angle (scalar, degrees)
    ac|geom|wing|twist : float
        List of twist angles at control points of spline (vector, degrees)
        NOTE: length of vector is num_twist (set in options)
    ac|geom|wing|toverc : float
        List of thickness to chord ratios at control points of spline (vector, dimensionless)
        NOTE: length of vector is num_toverc (set in options)
    ac|geom|wing|skin_thickness : float
        List of skin thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_skin (set in options)
    ac|geom|wing|spar_thickness : float
        List of spar thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_spar (set in options)
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
    failure_train : float
        Grid of KS structural failure constraint to train structured surrogate; constrain failure < 0
    W_wing : float
        Wing structural weight (scalar, kg)

    Options
    -------
    num_x : int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of panels in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist (scalar, dimensionless)
    num_toverc : int
        Number of spline control points for thickness to chord ratio (scalar, dimensionless)
    num_skin : int
        Number of spline control points for skin thickness (scalar, dimensionless)
    num_spar : int
        Number of spline control points for spar thickness (scalar, dimensionless)
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
        the <transformation>_cp options are not supported. The input ac|geom|wing|twist is the same
        as modifying the twist_cp option in the surface dictionary. The mesh geometry modification
        is limited to adjusting the input parameters to this component.
    regen_tol : float
        Difference in input variable above which to regenerate the training data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare("num_y", default=6, desc="Number of spanwise (half wing) mesh panels")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare("num_toverc", default=4, desc="Number of thickness to chord ratio spline control points")
        self.options.declare("num_skin", default=4, desc="Number of skin thickness spline control points")
        self.options.declare("num_spar", default=4, desc="Number of spar thickness spline control points")
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
        self.options.declare("regen_tol", default=1e-14, desc="Difference in variable beyond which to regenerate data")

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="m**2")
        self.add_input("ac|geom|wing|AR")
        self.add_input("ac|geom|wing|taper")
        self.add_input("ac|geom|wing|c4sweep", units="deg")
        self.add_input(
            "ac|geom|wing|twist",
            val=np.zeros(self.options["num_twist"]),
            shape=(self.options["num_twist"],),
            units="deg",
        )
        self.add_input(
            "ac|geom|wing|toverc", val=np.zeros(self.options["num_toverc"]), shape=(self.options["num_toverc"],)
        )
        self.add_input(
            "ac|geom|wing|skin_thickness",
            val=np.zeros(self.options["num_skin"]),
            shape=(self.options["num_skin"],),
            units="m",
        )
        self.add_input(
            "ac|geom|wing|spar_thickness",
            val=np.zeros(self.options["num_spar"]),
            shape=(self.options["num_spar"],),
            units="m",
        )
        self.add_input("ac|aero|CD_nonwing", val=0.0)
        self.add_input("fltcond|TempIncrement", val=0.0, units="degC")

        n_Mach = self.options["Mach_train"].size
        n_alpha = self.options["alpha_train"].size
        n_alt = self.options["alt_train"].size
        self.add_output("CL_train", shape=(n_Mach, n_alpha, n_alt))
        self.add_output("CD_train", shape=(n_Mach, n_alpha, n_alt))
        self.add_output("failure_train", shape=(n_Mach, n_alpha, n_alt))
        self.add_output("W_wing", units="kg")

        self.declare_partials("*", "*")

        # Check that the surf_options dictionary does not differ
        # from other instances of the OASDataGen object
        if hasattr(OASDataGen, "surf_options"):
            error = False
            if OASDataGen.surf_options.keys() != self.options["surf_options"].keys():
                error = True
            for key in OASDataGen.surf_options.keys():
                if isinstance(OASDataGen.surf_options[key], np.ndarray):
                    error = error or np.any(OASDataGen.surf_options[key] != self.options["surf_options"][key])
                else:
                    error = error or OASDataGen.surf_options[key] != self.options["surf_options"][key]
            if error:
                raise ValueError(
                    "The OASDataGen and AerostructDragPolar components do not support\n"
                    "differently-valued surf_options within an OpenMDAO model. Trying to replace:\n"
                    f"{OASDataGen.surf_options}\n"
                    f"with new options:\n{self.options['surf_options']}"
                )
        else:
            OASDataGen.surf_options = deepcopy(self.options["surf_options"])

        # Generate grids and default cached values for training inputs and outputs
        OASDataGen.S = -np.ones(1)
        OASDataGen.AR = -np.ones(1)
        OASDataGen.taper = -np.ones(1)
        OASDataGen.c4sweep = -np.ones(1)
        OASDataGen.twist = -1 * np.ones((self.options["num_twist"],))
        OASDataGen.toverc = -1 * np.ones((self.options["num_toverc"],))
        OASDataGen.skin = -1 * np.ones((self.options["num_skin"],))
        OASDataGen.spar = -1 * np.ones((self.options["num_spar"],))
        OASDataGen.temp_incr = -42 * np.ones(1)
        OASDataGen.Mach, OASDataGen.alpha, OASDataGen.alt = np.meshgrid(
            self.options["Mach_train"], self.options["alpha_train"], self.options["alt_train"], indexing="ij"
        )
        OASDataGen.CL = np.zeros((n_Mach, n_alpha, n_alt))
        OASDataGen.CD = np.zeros((n_Mach, n_alpha, n_alt))
        OASDataGen.failure = np.zeros((n_Mach, n_alpha, n_alt))
        OASDataGen.W_wing = 0
        OASDataGen.partials = None

    def compute(self, inputs, outputs):
        S = inputs["ac|geom|wing|S_ref"]
        AR = inputs["ac|geom|wing|AR"]
        taper = inputs["ac|geom|wing|taper"]
        sweep = inputs["ac|geom|wing|c4sweep"]
        twist = inputs["ac|geom|wing|twist"]
        toverc = inputs["ac|geom|wing|toverc"]
        skin = inputs["ac|geom|wing|skin_thickness"]
        spar = inputs["ac|geom|wing|spar_thickness"]
        CD_nonwing = inputs["ac|aero|CD_nonwing"]
        temp_incr = inputs["fltcond|TempIncrement"]

        # If the inputs are unchaged, use the previously calculated values
        tol = self.options["regen_tol"]  # floating point comparison tolerance
        if (
            np.abs(S - OASDataGen.S) < tol
            and np.abs(AR - OASDataGen.AR) < tol
            and np.abs(taper - OASDataGen.taper) < tol
            and np.abs(sweep - OASDataGen.c4sweep) < tol
            and np.all(np.abs(twist - OASDataGen.twist) < tol)
            and np.all(np.abs(toverc - OASDataGen.toverc) < tol)
            and np.all(np.abs(skin - OASDataGen.skin) < tol)
            and np.all(np.abs(spar - OASDataGen.spar) < tol)
            and np.abs(temp_incr - OASDataGen.temp_incr) < tol
        ):
            outputs["CL_train"] = OASDataGen.CL
            outputs["CD_train"] = OASDataGen.CD + CD_nonwing
            outputs["failure_train"] = OASDataGen.failure
            outputs["W_wing"] = OASDataGen.W_wing
            return

        print(f"S = {S}; AR = {AR}; taper = {taper}; sweep = {sweep}; twist = {twist};")
        print(f"toverc = {toverc}; skin = {skin}; spar = {spar}; temp_incr = {temp_incr}")

        # Copy new values to cached ones
        OASDataGen.S[:] = S
        OASDataGen.AR[:] = AR
        OASDataGen.taper[:] = taper
        OASDataGen.c4sweep[:] = sweep
        OASDataGen.twist[:] = twist
        OASDataGen.toverc[:] = toverc
        OASDataGen.skin[:] = skin
        OASDataGen.spar[:] = spar
        OASDataGen.temp_incr[:] = temp_incr

        # Compute new training values
        train_in = {}
        train_in["Mach_number_grid"] = OASDataGen.Mach
        train_in["alpha_grid"] = OASDataGen.alpha
        train_in["alt_grid"] = OASDataGen.alt
        train_in["TempIncrement"] = temp_incr
        train_in["S_ref"] = S
        train_in["AR"] = AR
        train_in["taper"] = taper
        train_in["c4sweep"] = sweep
        train_in["twist"] = twist
        train_in["toverc"] = toverc
        train_in["skin"] = skin
        train_in["spar"] = spar
        train_in["num_x"] = self.options["num_x"]
        train_in["num_y"] = self.options["num_y"]

        data = compute_training_data(train_in, surf_dict=self.options["surf_options"])
        OASDataGen.CL[:] = data["CL"]
        OASDataGen.CD[:] = data["CD"]
        OASDataGen.failure[:] = data["failure"]
        OASDataGen.W_wing = data["W_wing"]
        OASDataGen.partials = copy(data["partials"])
        outputs["CL_train"] = OASDataGen.CL
        outputs["CD_train"] = OASDataGen.CD + CD_nonwing
        outputs["failure_train"] = OASDataGen.failure
        outputs["W_wing"] = OASDataGen.W_wing

    def compute_partials(self, inputs, partials):
        # Compute partials if they haven't been already and return them
        self.compute(inputs, {})
        for key, value in OASDataGen.partials.items():
            partials[key][:] = value
        partials["CD_train", "ac|aero|CD_nonwing"] = np.ones(OASDataGen.CD.shape)


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
    S_ref : float
        Wing planform area (scalar, m^2)
    AR : float
        Wing aspect ratio (scalar, dimensionless)
    taper : float
        Wing taper ratio (scalar, dimensionless)
    c4sweep : float
        Wing sweep measured at quarter chord (scalar, degrees)
    twist : float
        List of twist angles at control points of spline (vector, degrees)
        NOTE: length of vector is num_twist (set in options of OASDataGen)
    toverc : float
        List of thickness to chord ratios at control points of spline (vector, dimensionless)
        NOTE: length of vector is num_toverc (set in options of OASDataGen)
    skin : float
        List of skin thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_skin (set in options of OASDataGen)
    spar : float
        List of spar thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_spar (set in options of OASDataGen)
    num_x: int
        number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of panels in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
surf_dict : dict
    Dictionary of OpenAeroStruct surface options; any options provided here
    will override the default ones; see the OpenAeroStruct documentation for more information.
    Because the geometry transformations are excluded in this model (to simplify the interface),
    the <transformation>_cp options are not supported. The input ac|geom|wing|twist is the same
    as modifying the twist_cp option in the surface dictionary. The mesh geometry modification
    is limited to adjusting the input parameters to this component.

Returns
-------
data : dict
    A dictionary containing the following entries:
    CL : ndarray
        Lift coefficients at training points (3D meshgrid 'ij'-style ndarray, dimensionless)
    CD : ndarray
        Drag coefficients at training points (3D meshgrid 'ij'-style ndarray, dimensionless)
    failure : ndarray
        KS structural failure constraint at training points (3D meshgrid 'ij'-style ndarray, dimensionless)
    W_wing : float
        Wing structural weight; same regardless of flight condition (scalar, kg)
    partials : dict
        Partial derivatives of the training data flattened in the proper OpenMDAO-style
        format for use as partial derivatives in the OASDataGen component
"""


def compute_training_data(inputs, surf_dict=None):
    t_start = time()
    print("Generating OpenAeroStruct aerostructural training data...")

    # Set up test points for use in parallelized map function ([Mach, alpha, altitude, inputs] for each point)
    test_points = np.array(
        [
            inputs["Mach_number_grid"].flatten(),
            inputs["alpha_grid"].flatten(),
            inputs["alt_grid"].flatten(),
            np.zeros(inputs["Mach_number_grid"].size),  # inputs_to_send goes here
            np.zeros(inputs["Mach_number_grid"].size),  # compute W_wing goes here
        ]
    ).T.tolist()
    inputs_to_send = {"surf_dict": surf_dict}
    keys = ["TempIncrement", "S_ref", "AR", "taper", "c4sweep", "twist", "toverc", "skin", "spar", "num_x", "num_y"]
    for key in keys:
        inputs_to_send[key] = inputs[key]
    for i, row in enumerate(test_points):
        # Only compute wing weight on the first iteration (identical across flight conditions)
        row[4] = False
        if i == 0:
            row[4] = True
        row[3] = inputs_to_send

    # Catch ComplexWarning from OpenAeroStruct runs since with a full
    # grid it'll come up 100s of times
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=np.ComplexWarning)

        # Initialize the parallel pool and compute the OpenAeroStruct data
        with mp.Pool() as parallel_pool:
            if progress_bar:
                out = list(tqdm.tqdm(parallel_pool.imap(compute_aerodynamic_data, test_points), total=len(test_points)))
            else:
                out = list(parallel_pool.map(compute_aerodynamic_data, test_points))

    # Initialize output arrays
    CL = np.zeros(inputs["Mach_number_grid"].shape)
    CD = np.zeros(inputs["Mach_number_grid"].shape)
    failure = np.zeros(inputs["Mach_number_grid"].shape)
    jac_num_rows = inputs["Mach_number_grid"].size  # product of array dimensions
    of = ["CL_train", "CD_train", "failure_train"]
    wrt = [
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|taper",
        "ac|geom|wing|c4sweep",
        "ac|geom|wing|twist",
        "ac|geom|wing|toverc",
        "ac|geom|wing|skin_thickness",
        "ac|geom|wing|spar_thickness",
        "fltcond|TempIncrement",
    ]
    vec_wrt = {
        "ac|geom|wing|twist": inputs["twist"].size,
        "ac|geom|wing|toverc": inputs["toverc"].size,
        "ac|geom|wing|skin_thickness": inputs["skin"].size,
        "ac|geom|wing|spar_thickness": inputs["spar"].size,
    }
    partials = {}
    for f in of:
        for u in wrt:
            # States that are vectors have different partial shapes
            partials[f, u] = (
                np.zeros((jac_num_rows, vec_wrt[u])) if u in vec_wrt.keys() else np.zeros((jac_num_rows, 1))
            )
    data = {"CL": CL, "CD": CD, "failure": failure, "partials": partials}

    # Transfer data into output data structure the proper format
    for i in range(len(out)):
        data["CL"][np.unravel_index(i, inputs["Mach_number_grid"].shape)] = out[i]["CL"]
        data["CD"][np.unravel_index(i, inputs["Mach_number_grid"].shape)] = out[i]["CD"]
        data["failure"][np.unravel_index(i, inputs["Mach_number_grid"].shape)] = out[i]["failure"]
        for f in of:
            for u in wrt:
                data["partials"][f, u][i] = out[i]["partials"][f, u]

    # Handle wing weight separately since the output has a different shape than the rest
    data["W_wing"] = out[0]["W_wing"]
    for u in wrt:
        # States that are vectors have different partial shapes
        partials["W_wing", u] = np.zeros((vec_wrt[u],)) if u in vec_wrt.keys() else np.zeros((1,))
        partials["W_wing", u][:] = out[0]["partials"]["W_wing", u]

    print(f"done in {time() - t_start} sec")

    return data


# Function to compute CL, CD, failure, wing weight, and derivatives at a
# given test point. Used for the parallel mapping function in
# compute_training_data. Input "point" is row in test_points array.
def compute_aerodynamic_data(point):
    inputs = point[3]
    compute_W_wing = point[4]

    # Set up OpenAeroStruct problem
    p = om.Problem(reports=False)
    p.model.add_subsystem(
        "aero_analysis",
        Aerostruct(
            num_x=inputs["num_x"],
            num_y=inputs["num_y"],
            num_twist=inputs["twist"].size,
            num_toverc=inputs["toverc"].size,
            num_skin=inputs["skin"].size,
            num_spar=inputs["spar"].size,
            surf_options=inputs["surf_dict"],
        ),
        promotes=["*"],
    )

    # Set design variables
    p.model.set_input_defaults("fltcond|TempIncrement", val=inputs["TempIncrement"], units="degC")
    p.model.set_input_defaults("ac|geom|wing|S_ref", val=inputs["S_ref"], units="m**2")
    p.model.set_input_defaults("ac|geom|wing|AR", val=inputs["AR"])
    p.model.set_input_defaults("ac|geom|wing|taper", val=inputs["taper"])
    p.model.set_input_defaults("ac|geom|wing|c4sweep", val=inputs["c4sweep"], units="deg")
    p.model.set_input_defaults("ac|geom|wing|twist", val=inputs["twist"], units="deg")
    p.model.set_input_defaults("ac|geom|wing|toverc", val=inputs["toverc"])
    p.model.set_input_defaults("ac|geom|wing|skin_thickness", val=inputs["skin"], units="m")
    p.model.set_input_defaults("ac|geom|wing|spar_thickness", val=inputs["spar"], units="m")
    p.setup()

    # Silence OpenAeroStruct's NLGBS aerostructural solver
    p.model.aero_analysis.aerostruct_point.coupled.nonlinear_solver.options["iprint"] = 0

    p.set_val("fltcond|M", point[0])
    p.set_val("fltcond|alpha", point[1], units="deg")
    p.set_val("fltcond|h", point[2], units="m")

    p.run_model()

    output = {}
    output["CL"] = p.get_val("fltcond|CL")
    output["CD"] = p.get_val("fltcond|CD")
    output["failure"] = p.get_val("failure")
    if compute_W_wing:
        output["W_wing"] = p.get_val("ac|weights|W_wing")

    # Compute derivatives
    output["partials"] = {}
    of = ["fltcond|CL", "fltcond|CD", "failure"]
    of_out = ["CL_train", "CD_train", "failure_train"]
    if compute_W_wing:
        of += ["ac|weights|W_wing"]
        of_out += ["W_wing"]
    wrt = [
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|taper",
        "ac|geom|wing|c4sweep",
        "ac|geom|wing|twist",
        "ac|geom|wing|toverc",
        "ac|geom|wing|skin_thickness",
        "ac|geom|wing|spar_thickness",
        "fltcond|TempIncrement",
    ]
    deriv = p.compute_totals(of, wrt)
    for n, f in enumerate(of):
        for u in wrt:
            output["partials"][of_out[n], u] = np.copy(deriv[f, u])

    return output


class Aerostruct(om.Group):
    """
    Perform a coupled aerostructural analysis using OpenAeroStruct.
    This component currently does not support distributed fuel loads
    or point loads added to the structure.

    Notes
    -----
    The spanwise variables (twist, toverc, skin/spar thickness) are ordered
    starting at the tip and moving to the root; a twist of [-1, 0, 1] would
    have a tip twist of -1 deg and root twist of 1 deg

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
    ac|geom|wing|S_ref : float
        Wing planform area (scalar, m^2)
    ac|geom|wing|AR : float
        Wing aspect ratio (scalar, dimensionless)
    ac|geom|wing|taper : float
        Wing taper ratio (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Wing sweep measured at quarter chord (scalar, degrees)
    ac|geom|wing|twist : float
        List of twist angles at control points of spline (vector, degrees)
        NOTE: length of vector is num_twist (set in options)
    ac|geom|wing|toverc : float
        List of thickness to chord ratios at control points of spline (vector, dimensionless)
        NOTE: length of vector is num_toverc (set in options)
    ac|geom|wing|skin_thickness : float
        List of skin thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_skin (set in options)
    ac|geom|wing|spar_thickness : float
        List of spar thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_spar (set in options)

    Outputs
    -------
    fltcond|CL : float
        Lift coefficient of wing (scalar, dimensionless)
    fltcond|CD : float
        Drag coefficient of wing (scalar, dimensionless)
    failure : float
        KS structural failure constraint; constrain failure < 0 (scalar, dimensionless)
    ac|weights|W_wing : float
        Wing structural weight (scalar, kg)

    Options
    -------
    num_x : int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of panels in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist (scalar, dimensionless)
    num_toverc : int
        Number of spline control points for thickness to chord ratio (scalar, dimensionless)
    num_skin : int
        Number of spline control points for skin thickness (scalar, dimensionless)
    num_spar : int
        Number of spline control points for spar thickness (scalar, dimensionless)
    surf_options : dict
        Dictionary of OpenAeroStruct surface options; any options provided here
        will override the default ones; see the OpenAeroStruct documentation for more information.
        Because the geometry transformations are excluded in this model (to simplify the interface),
        the <transformation>_cp options are not supported. The input ac|geom|wing|twist is the same
        as modifying the twist_cp option in the surface dictionary. The mesh geometry modification
        is limited to adjusting the input parameters to this component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare("num_y", default=6, desc="Number of spanwise (half wing) mesh panels")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare("num_toverc", default=4, desc="Number of thickness to chord ratio spline control points")
        self.options.declare("num_skin", default=4, desc="Number of skin thickness spline control points")
        self.options.declare("num_spar", default=4, desc="Number of spar thickness spline control points")
        self.options.declare("surf_options", default=None, desc="Dictionary of OpenAeroStruct surface options")

    def setup(self):
        # Number of coordinates is one more than the number of panels
        nx = int(self.options["num_x"]) + 1
        ny = int(self.options["num_y"]) + 1

        n_twist = int(self.options["num_twist"])
        n_skin = int(self.options["num_skin"])
        n_spar = int(self.options["num_spar"])
        n_toverc = int(self.options["num_toverc"])

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
        #                       Setup OpenAeroStruct
        # =================================================================
        # Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray
        # with dtype=complex (to work with the complex-step derivative approximation). These should
        # be for an airfoil with the chord scaled to 1. We use the 10% to 60% portion of the NACA
        # SC2-0612 airfoil for this case. We use the coordinates available from airfoiltools.com.
        # Using such a large number of coordinates is not necessary. The first and last x-coordinates
        # of the upper and lower surfaces must be the same.
        upper_x = np.array(
            [
                0.1,
                0.11,
                0.12,
                0.13,
                0.14,
                0.15,
                0.16,
                0.17,
                0.18,
                0.19,
                0.2,
                0.21,
                0.22,
                0.23,
                0.24,
                0.25,
                0.26,
                0.27,
                0.28,
                0.29,
                0.3,
                0.31,
                0.32,
                0.33,
                0.34,
                0.35,
                0.36,
                0.37,
                0.38,
                0.39,
                0.4,
                0.41,
                0.42,
                0.43,
                0.44,
                0.45,
                0.46,
                0.47,
                0.48,
                0.49,
                0.5,
                0.51,
                0.52,
                0.53,
                0.54,
                0.55,
                0.56,
                0.57,
                0.58,
                0.59,
                0.6,
            ]
        )
        lower_x = np.array(
            [
                0.1,
                0.11,
                0.12,
                0.13,
                0.14,
                0.15,
                0.16,
                0.17,
                0.18,
                0.19,
                0.2,
                0.21,
                0.22,
                0.23,
                0.24,
                0.25,
                0.26,
                0.27,
                0.28,
                0.29,
                0.3,
                0.31,
                0.32,
                0.33,
                0.34,
                0.35,
                0.36,
                0.37,
                0.38,
                0.39,
                0.4,
                0.41,
                0.42,
                0.43,
                0.44,
                0.45,
                0.46,
                0.47,
                0.48,
                0.49,
                0.5,
                0.51,
                0.52,
                0.53,
                0.54,
                0.55,
                0.56,
                0.57,
                0.58,
                0.59,
                0.6,
            ]
        )
        upper_y = np.array(
            [
                0.0447,
                0.046,
                0.0472,
                0.0484,
                0.0495,
                0.0505,
                0.0514,
                0.0523,
                0.0531,
                0.0538,
                0.0545,
                0.0551,
                0.0557,
                0.0563,
                0.0568,
                0.0573,
                0.0577,
                0.0581,
                0.0585,
                0.0588,
                0.0591,
                0.0593,
                0.0595,
                0.0597,
                0.0599,
                0.06,
                0.0601,
                0.0602,
                0.0602,
                0.0602,
                0.0602,
                0.0602,
                0.0601,
                0.06,
                0.0599,
                0.0598,
                0.0596,
                0.0594,
                0.0592,
                0.0589,
                0.0586,
                0.0583,
                0.058,
                0.0576,
                0.0572,
                0.0568,
                0.0563,
                0.0558,
                0.0553,
                0.0547,
                0.0541,
            ]
        )  # noqa: E201, E241
        lower_y = np.array(
            [
                -0.0447,
                -0.046,
                -0.0473,
                -0.0485,
                -0.0496,
                -0.0506,
                -0.0515,
                -0.0524,
                -0.0532,
                -0.054,
                -0.0547,
                -0.0554,
                -0.056,
                -0.0565,
                -0.057,
                -0.0575,
                -0.0579,
                -0.0583,
                -0.0586,
                -0.0589,
                -0.0592,
                -0.0594,
                -0.0595,
                -0.0596,
                -0.0597,
                -0.0598,
                -0.0598,
                -0.0598,
                -0.0598,
                -0.0597,
                -0.0596,
                -0.0594,
                -0.0592,
                -0.0589,
                -0.0586,
                -0.0582,
                -0.0578,
                -0.0573,
                -0.0567,
                -0.0561,
                -0.0554,
                -0.0546,
                -0.0538,
                -0.0529,
                -0.0519,
                -0.0509,
                -0.0497,
                -0.0485,
                -0.0472,
                -0.0458,
                -0.0444,
            ]
        )

        # This dummy mesh must be passed to the surface dict so OpenAeroStruct
        # knows the dimensions of the mesh and whether it is a left or right wing
        dummy_mesh = np.zeros((nx, ny, 3))
        dummy_mesh[:, :, 0], dummy_mesh[:, :, 1] = np.meshgrid(
            np.linspace(0, 1, nx), np.linspace(-1, 0, ny), indexing="ij"
        )

        surf_dict = {
            # Wing definition
            "name": "wing",  # name of surface
            "symmetry": True,  # if true, model half of the wing (reflect across plane y = 0)
            "S_ref_type": "projected",  # how we compute the wing area (can be wetted or projected)
            "mesh": dummy_mesh,
            "fem_model_type": "wingbox",  # wingbox or tube
            "data_x_upper": upper_x,
            "data_x_lower": lower_x,
            "data_y_upper": upper_y,
            "data_y_lower": lower_y,
            "twist_cp": np.ones(n_twist),
            "spar_thickness_cp": np.ones(n_spar),  # [m]
            "skin_thickness_cp": np.ones(n_skin),  # [m]
            "t_over_c_cp": np.ones(n_toverc),
            "original_wingbox_airfoil_t_over_c": 0.12,
            "thickness_cp": np.array([0.1, 0.2, 0.3]),
            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha = 0)
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to
            # get the total CL and CD. These CL0 and CD0 values
            # don't vary with alpha.
            "CL0": 0.0,  # CL of the surface at alpha = 0
            "CD0": 0.0078,  # CD of the surface at alpha = 0
            # Airfoil properties for viscous drag calculation
            "k_lam": 0.05,  # percentage of chord with laminar flow
            "c_max_t": 0.38,  # chordwise location of maximum thickness
            "with_viscous": True,
            "with_wave": True,  # if true, compute wave drag
            # Structural values are based on aluminum 7075
            "E": 73.1e9,  # [Pa] Young's modulus of the spar
            "G": (73.1e9 / 2 / 1.33),  # [Pa] shear modulus of the spar (calculated using E and Poisson's ratio)
            "yield": 420e6 / 1.5,  # [Pa] yield stress divided by safety factor of 1.5
            "mrho": 2.78e3,  # [kg/m^3] material density
            "fem_origin": 0.35,  # normalized chordwise location of the spar
            "wing_weight_ratio": 1.25,  # estimate weight of other components like fasteners, overlaps, etc.
            "strength_factor_for_upper_skin": 1.0,  # yield stress is multiplied by this factor for upper skin
            "struct_weight_relief": True,  # if true, add the weight of the structure to its loads
            "distributed_fuel_weight": False,
            # Constraints
            "exact_failure_constraint": False,  # if false, use KS function
        }

        # Overwrite any options in the surface dict with those provided in the options
        if self.options["surf_options"] is not None:
            for key in self.options["surf_options"]:
                surf_dict[key] = self.options["surf_options"][key]

        # =================================================================
        #              Set up aerodynamic and structural mesh
        # =================================================================
        wing_group = om.Group()

        # Add bspline component for twist
        x_interp = np.linspace(0.0, 1.0, ny)
        comp = wing_group.add_subsystem(
            "twist_bsp",
            om.SplineComp(
                method="bsplines", x_interp_val=x_interp, num_cp=n_twist, interp_options={"order": min(n_twist, 4)}
            ),
            promotes_inputs=[("twist_cp", "ac|geom|wing|twist")],
        )
        comp.add_spline(y_cp_name="twist_cp", y_interp_name="twist", y_units="deg")

        # Add bspline component for thickness to chord ratio
        x_interp = np.linspace(0.0, 1.0, ny - 1)
        comp = wing_group.add_subsystem(
            "t_over_c_bsp",
            om.SplineComp(
                method="bsplines", x_interp_val=x_interp, num_cp=n_toverc, interp_options={"order": min(n_toverc, 4)}
            ),
            promotes_inputs=[("t_over_c_cp", "ac|geom|wing|toverc")],
            promotes_outputs=["t_over_c"],
        )
        comp.add_spline(y_cp_name="t_over_c_cp", y_interp_name="t_over_c")

        # Wing mesh generator
        wing_group.add_subsystem(
            "mesh_gen",
            TrapezoidalPlanformMesh(num_x=self.options["num_x"], num_y=self.options["num_y"]),
            promotes_inputs=["*"],
        )

        # Apply twist spline to mesh
        wing_group.add_subsystem(
            "twist_mesh", Rotate(val=np.zeros(ny), mesh_shape=(nx, ny, 3), symmetry=True), promotes_outputs=["mesh"]
        )
        wing_group.connect("twist_bsp.twist", "twist_mesh.twist")
        wing_group.connect("mesh_gen.mesh", "twist_mesh.in_mesh")

        wing_group.add_subsystem(
            "wingbox_group",
            WingboxGroup(surface=surf_dict),
            promotes_inputs=["mesh", "t_over_c", "skin_thickness_cp", "spar_thickness_cp"],
            promotes_outputs=[
                "A",
                "Iy",
                "Iz",
                "J",
                "Qz",
                "A_enc",
                "A_int",
                "htop",
                "hbottom",
                "hfront",
                "hrear",
                "skin_thickness",
                "spar_thickness",
            ],
        )

        wing_group.add_subsystem(
            "struct_setup",
            SpatialBeamSetup(surface=surf_dict),
            promotes_inputs=["mesh", "A", "Iy", "Iz", "J", "A_int"],
            promotes_outputs=[
                "nodes",
                "local_stiff_transformed",
                ("structural_mass", "ac|weights|W_wing"),
                "cg_location",
                "element_mass",
            ],
        )

        self.add_subsystem(
            "wing",
            wing_group,
            promotes_inputs=[
                ("S", "ac|geom|wing|S_ref"),
                ("AR", "ac|geom|wing|AR"),
                ("taper", "ac|geom|wing|taper"),
                ("sweep", "ac|geom|wing|c4sweep"),
                "ac|geom|wing|twist",
                "ac|geom|wing|toverc",
                ("skin_thickness_cp", "ac|geom|wing|skin_thickness"),
                ("spar_thickness_cp", "ac|geom|wing|spar_thickness"),
            ],
            promotes_outputs=["ac|weights|W_wing"],
        )

        self.add_subsystem(
            "aerostruct_point",
            AerostructPoint(surfaces=[surf_dict], internally_connect_fuelburn=False),
            promotes_inputs=[
                ("Mach_number", "fltcond|M"),
                ("alpha", "fltcond|alpha"),
                "W0",
                "empty_cg",
                "load_factor",
                ("coupled.load_factor", "load_factor"),
                ("total_perf.wing_structural_mass", "ac|weights|W_wing"),
            ],
            promotes_outputs=[("CL", "fltcond|CL"), ("CD", "fltcond|CD"), ("wing_perf.failure", "failure")],
        )
        self.connect("airspeed.Utrue", "aerostruct_point.v")
        self.connect("density.fltcond|rho", "aerostruct_point.rho")
        self.connect("Re_calc.re", "aerostruct_point.re")
        self.connect("sound_speed.fltcond|a", "aerostruct_point.speed_of_sound")

        # Set input defaults for inputs that go to multiple locations
        self.set_input_defaults("fltcond|M", 0.1)
        self.set_input_defaults("fltcond|alpha", 0.0)
        self.set_input_defaults("load_factor", 1.0)
        self.set_input_defaults("aerostruct_point.coupled.wing.nodes", np.zeros((ny, 3)), units="m")
        self.set_input_defaults("W0", 1.0, units="kg")  # unused variable but must be set since promoted
        # from multiple locations (may be used in the future)

        # Connect geometry parameters from the wing to the aerostructural analysis
        self.connect("wing.mesh", "aerostruct_point.coupled.wing.mesh")
        self.connect("wing.local_stiff_transformed", "aerostruct_point.coupled.wing.local_stiff_transformed")
        self.connect("wing.nodes", ["aerostruct_point.wing_perf.nodes", "aerostruct_point.coupled.wing.nodes"])
        if surf_dict["struct_weight_relief"]:
            self.connect("wing.element_mass", "aerostruct_point.coupled.wing.element_mass")
        self.connect("wing.cg_location", "aerostruct_point.total_perf.wing_cg_location")
        self.connect("wing.t_over_c", "aerostruct_point.wing_perf.t_over_c")
        self.connect("wing.spar_thickness", "aerostruct_point.wing_perf.spar_thickness")
        self.connect("wing.A_enc", "aerostruct_point.wing_perf.A_enc")
        self.connect("wing.Qz", "aerostruct_point.wing_perf.Qz")
        self.connect("wing.J", "aerostruct_point.wing_perf.J")
        self.connect("wing.htop", "aerostruct_point.wing_perf.htop")
        self.connect("wing.hbottom", "aerostruct_point.wing_perf.hbottom")
        self.connect("wing.hfront", "aerostruct_point.wing_perf.hfront")
        self.connect("wing.hrear", "aerostruct_point.wing_perf.hrear")
        self.connect("aerostruct_point.fuelburn", "aerostruct_point.total_perf.L_equals_W.fuelburn")


class AerostructDragPolarExact(om.Group):
    """
    .. warning:: This component is far more computationally expensive than the
                 AerostructDragPolar component, which uses a surrogate. For missions
                 with many flight segments, many num_nodes, or wing models with high
                 num_x and num_y values this component will result in a system that
                 returns a memory error when solved with a DirectSolver linear solver
                 because the Jacobian is too large to be factorized. Unless you know
                 what you're doing, this component should not be used (use
                 AerostructDragPolar instead).

    Drag polar and wing weight estimate generated using OpenAeroStruct's
    aerostructural analysis capabilities directly, without a surrogate in the loop.

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
    ac|geom|wing|AR : float
        Aspect ratio (scalar, dimensionless)
    ac|geom|wing|taper : float
        Taper ratio (must be >0 and <=1); tip chord / root chord (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Quarter chord sweep angle (scalar, degrees)
    ac|geom|wing|twist : float
        List of twist angles at control points of spline (vector, degrees)
        NOTE: length of vector is num_twist (set in options), NOT num_nodes
    ac|geom|wing|toverc : float
        List of thickness to chord ratios at control points of spline (vector, dimensionless)
        NOTE: length of vector is num_toverc (set in options)
    ac|geom|wing|skin_thickness : float
        List of skin thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_skin (set in options)
    ac|geom|wing|spar_thickness : float
        List of spar thicknesses at control points of spline (vector, m)
        NOTE: length of vector is num_spar (set in options)
    ac|aero|CD_nonwing : float
        Drag coefficient of components other than the wing; e.g. fuselage,
        tail, interference drag, etc.; this value is simply added to the
        drag coefficient computed by OpenAeroStruct (scalar, dimensionless)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)
        NOTE: fltcond|TempIncrement is a scalar in this component but a vector in OC. \
              This will be the case for the forseeable future because of the way the \
              OASDataGen component is set up. To make it work, TempIncrement would \
              need to be an input to the surrogate, which is not worth the extra \
              training cost (at minimum a 2x increase).

    Outputs
    -------
    drag : float
        Drag force (vector, Newtons)
    failure : float
        KS aggregation quantity obtained by combining the failure criteria
        for each FEM node. Must be < 0 to constrain wingboxes stresses to
        be less than yield stress. Used to simplify the optimization problem by
        reducing the number of constraints (vector, dimensionless)
    ac|weights|W_wing : float
        Weight of the wing (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points per mission segment (scalar, dimensionless)
    num_x : int
        Number of panels in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of panels in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist (scalar, dimensionless)
    num_toverc : int
        Number of spline control points for thickness to chord ratio (scalar, dimensionless)
    num_skin : int
        Number of spline control points for skin thickness (scalar, dimensionless)
    num_spar : int
        Number of spline control points for spar thickness (scalar, dimensionless)
    surf_options : dict
        Dictionary of OpenAeroStruct surface options; any options provided here
        will override the default ones; see the OpenAeroStruct documentation for more information.
        Because the geometry transformations are excluded in this model (to simplify the interface),
        the <transformation>_cp options are not supported. The input ac|geom|wing|twist is the same
        as modifying the twist_cp option in the surface dictionary. The mesh geometry modification
        is limited to adjusting the input parameters to this component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points to run")
        self.options.declare("num_x", default=2, desc="Number of streamwise mesh panels")
        self.options.declare("num_y", default=6, desc="Number of spanwise (half wing) mesh panels")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare("num_toverc", default=4, desc="Number of thickness to chord ratio spline control points")
        self.options.declare("num_skin", default=4, desc="Number of skin thickness spline control points")
        self.options.declare("num_spar", default=4, desc="Number of spar thickness spline control points")
        self.options.declare("surf_options", default=None, desc="Dictionary of OpenAeroStruct surface options")

    def setup(self):
        nn = self.options["num_nodes"]

        # Add an aerostructural analysis case for every node
        for node in range(nn):
            comp_name = f"aerostruct_{node}"
            self.add_subsystem(
                comp_name,
                Aerostruct(
                    num_x=self.options["num_x"],
                    num_y=self.options["num_y"],
                    num_twist=self.options["num_twist"],
                    num_toverc=self.options["num_toverc"],
                    num_skin=self.options["num_skin"],
                    num_spar=self.options["num_spar"],
                    surf_options=self.options["surf_options"],
                ),
                promotes_inputs=[
                    "ac|geom|wing|S_ref",
                    "ac|geom|wing|AR",
                    "ac|geom|wing|taper",
                    "ac|geom|wing|c4sweep",
                    "ac|geom|wing|twist",
                    "ac|geom|wing|toverc",
                    "ac|geom|wing|skin_thickness",
                    "ac|geom|wing|spar_thickness",
                    "fltcond|TempIncrement",
                ],
            )
            self.promotes(
                comp_name,
                inputs=["fltcond|alpha", "fltcond|M", "fltcond|h"],
                src_indices=[node],
                flat_src_indices=True,
                src_shape=(nn,),
            )

            # Promote wing weight from one, doesn't really matter which
            if node == 0:
                self.promotes(comp_name, outputs=["ac|weights|W_wing"])

        # Combine lift and drag coefficients from different aerostructural analyses into one vector
        comb = self.add_subsystem("vec_combine", VectorConcatenateComp(), promotes_outputs=["failure"])
        comb.add_relation(output_name="CL_OAS", input_names=[f"CL_{node}" for node in range(nn)], vec_sizes=[1] * nn)
        comb.add_relation(output_name="CD_OAS", input_names=[f"CD_{node}" for node in range(nn)], vec_sizes=[1] * nn)
        comb.add_relation(
            output_name="failure", input_names=[f"failure_{node}" for node in range(nn)], vec_sizes=[1] * nn
        )
        for node in range(nn):
            self.connect(f"aerostruct_{node}.fltcond|CL", f"vec_combine.CL_{node}")
            self.connect(f"aerostruct_{node}.fltcond|CD", f"vec_combine.CD_{node}")
            self.connect(f"aerostruct_{node}.failure", f"vec_combine.failure_{node}")

        # Solve for angle of attack that meets input lift coefficient
        self.add_subsystem(
            "alpha_bal",
            om.BalanceComp(
                "fltcond|alpha", eq_units=None, lhs_name="CL_OAS", rhs_name="fltcond|CL", val=np.ones(nn), units="deg"
            ),
            promotes_inputs=["fltcond|CL"],
            promotes_outputs=["fltcond|alpha"],
        )
        self.connect("vec_combine.CL_OAS", "alpha_bal.CL_OAS")

        # Compute drag force from drag coefficient
        self.add_subsystem(
            "drag_calc",
            om.ExecComp(
                "drag = q * S * (CD + CD0)",
                drag={"units": "N", "shape": (nn,)},
                q={"units": "Pa", "shape": (nn,)},
                S={"units": "m**2"},
                CD={"shape": (nn,)},
                CD0={"shape": (1,)},
            ),
            promotes_inputs=[("q", "fltcond|q"), ("S", "ac|geom|wing|S_ref"), ("CD0", "ac|aero|CD_nonwing")],
            promotes_outputs=["drag"],
        )
        self.connect("vec_combine.CD_OAS", "drag_calc.CD")


# Example usage of the aerostructural drag polar that compares
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
    t_over_c = np.array([0.12, 0.12])
    n_t_over_c = t_over_c.size
    skin = np.array([0.005, 0.025])
    n_skin = skin.size
    spar = np.array([0.004, 0.01])
    n_spar = spar.size

    M = 0.7
    CL = 0.35
    h = 0  # m

    p = om.Problem()
    p.model.add_subsystem(
        "aerostruct",
        AerostructDragPolar(
            num_nodes=nn,
            num_x=num_x,
            num_y=num_y,
            num_twist=n_twist,
            num_toverc=n_t_over_c,
            num_skin=n_skin,
            num_spar=n_spar,
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
    p.set_val("ac|geom|wing|toverc", t_over_c)
    p.set_val("ac|geom|wing|skin_thickness", skin, units="m")
    p.set_val("ac|geom|wing|spar_thickness", spar, units="m")
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
    print(f"Failure: {p.get_val('failure')}")
    print(f"Wing weight: {p.get_val('ac|weights|W_wing', units='kg')} kg")

    # Call OpenAeroStruct at the same flight condition to compare
    prob = om.Problem()
    prob.model.add_subsystem(
        "model",
        Aerostruct(
            num_x=num_x, num_y=num_y, num_twist=n_twist, num_toverc=n_t_over_c, num_skin=n_skin, num_spar=n_spar
        ),
        promotes=["*"],
    )

    prob.setup()

    # Set values
    # Geometry
    prob.set_val("ac|geom|wing|S_ref", S, units="m**2")
    prob.set_val("ac|geom|wing|AR", AR)
    prob.set_val("ac|geom|wing|taper", taper)
    prob.set_val("ac|geom|wing|c4sweep", sweep, units="deg")
    prob.set_val("ac|geom|wing|twist", twist, units="deg")
    prob.set_val("ac|geom|wing|toverc", t_over_c)
    prob.set_val("ac|geom|wing|skin_thickness", skin, units="m")
    prob.set_val("ac|geom|wing|spar_thickness", spar, units="m")

    # Flight condition
    prob.set_val("fltcond|M", M)
    prob.set_val("fltcond|alpha", p.get_val("aero_surrogate.alpha", units="deg"), units="deg")
    prob.set_val("fltcond|h", h, units="m")

    prob.run_model()

    print("================== OpenAeroStruct ==================")
    print(f"CL: {prob.get_val('fltcond|CL')}")
    print(f"CD: {prob.get_val('fltcond|CD')}")
    print(f"Alpha: {prob.get_val('fltcond|alpha', units='deg')} deg")
    print(f"Failure: {prob.get_val('failure')}")
    print(f"Wing weight: {prob.get_val('ac|weights|W_wing', units='kg')} kg")


if __name__ == "__main__":
    example_usage()
