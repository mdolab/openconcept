import numpy as np
import openmdao.api as om
from time import time
from copy import copy, deepcopy
import multiprocessing as mp

# Atmospheric calculations
from openconcept.atmospherics import TemperatureComp, PressureComp, DensityComp, SpeedOfSoundComp

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
    model to decrease the computational cost.

    Notes
    -----
    Twist is ordered starting at the tip and moving to the root; a twist
    of [-1, 0, 1] would have a tip twist of -1 deg and root twist of 1 deg

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
    ac|aero|CD_nonwing : float
        Drag coefficient of components other than the wing; e.g. fuselage,
        tail, interference drag, etc.; this value is simply added to the
        drag coefficient computed by OpenAeroStruct (scalar, dimensionless)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)
        NOTE: fltcond|TempIncrement is a scalar in this component but a vector in OC. \
              This will be the case for the forseeable future because of the way the \
              VLMDataGen component is set up. To make it work, TempIncrement would \
              need to be an input to the surrogate, which is not worth the extra \
              training cost (at minimum a 2x increase).

    Outputs
    -------
    drag : float
        Drag force (vector, Newtons)

    Options
    -------
    num_nodes : int
        Number of analysis points per mission segment (scalar, dimensionless)
    num_x : int
        Number of points in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of points in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist (scalar, dimensionless)
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
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
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
            VLMDataGen(
                num_x=self.options["num_x"],
                num_y=self.options["num_y"],
                num_twist=self.options["num_twist"],
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
                "ac|aero|CD_nonwing",
                "fltcond|TempIncrement",
            ],
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


class VLMDataGen(om.ExplicitComponent):
    """
    Generates a grid of OpenAeroStruct lift and drag data to train
    a surrogate model. The grid is defined by the options and the
    planform geometry by the inputs. This component will only recalculate
    the lift and drag grid when the planform shape changes. All VLMDataGen
    components in the model must use the same training points and surf_options.

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
        Number of points in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of points in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist (scalar, dimensionless)
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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
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
        VLMDataGen.S = -np.ones(1)
        VLMDataGen.AR = -np.ones(1)
        VLMDataGen.taper = -np.ones(1)
        VLMDataGen.c4sweep = -np.ones(1)
        VLMDataGen.twist = -1 * np.ones((self.options["num_twist"],))
        VLMDataGen.temp_incr = -42 * np.ones(1)
        VLMDataGen.Mach, VLMDataGen.alpha, VLMDataGen.alt = np.meshgrid(
            self.options["Mach_train"], self.options["alpha_train"], self.options["alt_train"], indexing="ij"
        )
        VLMDataGen.CL = np.zeros((n_Mach, n_alpha, n_alt))
        VLMDataGen.CD = np.zeros((n_Mach, n_alpha, n_alt))
        VLMDataGen.partials = None

    def compute(self, inputs, outputs):
        S = inputs["ac|geom|wing|S_ref"]
        AR = inputs["ac|geom|wing|AR"]
        taper = inputs["ac|geom|wing|taper"]
        sweep = inputs["ac|geom|wing|c4sweep"]
        twist = inputs["ac|geom|wing|twist"]
        CD_nonwing = inputs["ac|aero|CD_nonwing"]
        temp_incr = inputs["fltcond|TempIncrement"]

        # If the inputs are unchaged, use the previously calculated values
        tol = 1e-13  # floating point comparison tolerance
        if (
            np.abs(S - VLMDataGen.S) < tol
            and np.abs(AR - VLMDataGen.AR) < tol
            and np.abs(taper - VLMDataGen.taper) < tol
            and np.abs(sweep - VLMDataGen.c4sweep) < tol
            and np.all(np.abs(twist - VLMDataGen.twist) < tol)
            and np.abs(temp_incr - VLMDataGen.temp_incr) < tol
        ):
            outputs["CL_train"] = VLMDataGen.CL
            outputs["CD_train"] = VLMDataGen.CD + CD_nonwing
            return

        print(f"S = {S}; AR = {AR}; taper = {taper}; sweep = {sweep}; twist = {twist}; temp_incr = {temp_incr}")
        # Copy new values to cached ones
        VLMDataGen.S[:] = S
        VLMDataGen.AR[:] = AR
        VLMDataGen.taper[:] = taper
        VLMDataGen.c4sweep[:] = sweep
        VLMDataGen.twist[:] = twist
        VLMDataGen.temp_incr[:] = temp_incr

        # Compute new training values
        train_in = {}
        train_in["Mach_number_grid"] = VLMDataGen.Mach
        train_in["alpha_grid"] = VLMDataGen.alpha
        train_in["alt_grid"] = VLMDataGen.alt
        train_in["TempIncrement"] = temp_incr
        train_in["S_ref"] = S
        train_in["AR"] = AR
        train_in["taper"] = taper
        train_in["c4sweep"] = sweep
        train_in["twist"] = twist
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
        NOTE: length of vector is num_twist (set in options of VLMDataGen)
    num_x: int
        number of points in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of points in y (spanwise) direction for one wing because
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
    keys = ["TempIncrement", "S_ref", "AR", "taper", "c4sweep", "twist", "num_x", "num_y"]
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
    of = ["CL_train", "CD_train"]
    wrt = [
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|taper",
        "ac|geom|wing|c4sweep",
        "ac|geom|wing|twist",
        "fltcond|TempIncrement",
    ]
    partials = {}
    for f in of:
        for u in wrt:
            if u == "ac|geom|wing|twist":
                partials[f, u] = np.zeros((jac_num_rows, inputs["twist"].size))
            else:
                partials[f, u] = np.zeros((jac_num_rows, 1))
    data = {"CL": CL, "CD": CD, "partials": partials}

    # Transfer data into output data structure the proper format
    for i in range(len(out)):
        data["CL"][np.unravel_index(i, inputs["Mach_number_grid"].shape)] = out[i]["CL"]
        data["CD"][np.unravel_index(i, inputs["Mach_number_grid"].shape)] = out[i]["CD"]
        for f in of:
            for u in wrt:
                data["partials"][f, u][i] = out[i]["partials"][f, u]

    print(f"        ...done in {time() - t_start} sec\n")

    return data


# Function to compute CL, CD, and derivatives at a given test point. Used for
# the parallel mapping function in compute_training_data
# Input "point" is row in test_points array
def compute_aerodynamic_data(point):
    inputs = point[3]

    # Set up OpenAeroStruct problem
    p = om.Problem()
    p.model.add_subsystem(
        "aero_analysis",
        VLM(
            num_x=inputs["num_x"],
            num_y=inputs["num_y"],
            num_twist=inputs["twist"].size,
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
    p.setup()

    p.set_val("fltcond|M", point[0])
    p.set_val("fltcond|alpha", point[1], units="deg")
    p.set_val("fltcond|h", point[2], units="m")

    p.run_model()

    output = {"CL": p.get_val("fltcond|CL"), "CD": p.get_val("fltcond|CD"), "partials": {}}

    # Compute derivatives
    of = ["fltcond|CL", "fltcond|CD"]
    of_out = ["CL_train", "CD_train"]
    wrt = [
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|taper",
        "ac|geom|wing|c4sweep",
        "ac|geom|wing|twist",
        "fltcond|TempIncrement",
    ]
    deriv = p.compute_totals(of, wrt)
    for n, f in enumerate(of):
        for u in wrt:
            output["partials"][of_out[n], u] = np.copy(deriv[f, u])

    return output


class VLM(om.Group):
    """
    Computes lift and drag using OpenAeroStruct's vortex lattice implementation.

    Notes
    -----
    Twist is ordered starting at the tip and moving to the root; a twist
    of [-1, 0, 1] would have a tip twist of -1 deg and root twist of 1 deg

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

    Outputs
    -------
    fltcond|CL : float
        Lift coefficient of wing (scalar, dimensionless)
    fltcond|CD : float
        Drag coefficient of wing (scalar, dimensionless)

    Options
    -------
    num_x : int
        Number of points in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of points in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist (scalar, dimensionless)
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
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare("surf_options", default=None, desc="Dictionary of OpenAeroStruct surface options")

    def setup(self):
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])
        n_twist = int(self.options["num_twist"])

        # =================================================================
        #                            Set up mesh
        # =================================================================
        self.add_subsystem(
            "mesh",
            PlanformMesh(num_x=nx, num_y=ny),
            promotes_inputs=[
                ("S", "ac|geom|wing|S_ref"),
                ("AR", "ac|geom|wing|AR"),
                ("taper", "ac|geom|wing|taper"),
                ("sweep", "ac|geom|wing|c4sweep"),
            ],
        )

        # Add bspline component for twist
        x_interp = np.linspace(0.0, 1.0, ny)
        comp = self.add_subsystem(
            "twist_bsp",
            om.SplineComp(
                method="bsplines", x_interp_val=x_interp, num_cp=n_twist, interp_options={"order": min(n_twist, 4)}
            ),
            promotes_inputs=[("twist_cp", "ac|geom|wing|twist")],
        )
        comp.add_spline(y_cp_name="twist_cp", y_interp_name="twist", y_units="deg")

        # Apply twist spline to mesh
        self.add_subsystem("twist_mesh", Rotate(val=np.zeros(ny), mesh_shape=(nx, ny, 3), symmetry=True))
        self.connect("twist_bsp.twist", "twist_mesh.twist")
        self.connect("mesh.mesh", "twist_mesh.in_mesh")

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

        surf_dict = {
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
                surf_dict[key] = self.options["surf_options"][key]

        self.add_subsystem(
            "aero_point",
            AeroPoint(surfaces=[surf_dict]),
            promotes_inputs=[("Mach_number", "fltcond|M"), ("alpha", "fltcond|alpha")],
            promotes_outputs=[
                (f"{surf_dict['name']}_perf.CD", "fltcond|CD"),
                (f"{surf_dict['name']}_perf.CL", "fltcond|CL"),
            ],
        )
        self.connect(
            "twist_mesh.mesh",
            [f"aero_point.{surf_dict['name']}.def_mesh", f"aero_point.aero_states.{surf_dict['name']}_def_mesh"],
        )
        self.connect("airspeed.Utrue", "aero_point.v")
        self.connect("density.fltcond|rho", "aero_point.rho")
        self.connect("Re_calc.re", "aero_point.re")

        # Set input defaults for inputs that go to multiple locations
        self.set_input_defaults("fltcond|M", 0.1)
        self.set_input_defaults("fltcond|alpha", 0.0)

        # Set the thickness to chord ratio for wave and viscous drag calculation.
        # It must have a thickness to chord ratio for each panel, so there must be
        # ny-1 elements. Allow either one value (and duplicate it ny-1 times) or
        # an array of length ny-1, but nothing else.
        # NOTE: for aerostructural cases, this should be a design variable with control points over a spline
        if isinstance(surf_dict["t_over_c"], (int, float)) or surf_dict["t_over_c"].size == 1:
            self.set_input_defaults(
                f"aero_point.{surf_dict['name']}_perf.t_over_c", val=surf_dict["t_over_c"] * np.ones(ny - 1)
            )
        elif surf_dict["t_over_c"].size == ny - 1:
            self.set_input_defaults(f"aero_point.{surf_dict['name']}_perf.t_over_c", val=surf_dict["t_over_c"])
        else:
            raise ValueError(
                f"t_over_c in the surface dict must be either a number or an ndarray "
                f"with either one or ny-1 elements, not {surf_dict['t_over_c']}"
            )


class PlanformMesh(om.ExplicitComponent):
    """
    Generate an OpenAeroStruct mesh based on basic wing design parameters.
    Resulting mesh is for a half wing (meant to use with OpenAeroStruct symmetry),
    but the input reference area is for the full wing.

    Inputs
    ------
    S: float
        full planform area (scalar, m^2)
    AR: float
        aspect ratio (scalar, dimensionless)
    taper: float
        taper ratio (must be >0 and <=1); tip chord / root chord (scalar, dimensionless)
    sweep: float
        quarter chord sweep angle (scalar, degrees)

    Outputs
    -------
    mesh: ndarray
        OpenAeroStruct 3D mesh (num_x x num_y x 3 ndarray, m)

    Options
    -------
    num_x: int
        number of points in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of points in y (spanwise) direction (scalar, dimensionless)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cite = CITATION

    def initialize(self):
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")

    def setup(self):
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])

        # Generate default mesh
        x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(-1, 0, ny), indexing="ij")
        y *= 5
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = x
        mesh[:, :, 1] = y

        self.add_input("S", val=10, units="m**2")
        self.add_input("AR", val=10)
        self.add_input("taper", val=1.0)
        self.add_input("sweep", val=10, units="deg")

        self.add_output("mesh", val=mesh, shape=(nx, ny, 3), units="m")

        self.declare_partials("mesh", "*")

    def compute(self, inputs, outputs):
        S = inputs["S"]
        AR = inputs["AR"]
        taper = inputs["taper"]
        sweep = inputs["sweep"]
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])

        # Compute absolute dimensions from wing geometry spec
        half_span = np.sqrt(AR * S) / 2
        c_root = S / (half_span * (1 + taper))

        # Create baseline square mesh from 0 to 1 in each direction
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(-1, 0, ny), indexing="ij")

        # Morph the mesh to fit the desired wing shape
        x_mesh *= c_root
        y_mesh *= half_span  # rectangular wing with correct root chord and wingspan
        x_mesh *= np.linspace(taper, 1, ny).reshape(1, ny)  # taper wing
        x_mesh -= np.linspace(c_root * taper, c_root, ny).reshape(1, ny) / 4  # shift to quarter chord at x=0
        x_mesh += np.linspace(half_span, 0, ny).reshape(1, ny) * np.tan(
            np.deg2rad(sweep)
        )  # sweep wing at quarter chord

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = x_mesh
        mesh[:, :, 1] = y_mesh

        outputs["mesh"] = mesh

    def compute_partials(self, inputs, J):
        S = inputs["S"]
        AR = inputs["AR"]
        taper = inputs["taper"]
        sweep = inputs["sweep"]
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])

        # Compute absolute dimensions from wing geometry spec
        half_span = np.sqrt(AR * S) / 2
        c_root = S / (half_span * (1 + taper))

        # Create baseline square mesh from 0 to 1 in each direction
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(-1, 0, ny), indexing="ij")

        # Compute derivatives in a way analogous to forward AD
        db_dS = AR / (4 * np.sqrt(AR * S))
        db_dAR = S / (4 * np.sqrt(AR * S))
        dcroot_dS = 1 / (half_span * (1 + taper)) - S / (half_span**2 * (1 + taper)) * db_dS
        dcroot_dAR = -S / (half_span**2 * (1 + taper)) * db_dAR
        dcroot_dtaper = -S / (half_span * (1 + taper) ** 2)

        dy_dS = y_mesh * db_dS
        dy_dAR = y_mesh * db_dAR

        dx_dS = x_mesh * np.linspace(taper, 1, ny).reshape(1, ny) * dcroot_dS
        dx_dS -= np.linspace(dcroot_dS * taper, dcroot_dS, ny).reshape(1, ny) / 4
        dx_dS += np.linspace(db_dS, 0, ny).reshape(1, ny) * np.tan(np.deg2rad(sweep))

        dx_dAR = x_mesh * np.linspace(taper, 1, ny).reshape(1, ny) * dcroot_dAR
        dx_dAR -= np.linspace(dcroot_dAR * taper, dcroot_dAR, ny).reshape(1, ny) / 4
        dx_dAR += np.linspace(db_dAR, 0, ny).reshape(1, ny) * np.tan(np.deg2rad(sweep))

        dx_dtaper = (
            x_mesh * c_root * np.linspace(1, 0, ny).reshape(1, ny)
            + x_mesh * np.linspace(taper, 1, ny).reshape(1, ny) * dcroot_dtaper
        )
        dx_dtaper -= (
            np.linspace(c_root, 0, ny).reshape(1, ny) / 4
            + np.linspace(dcroot_dtaper * taper, dcroot_dtaper, ny).reshape(1, ny) / 4
        )

        dx_dsweep = (
            0 * x_mesh + np.linspace(half_span, 0, ny).reshape(1, ny) / np.cos(np.deg2rad(sweep)) ** 2 * np.pi / 180.0
        )

        J["mesh", "S"] = np.dstack((dx_dS, dy_dS, np.zeros((nx, ny)))).flatten()
        J["mesh", "AR"] = np.dstack((dx_dAR, dy_dAR, np.zeros((nx, ny)))).flatten()
        J["mesh", "taper"] = np.dstack((dx_dtaper, np.zeros((nx, ny)), np.zeros((nx, ny)))).flatten()
        J["mesh", "sweep"] = np.dstack((dx_dsweep, np.zeros((nx, ny)), np.zeros((nx, ny)))).flatten()


# Example usage of the drag polar that compares
# the surrogate to a direction OpenAeroStruct call
# with a very coarse mesh and training point distribution
def example_usage():
    # Define parameters
    nn = 1
    num_x = 3
    num_y = 5
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
    prob.model.add_subsystem("model", VLM(num_x=num_x, num_y=num_y, num_twist=n_twist), promotes=["*"])

    prob.setup()

    # Set values
    # Geometry
    prob.set_val("ac|geom|wing|S_ref", S, units="m**2")
    prob.set_val("ac|geom|wing|AR", AR)
    prob.set_val("ac|geom|wing|taper", taper)
    prob.set_val("ac|geom|wing|c4sweep", sweep, units="deg")
    prob.set_val("ac|geom|wing|twist", twist, units="deg")

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
