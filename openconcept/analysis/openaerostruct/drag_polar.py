from __future__ import division

import numpy as np
import openmdao.api as om
from openaerostruct.geometry.geometry_mesh_transformations import Rotate
from openaerostruct.aerodynamics.aero_groups import AeroPoint

class OASDataGen(om.ExplicitComponent):
    """
    Generates a grid of OpenAeroStruct lift and drag data to train
    a surrogate model. The grid is defined by the options and the
    planform geometry by the inputs. This component will only recalculate
    the lift and drag grid when the planform shape changes.

    Inputs
    ------
    ac|geom|S_ref : float
        Full planform area (scalar, m^2)
    ac|geom|AR : float
        Aspect ratio (scalar, dimensionless)
    ac|geom|taper : float
        Taper ratio (must be >0 and <=1); tip chord / root chord (scalar, dimensionless)
    ac|geom|c4sweep : float
        Quarter chord sweep angle (scalar, degrees)
    ac|geom|twist : float
        List of twist angles at control points of spline (vector, degrees)
        NOTE: length of vector is num_twist (set in options)

    Outputs
    -------
    CL_train : 3-dim ndarray
        Grid of lift coefficient data to train structured surrogate model
    CD_train : 3-dim ndarray
        Grid of drag coefficient data to train structured surrogate model

    Options
    -------
    num_x: int
        number of points in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of points in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        number of spline control points for twist (scalar, dimensionless)
    alpha_train : list or ndarray
        List of angle of attack values at which to train the model (ndarray, degrees)
    Mach_train : 3-dim ndarray
        List of Mach numbers at which to train the model (ndarray, dimensionless)
    alt_train : 3-dim ndarray
        List of altitude values at which to train the model (ndarray, m)
    """
    def initialize(self):
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare('alpha_train', default=np.zeros((1,1,1)),
                             desc='List of angle of attack training values (degrees)')
        self.options.declare('Mach_train', default=np.zeros((1,1,1)),
                             desc='List of Mach number training values (dimensionless)')
        self.options.declare('alt_train', default=np.zeros((1,1,1)),
                             desc='List of altitude training values (meters)')
    
    def setup(self):
        self.add_input('ac|geom|S_ref', units='m**2')
        self.add_input('ac|geom|AR')
        self.add_input('ac|geom|taper')
        self.add_input('ac|geom|c4sweep', units='deg')

        n_alpha = self.options['alpha'].size
        n_Mach = self.options['Mach'].size
        n_alt = self.options['alt'].size
        self.add_output("CL_train", shape=(n_alpha, n_Mach, n_alt))
        self.add_output("CD_train", shape=(n_alpha, n_Mach, n_alt))

        self.declare_partials(['*'], ['*'], method='cs')

        # Generate grids and default cached values for training inputs and outputs
        self.S = -1
        self.AR = -1
        self.taper = -1
        self.c4sweep = -1
        self.alpha, self.Mach, self.alt = np.meshgrid(self.options['alpha_train'],
                                                      self.options['Mach_train'],
                                                      self.options['alt_train'],
                                                      indexing='ij')
        self.CL = np.zeros((n_alpha, n_Mach, n_alt))
        self.CD = np.zeros((n_alpha, n_Mach, n_alt))

class VLM(om.Group):
    """
    Computes lift and drag using OpenAeroStruct's vortex lattice implementation.

    Inputs
    ------
    fltcond|alpha : float
        Angle of attack (scalar, degrees)
    fltcond|M : float
        Mach number (vector, dimensionless)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)
    # fltcond|rho : float
    #     Density (vector, kg/m^3)
    # fltcond|Utrue : float
    #     True airspeed (vector, m/s)
    fltcond|h : float
        Altitude (vector, m)
    ac|geom|S_ref : float
        Wing planform area (scalar, m^2)
    ac|geom|AR : float
        Wing aspect ratio (scalar, dimensionless)
    ac|geom|taper : float
        Wing taper ratio (scalar, dimensionless)
    ac|geom|c4sweep : float
        Wing sweep measured at quarter chord (scalar, degrees)
    ac|geom|twist : float
        List of twist angles at control points of spline (vector, degrees)
        NOTE: length of vector is num_twist (set in options)

    Outputs
    -------
    fltcond|CL : float
        Lift coefficient of wing (vector, dimensionless)
    fltcond|CD : float
        Drag coefficient of wing (vector, dimensionless)
    
    Options
    -------
    num_x: int
        number of points in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of points in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        number of spline control points for twist (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
    
    def setup(self):
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])
        n_twist = int(self.options["num_twist"])

        # Set up mesh
        self.add_subsystem("mesh", PlanformMesh(num_x=nx, num_y=ny),
                           promotes_inputs=[("S", "ac|geom|S_ref"), ("AR", "ac|geom|AR"),
                                            ("taper", "ac|geom|taper"), ("sweep", "ac|geom|c4sweep")])

        # Add bspline component for twist
        x_interp = np.linspace(0.0, 1.0, ny)
        comp = self.add_subsystem(
            "twist_bsp",
            om.SplineComp(
                method="bsplines", x_interp_val=x_interp, num_cp=n_twist, interp_options={"order": min(n_twist, 4)}
            ),
            promotes_inputs=[("twist_cp", "ac|geom|twist")]
        )
        comp.add_spline(y_cp_name="twist_cp", y_interp_name="twist", y_units="deg")

        # Apply twist spline to mesh
        self.add_subsystem("twist_mesh", Rotate(val=np.zeros(ny), mesh_shape=(nx, ny, 3), symmetry=True))
        self.connect("twist_bsp.twist", "twist_mesh.twist")
        self.connect("mesh.mesh", "twist_mesh.in_mesh")

        # Compute dimensionalized Reynolds number (use linear interpolation from standard atmosphere up
        # to 35k ft to estimate dynamic viscosity)
        self.add_subsystem("Re_calc", om.ExecComp("re = rho * u / (-3.329134*10**(-10) * h + 1.792398*10**(-5))",
                                                  re={"units": "1/m", "val": 1e6, "ref": 1e6},
                                                  rho={"units": "kg/m**3", "val": 1.},
                                                  u={"units": "m/s", "val": 100.},
                                                  h={"units": "m", "val": 1.}),
                           promotes_inputs=[("rho", "fltcond|rho"), ("u", "fltcond|Utrue"), ("h", "fltcond|h")])

        surf_dict = {
            "name": "wing",
            "mesh": np.zeros((nx, ny, 3)),  # this must be defined
                                # because the VLMGeometry component uses the shape of the mesh in this
                                # dictionary to determine the size of the mesh; the values don't matter
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : 0.0,            # CL of the surface at alpha=0
            'CD0' : 0.0078,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c' : np.array([0.12]),      # thickness over chord ratio (NACA SC2-0612)
            'c_max_t' : .37,       # chordwise location of maximum (NACA SC2-0612)
                                    # thickness
            'with_viscous' : True,  # if true, compute viscous drag
            'with_wave' : True,     # if true, compute wave drag
            }

        self.add_subsystem("aero_point", AeroPoint(surfaces=[surf_dict]),
                           promotes_inputs=[("v", "fltcond|Utrue"), ("Mach_number", "fltcond|M"),
                                            ("rho", "fltcond|rho"), ("alpha", "fltcond|alpha")],
                           promotes_outputs=[(f"{surf_dict['name']}_perf.CD", "fltcond|CD"),
                                             (f"{surf_dict['name']}_perf.CL", "fltcond|CL")])
        self.connect("twist_mesh.mesh", [f"aero_point.{surf_dict['name']}.def_mesh",
                                         f"aero_point.aero_states.{surf_dict['name']}_def_mesh"])
        self.connect("Re_calc.re", "aero_point.re")

        # Set default input values of values that come from multiple sources
        self.set_input_defaults("fltcond|rho", val=1., units="kg/m**3")
        self.set_input_defaults("fltcond|Utrue", val=200., units="kn")

        # Set the thickness to chord ratio for wave and viscous drag calculation.
        # It must have a thickness to chord ratio for each panel, so there must be
        # ny-1 elements. Allow either one value (and duplicate it ny-1 times) or
        # an array of length ny-1, but nothing else.
        # NOTE: for aerostructural cases, this should be a design variable with control points over a spline
        if isinstance(surf_dict["t_over_c"], (int, float)) or surf_dict["t_over_c"].size == 1:
            self.set_input_defaults(f"aero_point.{surf_dict['name']}_perf.t_over_c", val=surf_dict["t_over_c"]*np.ones(ny-1))
        elif surf_dict["t_over_c"].size == ny - 1:
            self.set_input_defaults(f"aero_point.{surf_dict['name']}_perf.t_over_c", val=surf_dict["t_over_c"])
        else:
            raise ValueError(f"t_over_c_ in the surface dict must be either a number or an ndarray " \
                             f"with either one or ny-1 elements, not {surf_dict['t_over_c']}")


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
class PlanformMesh(om.ExplicitComponent):
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
        self.add_input("taper", val=1.)
        self.add_input("sweep", val=10, units="deg")

        self.add_output("mesh", val=mesh, shape=(nx, ny, 3), units="m")

        self.declare_partials("mesh", ["*"], method="fd")  # TODO: do this analytically
    
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
        x_mesh -= np.linspace(c_root*taper, c_root, ny).reshape(1, ny)/4  # shift to quarter chord at x=0
        x_mesh += np.linspace(half_span, 0, ny).reshape(1, ny) * np.tan(np.deg2rad(sweep))  # sweep wing at quarter chord

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = x_mesh
        mesh[:, :, 1] = y_mesh

        outputs["mesh"] = mesh