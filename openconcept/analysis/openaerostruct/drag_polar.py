from __future__ import division

import numpy as np
import openmdao.api as om
from time import time
from copy import copy

# OpenAeroStruct
try:
    from openaerostruct.geometry.geometry_mesh_transformations import Rotate
    from openaerostruct.aerodynamics.aero_groups import AeroPoint
except ImportError:
    raise ImportError("OpenAeroStruct must be installed to use the OASDragPolar aerodynamic component")

# Atmospheric calculations
from openconcept.analysis.atmospherics.temperature_comp import TemperatureComp
from openconcept.analysis.atmospherics.pressure_comp import PressureComp
from openconcept.analysis.atmospherics.density_comp import DensityComp
from openconcept.analysis.atmospherics.speedofsound_comp import SpeedOfSoundComp


class OASDragPolar(om.Group):
    """
    Drag polar generated using OpenAeroStruct's vortex lattice method and a surrogate
    model to decrease the computational cost.

    Inputs
    ------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
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
        NOTE: length of vector is num_twist (set in options), NOT num_nodes
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)

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
        will override the default ones; see the OpenAeroStruct documentation for more information
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points to run')
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare('alpha_train', default=np.zeros((1,1,1)),
                             desc='List of angle of attack training values (degrees)')
        self.options.declare('Mach_train', default=np.zeros((1,1,1)),
                             desc='List of Mach number training values (dimensionless)')
        self.options.declare('alt_train', default=np.zeros((1,1,1)),
                             desc='List of altitude training values (meters)')
        self.options.declare('surf_options', default=None, desc="Dictionary of OpenAeroStruct surface options")
    
    def setup(self):
        nn = self.options['num_nodes']
        n_alpha = self.options['alpha_train'].size
        n_Mach = self.options['Mach_train'].size
        n_alt = self.options['alt_train'].size

        # Training data
        self.add_subsystem('training_data', OASDataGen(num_x=self.options['num_x'], num_y=self.options['num_y'],
                                                       num_twist=self.options['num_twist'], alpha_train=self.options['alpha_train'],
                                                       Mach_train=self.options['Mach_train'], alt_train=self.options['alt_train'],
                                                       surf_options=self.options['surf_options']),
                           promotes_inputs=['ac|geom|S_ref', 'ac|geom|AR', 'ac|geom|taper', 'ac|geom|c4sweep',
                                            'ac|geom|twist', 'fltcond|TempIncrement'])

        # Surrogate model
        interp = om.MetaModelStructuredComp(method='lagrange3', training_data_gradients=True)
        interp.add_input('alpha', 0., shape=(nn,), units='deg', training_data=self.options['alpha_train'])
        interp.add_input('fltcond|M', 0.1, shape=(nn,), training_data=self.options['Mach_train'])
        interp.add_input('fltcond|h', 0., shape=(nn,), units='m', training_data=self.options['alt_train'])
        interp.add_output('CL', shape=(nn,), training_data=np.zeros((n_alpha, n_Mach, n_alt)))
        interp.add_output('CD', shape=(nn,), training_data=np.zeros((n_alpha, n_Mach, n_alt)))
        self.add_subsystem('aero_surrogate', interp, promotes_inputs=['fltcond|M', 'fltcond|h'])
        self.connect('training_data.CL_train', 'aero_surrogate.CL_train')
        self.connect('training_data.CD_train', 'aero_surrogate.CD_train')

        # Solve for angle of attack that meets input lift coefficient
        self.add_subsystem("alpha_bal", om.BalanceComp('alpha', eq_units=None, lhs_name="CL_VLM",
                                                       rhs_name="fltcond|CL", val=np.ones(nn),
                                                       units="deg"),
                           promotes_inputs=['fltcond|CL'])
        self.connect("alpha_bal.alpha", "aero_surrogate.alpha")
        self.connect("aero_surrogate.CL", "alpha_bal.CL_VLM")

        # Compute drag force from drag coefficient
        self.add_subsystem('drag_calc', om.ExecComp('drag = q * S * CD',
                                                    drag={'units': 'N', 'shape': (nn,)},
                                                    q={'units': 'Pa', 'shape': (nn,)},
                                                    S={'units': 'm**2', 'shape': (nn,)},
                                                    CD={'shape': (nn,)},),
                           promotes_inputs=[('q', 'fltcond|q'), ('S', 'ac|geom|S_ref')],
                           promotes_outputs=['drag'])
        self.connect('aero_surrogate.CD', 'drag_calc.CD')


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
    alpha_train : list or ndarray
        List of angle of attack values at which to train the model (ndarray, degrees)
    Mach_train : list or ndarray
        List of Mach numbers at which to train the model (ndarray, dimensionless)
    alt_train : list or ndarray
        List of altitude values at which to train the model (ndarray, m)
    surf_options : dict
        Dictionary of OpenAeroStruct surface options; any options provided here
        will override the default ones; see the OpenAeroStruct documentation for more information
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
        self.options.declare('surf_options', default=None, desc="Dictionary of OpenAeroStruct surface options")
    
    def setup(self):
        self.add_input('ac|geom|S_ref', units='m**2')
        self.add_input('ac|geom|AR')
        self.add_input('ac|geom|taper')
        self.add_input('ac|geom|c4sweep', units='deg')
        self.add_input('ac|geom|twist', shape=(self.options["num_twist"],), units='deg')
        self.add_input('fltcond|TempIncrement', val=0., units='degC')

        n_alpha = self.options['alpha'].size
        n_Mach = self.options['Mach'].size
        n_alt = self.options['alt'].size
        self.add_output("CL_train", shape=(n_alpha, n_Mach, n_alt))
        self.add_output("CD_train", shape=(n_alpha, n_Mach, n_alt))

        self.declare_partials(['*'], ['*'], method='fd')

        # Generate grids and default cached values for training inputs and outputs
        self.S = -1
        self.AR = -1
        self.taper = -1
        self.c4sweep = -1
        self.twist = -1*np.ones((self.options["num_twist"],))
        self.temp_incr = -42
        self.alpha, self.Mach, self.alt = np.meshgrid(self.options['alpha_train'],
                                                      self.options['Mach_train'],
                                                      self.options['alt_train'],
                                                      indexing='ij')
        self.CL = np.zeros((n_alpha, n_Mach, n_alt))
        self.CD = np.zeros((n_alpha, n_Mach, n_alt))
    
    def compute(self, inputs, outputs):
        S = inputs["ac|geom|S_ref"]
        AR = inputs["ac|geom|AR"]
        taper = inputs["ac|geom|taper"]
        sweep = inputs["ac|geom|c4sweep"]
        twist = inputs["ac|geom|twist"]
        temp_incr = inputs["fltcond|TempIncrement"]

        # If the inputs are unchaged, use the previously calculated values
        if (S == self.S and
           AR == self.AR and
           taper == self.taper and
           sweep == self.sweep and
           np.all(twist == self.twist) and
           temp_incr == self.temp_incr):
            outputs['CL_train'] = self.CL
            outputs['CD_train'] = self.CD
            return

        # Copy new values to cached ones
        self.S = copy(S)
        self.AR = copy(AR)
        self.taper = copy(taper)
        self.c4sweep = copy(sweep)
        self.twist = copy(twist)
        self.temp_incr = copy(temp_incr)
        
        # Compute new training values
        train_in = {}
        train_in['alpha_grid'] = self.alpha
        train_in['Mach_number_grid'] = self.Mach
        train_in['alt_grid'] = self.alt
        train_in['TempIncrement'] = temp_incr
        train_in['S_ref'] = S
        train_in['AR'] = AR
        train_in['taper'] = taper
        train_in['c4sweep'] = sweep
        train_in['twist'] = twist
        train_in['num_x'] = self.options['num_x']
        train_in['num_y'] = self.options['num_y']

        data = compute_training_data(inputs, surf_dict=self.options['surf_options'])
        self.CL = copy(data['CL'])
        self.CD = copy(data['CD'])


"""
Generates training data and its total derivatives by
calling OpenAeroStruct at each training point.

Inputs
------
inputs : dict
    A dictionary containing the following entries:
    alpha_grid : ndarray
        Angle of attack (3D meshgrid 'ij'-style ndarray, degrees)
    Mach_number_grid : mdarray
        Mach number (3D meshgrid 'ij'-style ndarray, dimensionless)
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
    num_x: int
        number of points in x (streamwise) direction (scalar, dimensionless)
    num_y: int
        number of points in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
TODO: add optional surf_dict input? Add any addition options to the default surf_dict in the VLM group

Outputs
-------
data : dict
    A dictionary containing the following entries:
    CL : ndarray
        Lift coefficients at training points (3D meshgrid 'ij'-style ndarray, dimensionless)
    CD : ndarray
        Drag coefficients at training points (3D meshgrid 'ij'-style ndarray, dimensionless)
    TODO: derivatives?

"""
def compute_training_data(inputs, surf_dict=None):
    # Initialize output arrays
    CL = np.zeros(inputs['alpha_grid'].shape)
    CD = np.zeros(inputs['alpha_grid'].shape)

    # Set up OpenAeroStruct problem
    p = om.Problem()
    p.model.add_subsystem('aero_analysis', VLM(num_x=inputs['num_x'],
                                               num_y=inputs['num_y'],
                                               num_twist=inputs['twist'].size,
                                               surf_options=surf_dict),
                           promotes=['*'])

    # Set design variables
    p.model.set_input_defaults('fltcond|TempIncrement', val=inputs['TempIncrement'], units='degC')
    p.model.set_input_defaults('ac|geom|S_ref', val=inputs['S_ref'], units='m**2')
    p.model.set_input_defaults('ac|geom|AR', val=inputs['AR'])
    p.model.set_input_defaults('ac|geom|taper', val=inputs['taper'])
    p.model.set_input_defaults('ac|geom|c4sweep', val=inputs['c4sweep'], units='deg')
    p.model.set_input_defaults('ac|geom|twist', val=inputs['twist'], units='deg')

    p.setup()

    # Iterate through the training points and evaluate the model
    iter = 0
    total = inputs['alpha_grid'].shape[0]*inputs['alpha_grid'].shape[1]*inputs['alpha_grid'].shape[2]
    model_time = 0
    deriv_time = 0
    for i in range(inputs['alpha_grid'].shape[0]):
        for j in range(inputs['alpha_grid'].shape[1]):
            for k in range(inputs['alpha_grid'].shape[2]):
                # Set the values for the current training point in the model
                p.set_val('fltcond|alpha', inputs['alpha_grid'][i,j,k], units='deg')
                p.set_val('fltcond|M', inputs['Mach_number_grid'][i,j,k])
                p.set_val('fltcond|h', inputs['alt_grid'][i,j,k], units='m')

                # Run the models and pull the lift and drag values out
                print(f"Generating training data...{iter/total*100:.1f}%", end='\r')
                iter += 1
                start = time()
                p.run_model()
                model_time += time() - start
                CL[i,j,k] = p.get_val('fltcond|CL')
                CD[i,j,k] = p.get_val('fltcond|CD')

                start = time()
                deriv = p.compute_totals(of=['fltcond|CL', 'fltcond|CD'], wrt=['ac|geom|S_ref', 'ac|geom|AR',
                                         'ac|geom|taper', 'ac|geom|c4sweep', 'ac|geom|twist', 'fltcond|TempIncrement'])
                deriv_time += time() - start
    
    print(f"Model convergence time = {model_time} sec")
    print(f"Derivative computation time = {deriv_time} sec")
    data = {'CL': CL, 'CD': CD}
    return data


class VLM(om.Group):
    """
    Computes lift and drag using OpenAeroStruct's vortex lattice implementation.

    Inputs
    ------
    fltcond|alpha : float
        Angle of attack (scalar, degrees)
    fltcond|M : float
        Mach number (vector, dimensionless)
    fltcond|h : float
        Altitude (vector, m)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (scalar, degC)
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
    num_x : int
        Number of points in x (streamwise) direction (scalar, dimensionless)
    num_y : int
        Number of points in y (spanwise) direction for one wing because
        uses symmetry (scalar, dimensionless)
    num_twist : int
        Number of spline control points for twist (scalar, dimensionless)
    surf_options : dict
        Dictionary of OpenAeroStruct surface options; any options provided here
        will override the default ones; see the OpenAeroStruct documentation for more information
    """
    def initialize(self):
        self.options.declare("num_x", default=3, desc="Number of streamwise mesh points")
        self.options.declare("num_y", default=7, desc="Number of spanwise (half wing) mesh points")
        self.options.declare("num_twist", default=4, desc="Number of twist spline control points")
        self.options.declare('surf_options', default=None, desc="Dictionary of OpenAeroStruct surface options")
    
    def setup(self):
        nx = int(self.options["num_x"])
        ny = int(self.options["num_y"])
        n_twist = int(self.options["num_twist"])

        # =================================================================
        #                            Set up mesh
        # =================================================================
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

        # =================================================================
        #              Compute atmospheric and fluid properties
        # =================================================================
        self.add_subsystem('temp', TemperatureComp(num_nodes=1),
                           promotes_inputs=['fltcond|h', 'fltcond|TempIncrement'])
        self.add_subsystem('pressure', PressureComp(num_nodes=1),
                           promotes_inputs=['fltcond|h'])
        self.add_subsystem('density', DensityComp(num_nodes=1))
        self.connect('temp.fltcond|T', 'density.fltcond|T')
        self.connect('pressure.fltcond|p', 'density.fltcond|p')
        self.add_subsystem('sound_speed', SpeedOfSoundComp(num_nodes=1))
        self.connect('temp.fltcond|T', 'sound_speed.fltcond|T')
        self.add_subsystem('airspeed', om.ExecComp("Utrue = Mach * a",
                                                   Utrue={'units': 'm/s', 'val': 200.},
                                                   a={'units': 'm/s', 'val': 300.}),
                           promotes_inputs=[('Mach', 'fltcond|M')])
        self.connect('sound_speed.fltcond|a', 'airspeed.a')

        # Compute dimensionalized Reynolds number (use linear interpolation from standard atmosphere up
        # to 35k ft to estimate dynamic viscosity)
        self.add_subsystem("Re_calc", om.ExecComp("re = rho * u / (-3.329134*10**(-10) * h + 1.792398*10**(-5))",
                                                  re={"units": "1/m", "val": 1e6},
                                                  rho={"units": "kg/m**3", "val": 1.},
                                                  u={"units": "m/s", "val": 100.},
                                                  h={"units": "m", "val": 1.}),
                           promotes_inputs=[("h", "fltcond|h")])
        self.connect('density.fltcond|rho', 'Re_calc.rho')
        self.connect('airspeed.Utrue', 'Re_calc.u')

        # =================================================================
        #                       Call OpenAeroStruct
        # =================================================================
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

        # Overwrite any options in the surface dict with those provided in the options
        if self.options['surf_options'] is not None:
            for key in self.options['surf_options']:
                surf_dict[key] = self.options['surf_options'][key]

        self.add_subsystem("aero_point", AeroPoint(surfaces=[surf_dict]),
                           promotes_inputs=[("Mach_number", "fltcond|M"), ("alpha", "fltcond|alpha")],
                           promotes_outputs=[(f"{surf_dict['name']}_perf.CD", "fltcond|CD"),
                                             (f"{surf_dict['name']}_perf.CL", "fltcond|CL")])
        self.connect("twist_mesh.mesh", [f"aero_point.{surf_dict['name']}.def_mesh",
                                         f"aero_point.aero_states.{surf_dict['name']}_def_mesh"])
        self.connect('airspeed.Utrue', 'aero_point.v')
        self.connect('density.fltcond|rho', 'aero_point.rho')
        self.connect("Re_calc.re", "aero_point.re")

        # Set input defaults for inputs that go to multiple locations
        self.set_input_defaults('fltcond|M', 0.1)
        self.set_input_defaults('fltcond|alpha', 0.)

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


if __name__=="__main__":
    alpha_list = np.linspace(-10, 10, 200)
    Mach_list = np.array([0.1])#np.linspace(0.01, 0.9, 100)
    alt_list = np.array([.1])#0.3048*np.linspace(0, 40000, 200)
    
    inputs = {}
    inputs['alpha_grid'], inputs['Mach_number_grid'], inputs['alt_grid'] = np.meshgrid(alpha_list,
                                                                                       Mach_list,
                                                                                       alt_list,
                                                                                       indexing='ij')
    inputs['TempIncrement'] = 0
    inputs['S_ref'] = 427.8
    inputs['AR'] = 9.82
    inputs['taper'] = 0.149
    inputs['c4sweep'] = 31.6
    inputs['twist'] = np.array([-3, 1.5, 6])
    inputs['num_x'] = 3
    inputs['num_y'] = 5

    res = compute_training_data(inputs)

    import matplotlib.pyplot as plt
    plt.plot(res['CD'][:,0,0], res['CL'][:,0,0])
    plt.xlabel('CD')
    plt.ylabel('CL')
    plt.show()

    