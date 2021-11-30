import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om

# Only run if OpenAeroStruct is installed
try:
    from openaerostruct.geometry.geometry_group import Geometry
    from openaerostruct.aerodynamics.aero_groups import AeroPoint
    from openconcept.analysis.openaerostruct.drag_polar import *
    OAS_installed = True
except:
    OAS_installed = False

@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class OASDragPolarTestCase(unittest.TestCase):
    def test(self):
        twist = np.array([-1, -0.5, 2])

        # Generate mesh to pass to OpenAeroStruct
        mesh = om.Problem(VLM(num_x=3, num_y=5, num_twist=twist.size))
        mesh.setup()
        mesh.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        mesh.set_val('ac|geom|wing|AR', 10)
        mesh.set_val('ac|geom|wing|taper', 0.1)
        mesh.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        mesh.set_val('ac|geom|wing|twist', twist, units='deg')
        mesh.set_val('fltcond|M', 0.45)
        mesh.set_val('fltcond|h', 7.5e3, units='m')
        mesh.set_val('fltcond|alpha', 2, units='deg')
        mesh.run_model()

        p = om.Problem(OASDragPolar(num_nodes=1, num_x=3, num_y=5, num_twist=twist.size, Mach_train=np.linspace(0.1, 0.8, 3),
                                  alpha_train=np.linspace(-11, 15, 3), alt_train=np.linspace(0, 15e3, 2)))
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val('fltcond|TempIncrement', 0, units='degC')
        p.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        p.set_val('ac|geom|wing|AR', 10)
        p.set_val('ac|geom|wing|taper', 0.1)
        p.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        p.set_val('ac|geom|wing|twist', twist, units='deg')
        p.set_val('ac|aero|CD_nonwing', 0.01)
        p.set_val('fltcond|q', 5e3, units='Pa')
        p.set_val('fltcond|M', 0.45)
        p.set_val('fltcond|h', 7.5e3, units='m')
        p.set_val('fltcond|CL', mesh.get_val('fltcond|CL'))
        p.run_model()

        # Test on training point
        assert_near_equal(mesh.get_val('fltcond|CL'), p.get_val('aero_surrogate.CL'), tolerance=1e-10)  # check convergence
        assert_near_equal(2, p.get_val('alpha_bal.alpha', units='deg'), tolerance=1e-7)
        assert_near_equal(mesh.get_val('fltcond|CD') + 0.01, p.get_val('aero_surrogate.CD'), tolerance=2e-2)
        assert_near_equal(p.get_val('drag', units='N'), p.get_val('aero_surrogate.CD') * 100 * 5e3, tolerance=2e-2)

        # Test off training point
        mesh.set_val('fltcond|M', 0.3)
        mesh.set_val('fltcond|h', 4e3, units='m')
        mesh.set_val('fltcond|alpha', 6, units='deg')
        mesh.run_model()

        p.set_val('fltcond|M', 0.3)
        p.set_val('fltcond|h', 4e3, units='m')
        p.set_val('fltcond|CL', mesh.get_val('fltcond|CL'))
        p.run_model()

        assert_near_equal(mesh.get_val('fltcond|CL'), p.get_val('aero_surrogate.CL'), tolerance=1e-10)  # check convergence
        assert_near_equal(6, p.get_val('alpha_bal.alpha', units='deg'), tolerance=1e-2)
        assert_near_equal(mesh.get_val('fltcond|CD') + 0.01, p.get_val('aero_surrogate.CD'), tolerance=5e-2)
        assert_near_equal(p.get_val('drag', units='N'), p.get_val('aero_surrogate.CD') * 100 * 5e3, tolerance=5e-2)
    
    def test_surf_options(self):
        nn = 1
        twist = np.array([-1, 0, 1])
        p = om.Problem(OASDragPolar(num_nodes=nn, num_x=3, num_y=5, num_twist=twist.size, Mach_train=np.linspace(0.1, 0.8, 2),
                                  alpha_train=np.linspace(-11, 15, 2), alt_train=np.linspace(0, 15e3, 2), surf_options={'k_lam': 0.9}))
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val('fltcond|TempIncrement', 0, units='degC')
        p.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        p.set_val('ac|geom|wing|AR', 10)
        p.set_val('ac|geom|wing|taper', 0.1)
        p.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        p.set_val('ac|geom|wing|twist', twist, units='deg')
        p.set_val('ac|aero|CD_nonwing', 0.01)
        p.set_val('fltcond|q', 5e3*np.ones(nn), units='Pa')
        p.set_val('fltcond|M', 0.5*np.ones(nn))
        p.set_val('fltcond|h', 7.5e3*np.ones(nn), units='m')
        p.set_val('fltcond|CL', 0.5*np.ones(nn))
        p.run_model()

        # Ensure they're all the same
        assert_near_equal(p.get_val('drag', units='N'), 34962.6043231*np.ones(nn), tolerance=1e-10)

    
    def test_vectorized(self):
        nn = 7
        twist = np.array([-1, 0, 1])
        p = om.Problem(OASDragPolar(num_nodes=nn, num_x=3, num_y=5, num_twist=twist.size, Mach_train=np.linspace(0.1, 0.8, 2),
                                  alpha_train=np.linspace(-11, 15, 2), alt_train=np.linspace(0, 15e3, 2)))
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val('fltcond|TempIncrement', 0, units='degC')
        p.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        p.set_val('ac|geom|wing|AR', 10)
        p.set_val('ac|geom|wing|taper', 0.1)
        p.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        p.set_val('ac|geom|wing|twist', twist, units='deg')
        p.set_val('ac|aero|CD_nonwing', 0.01)
        p.set_val('fltcond|q', 5e3*np.ones(nn), units='Pa')
        p.set_val('fltcond|M', 0.5*np.ones(nn))
        p.set_val('fltcond|h', 7.5e3*np.ones(nn), units='m')
        p.set_val('fltcond|CL', 0.5*np.ones(nn))
        p.run_model()

        # Ensure they're all the same
        assert_near_equal(p.get_val('drag', units='N'), 37845.94053713*np.ones(nn), tolerance=1e-10)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class VLMDataGenTestCase(unittest.TestCase):
    def test_defaults(self):
        # Regression test
        twist = np.array([-1, -0.5, 2])
        p = om.Problem(VLMDataGen(num_x=3, num_y=5, num_twist=twist.size, Mach_train=np.linspace(0.1, 0.85, 2),
                                  alpha_train=np.linspace(-10, 15, 2), alt_train=np.linspace(0, 15e3, 2)))
        p.setup()
        p.set_val('fltcond|TempIncrement', 0, units='degC')
        p.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        p.set_val('ac|geom|wing|AR', 10)
        p.set_val('ac|geom|wing|taper', 0.1)
        p.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        p.set_val('ac|geom|wing|twist', twist, units='deg')
        p.set_val('ac|aero|CD_nonwing', 0.01)
        p.run_model()

        CL = np.array([[[-0.79879583, -0.79879583],
                        [ 1.31170126,  1.31170126]],
                       [[-0.79879583, -0.79879583],
                        [ 1.31170126,  1.31170126]]])
        CD = np.array([[[0.03465792, 0.03701483],
                        [0.06816224, 0.07051915]],
                       [[0.03455214, 0.03648223],
                        [0.20238882, 0.20431891]]])

        assert_near_equal(CL, p.get_val('CL_train'), tolerance=1e-7)
        assert_near_equal(CD, p.get_val('CD_train'), tolerance=1e-7)

        partials = p.check_partials(out_stream=None, form='central')
        assert_check_partials(partials, atol=6e-5, rtol=2e-5)
    
    def test_different_surf_options(self):
        # Test that when there are different surf_options within a single model it catches it
        p = om.Problem()
        p.model.add_subsystem('one', VLMDataGen(surf_options={'a': 1.13521, 'b': np.linspace(0, 1, 10)}))
        p.model.add_subsystem('two', VLMDataGen(surf_options={'a': 1.13521, 'b': np.linspace(0, 1, 10)}))
        p.model.add_subsystem('three', VLMDataGen(surf_options={'a': 1.13521, 'b': np.linspace(0, 1, 10)}))
        p.setup()

        p = om.Problem()
        p.model.add_subsystem('one', VLMDataGen(surf_options={'a': 1.13521}))
        p.model.add_subsystem('two', VLMDataGen(surf_options={'a': 1.1352}))
        self.assertRaises(ValueError, p.setup)

        p = om.Problem()
        p.model.add_subsystem('one', VLMDataGen(surf_options={'a': 1.13521, 'b': np.linspace(0, 1, 10)}))
        p.model.add_subsystem('two', VLMDataGen(surf_options={'a': 1.13521, 'b': np.linspace(0, 1.0001, 10)}))
        p.model.add_subsystem('three', VLMDataGen(surf_options={'a': 1.13521, 'b': np.linspace(0, 1, 10)}))
        self.assertRaises(ValueError, p.setup)

        p = om.Problem()
        p.model.add_subsystem('one', VLMDataGen())
        p.model.add_subsystem('two', VLMDataGen(surf_options={'boof': True}))
        self.assertRaises(ValueError, p.setup)

@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class VLMTestCase(unittest.TestCase):
    def test_defaults(self):
        twist = np.array([-1, -0.5, 2])
        p = om.Problem(VLM(num_x=3, num_y=5, num_twist=twist.size))
        p.setup()
        p.set_val('fltcond|alpha', 2, units='deg')
        p.set_val('fltcond|M', 0.6)
        p.set_val('fltcond|h', 5e3, units='m')
        p.set_val('fltcond|TempIncrement', 0, units='degC')
        p.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        p.set_val('ac|geom|wing|AR', 10)
        p.set_val('ac|geom|wing|taper', 0.1)
        p.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        p.set_val('ac|geom|wing|twist', twist, units='deg')

        p.run_model()

        # Run OpenAeroStruct with the same inputs
        inputs = {}
        inputs['mesh'] = p.get_val('mesh.mesh', units='m')
        inputs['twist'] = twist
        inputs['v'] = p.get_val('airspeed.Utrue', units='m/s')
        inputs['alpha'] = p.get_val('fltcond|alpha', units='deg')
        inputs['Mach_number'] = p.get_val('fltcond|M')
        inputs['re'] = p.get_val('Re_calc.re', units='1/m')
        inputs['rho'] = p.get_val('density.fltcond|rho', units='kg/m**3')

        exact = run_OAS(inputs)

        assert_near_equal(exact['CL'], p.get_val('fltcond|CL'))
        assert_near_equal(exact['CD'], p.get_val('fltcond|CD'))

        partials = p.check_partials(out_stream=None, form='central')
        assert_check_partials(partials, atol=6e-5, rtol=2e-5)
    
    def test_wave_drag(self):
        twist = np.array([-1, -0.5, 2])
        p = om.Problem(VLM(num_x=3, num_y=5, num_twist=twist.size, surf_options={'with_wave': False}))
        p.setup()
        p.set_val('fltcond|alpha', 2, units='deg')
        p.set_val('fltcond|M', 0.85)
        p.set_val('fltcond|h', 5e3, units='m')
        p.set_val('fltcond|TempIncrement', 0, units='degC')
        p.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        p.set_val('ac|geom|wing|AR', 10)
        p.set_val('ac|geom|wing|taper', 0.1)
        p.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        p.set_val('ac|geom|wing|twist', twist, units='deg')

        p.run_model()

        # Run OpenAeroStruct with the same inputs
        inputs = {}
        inputs['mesh'] = p.get_val('mesh.mesh', units='m')
        inputs['twist'] = twist
        inputs['v'] = p.get_val('airspeed.Utrue', units='m/s')
        inputs['alpha'] = p.get_val('fltcond|alpha', units='deg')
        inputs['Mach_number'] = p.get_val('fltcond|M')
        inputs['re'] = p.get_val('Re_calc.re', units='1/m')
        inputs['rho'] = p.get_val('density.fltcond|rho', units='kg/m**3')

        exact = run_OAS(inputs, with_wave=False)

        assert_near_equal(exact['CL'], p.get_val('fltcond|CL'))
        assert_near_equal(exact['CD'], p.get_val('fltcond|CD'))
    
    def test_viscous_drag(self):
        twist = np.array([-1, -0.5, 2])
        p = om.Problem(VLM(num_x=3, num_y=5, num_twist=twist.size, surf_options={'with_viscous': False}))
        p.setup()
        p.set_val('fltcond|alpha', 2, units='deg')
        p.set_val('fltcond|M', 0.85)
        p.set_val('fltcond|h', 5e3, units='m')
        p.set_val('fltcond|TempIncrement', 0, units='degC')
        p.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        p.set_val('ac|geom|wing|AR', 10)
        p.set_val('ac|geom|wing|taper', 0.1)
        p.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        p.set_val('ac|geom|wing|twist', twist, units='deg')

        p.run_model()

        # Run OpenAeroStruct with the same inputs
        inputs = {}
        inputs['mesh'] = p.get_val('mesh.mesh', units='m')
        inputs['twist'] = twist
        inputs['v'] = p.get_val('airspeed.Utrue', units='m/s')
        inputs['alpha'] = p.get_val('fltcond|alpha', units='deg')
        inputs['Mach_number'] = p.get_val('fltcond|M')
        inputs['re'] = p.get_val('Re_calc.re', units='1/m')
        inputs['rho'] = p.get_val('density.fltcond|rho', units='kg/m**3')

        exact = run_OAS(inputs, with_viscous=False)

        assert_near_equal(exact['CL'], p.get_val('fltcond|CL'))
        assert_near_equal(exact['CD'], p.get_val('fltcond|CD'))
    
    def test_t_over_c(self):
        twist = np.array([-1, -0.5, 2])
        p = om.Problem(VLM(num_x=3, num_y=3, num_twist=twist.size, surf_options={'t_over_c': np.array([0.1, 0.2])}))
        p.setup()
        p.set_val('fltcond|alpha', 2, units='deg')
        p.set_val('fltcond|M', 0.85)
        p.set_val('fltcond|h', 5e3, units='m')
        p.set_val('fltcond|TempIncrement', 0, units='degC')
        p.set_val('ac|geom|wing|S_ref', 100, units='m**2')
        p.set_val('ac|geom|wing|AR', 10)
        p.set_val('ac|geom|wing|taper', 0.1)
        p.set_val('ac|geom|wing|c4sweep', 20, units='deg')
        p.set_val('ac|geom|wing|twist', twist, units='deg')

        p.run_model()

        # Run OpenAeroStruct with the same inputs
        inputs = {}
        inputs['mesh'] = p.get_val('mesh.mesh', units='m')
        inputs['twist'] = twist
        inputs['v'] = p.get_val('airspeed.Utrue', units='m/s')
        inputs['alpha'] = p.get_val('fltcond|alpha', units='deg')
        inputs['Mach_number'] = p.get_val('fltcond|M')
        inputs['re'] = p.get_val('Re_calc.re', units='1/m')
        inputs['rho'] = p.get_val('density.fltcond|rho', units='kg/m**3')

        exact = run_OAS(inputs, t_over_c=np.array([0.1, 0.2]))

        assert_near_equal(exact['CL'], p.get_val('fltcond|CL'))
        assert_near_equal(exact['CD'], p.get_val('fltcond|CD'))    

@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class PlanformMeshTestCase(unittest.TestCase):
    def test_easy(self):
        nx = 3
        ny = 5
        p = om.Problem(PlanformMesh(num_x=nx, num_y=ny))
        p.setup()
        p.set_val('S', 2, units='m**2')
        p.set_val('AR', 2)
        p.set_val('taper', 1.)
        p.set_val('sweep', 0., units='deg')
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0], mesh[:, :, 1] = np.meshgrid(np.linspace(-0.25, 0.75, nx),
                                                   np.linspace(-1, 0, ny), indexing='ij')
        
        assert_near_equal(p.get_val('mesh', units='m'), mesh)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)
    
    def test_S_AR(self):
        nx = 3
        ny = 5
        p = om.Problem(PlanformMesh(num_x=nx, num_y=ny))
        p.setup(force_alloc_complex=True)
        p.set_val('S', 48, units='m**2')
        p.set_val('AR', 3)
        p.set_val('taper', 1.)
        p.set_val('sweep', 0., units='deg')
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0], mesh[:, :, 1] = np.meshgrid(np.linspace(-1, 3, nx),
                                                   np.linspace(-6, 0, ny), indexing='ij')
        
        assert_near_equal(p.get_val('mesh', units='m'), mesh)

        partials = p.check_partials(out_stream=None, form='central')
        assert_check_partials(partials)
    
    def test_taper(self):
        nx = 2
        ny = 3
        p = om.Problem(PlanformMesh(num_x=nx, num_y=ny))
        p.setup()
        p.set_val('S', 1.3, units='m**2')
        p.set_val('AR', 4/1.3)  # pick S and AR for half span and root chord of 1
        p.set_val('taper', .3)
        p.set_val('sweep', 0., units='deg')
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.array([[-0.075, -0.1625, -0.25],
                                  [0.225, 0.4875, 0.75]])
        mesh[:, :, 1] = np.array([[-1, -0.5, 0],
                                  [-1, -0.5, 0]])
        
        assert_near_equal(p.get_val('mesh', units='m'), mesh)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)
    
    def test_sweep(self):
        nx = 3
        ny = 3
        p = om.Problem(PlanformMesh(num_x=nx, num_y=ny))
        p.setup()
        p.set_val('S', 2, units='m**2')
        p.set_val('AR', 2)
        p.set_val('taper', 1.)
        p.set_val('sweep', 45., units='deg')
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0], mesh[:, :, 1] = np.meshgrid(np.linspace(-0.25, 0.75, nx),
                                                   np.linspace(-1, 0, ny), indexing='ij')
        
        mesh[:, 0, 0] += 1
        mesh[:, 1, 0] += .5
        
        assert_near_equal(p.get_val('mesh', units='m'), mesh)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_taper_sweep(self):
        nx = 2
        ny = 3
        p = om.Problem(PlanformMesh(num_x=nx, num_y=ny))
        p.setup()
        p.set_val('S', 1.3, units='m**2')
        p.set_val('AR', 4/1.3)  # pick S and AR for half span and root chord of 1
        p.set_val('taper', .3)
        p.set_val('sweep', 45., units='deg')
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.array([[-0.075, -0.1625, -0.25],
                                  [0.225, 0.4875, 0.75]])
        mesh[:, :, 1] = np.array([[-1, -0.5, 0],
                                  [-1, -0.5, 0]])
        mesh[:, 0, 0] += 1
        mesh[:, 1, 0] += .5
        
        assert_near_equal(p.get_val('mesh', units='m'), mesh)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)
    
    def test_777ish_regression(self):
        nx = 3
        ny = 4
        p = om.Problem(PlanformMesh(num_x=nx, num_y=ny))
        p.setup()
        p.set_val('S', 427.8, units='m**2')
        p.set_val('AR', 9.82)
        p.set_val('taper', .149)
        p.set_val('sweep', 31.6, units='deg')
        p.run_model()

        mesh = np.array([[[ 19.50929722, -32.40754542,   0.        ],
                          [ 12.04879827, -21.60503028,   0.        ],
                          [  4.58829932, -10.80251514,   0.        ],
                          [ -2.87219963,   0.,           0.        ]],
                         [[ 20.36521271, -32.40754542,   0.        ],
                          [ 14.53420835, -21.60503028,   0.        ],
                          [  8.70320399, -10.80251514,   0.        ],
                          [  2.87219963,   0.,           0.        ]],
                         [[ 21.2211282,  -32.40754542,   0.        ],
                          [ 17.01961843, -21.60503028,   0.        ],
                          [ 12.81810866, -10.80251514,   0.        ],
                          [  8.61659889,   0.,           0.        ]]])
        
        assert_near_equal(p.get_val('mesh', units='m'), mesh, tolerance=1e-10)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials, atol=2e-5)


"""
Runs OpenAeroStruct with flight condition and mesh inputs.

Inputs
------
inputs : dict
    Input dictionary containing
        mesh : ndarray
            Flat wing mesh (m)
        twist : ndarray
            Twist control points (deg)
        v : float
            Flight speed (m/s)
        alpha : float
            Angle of attack (deg)
        Mach_number : float
            Mach number
        re : float
            Dimensional Reynolds number (1/m)
        rho : float
            Flow density (kg/m^3)
with_viscous : bool (optional)
    Include viscous drag
with_wave : bool (optional)
    Include wave drag
t_over_c : float (optional)
    Thickness to chord ratio of the airfoil

Outputs
-------
outputs : dict
    Output dictionary containing
        CL : float
            Lift coefficient
        CD : float
            Drag coefficient
"""
def run_OAS(inputs, with_viscous=True, with_wave=True, t_over_c=np.array([.12])):
    # Create a dictionary with info and options about the aerodynamic
    # lifting surface
    surface = {
                # Wing definition
                'name' : 'wing',        # name of the surface
                'symmetry' : True,     # if true, model one half of wing
                                        # reflected across the plane y = 0
                'S_ref_type' : 'projected', # how we compute the wing area,
                                        # can be 'wetted' or 'projected'

                'twist_cp' : inputs['twist'],
                'mesh' : inputs['mesh'],

                # Aerodynamic performance of the lifting surface at
                # an angle of attack of 0 (alpha=0).
                # These CL0 and CD0 values are added to the CL and CD
                # obtained from aerodynamic analysis of the surface to get
                # the total CL and CD.
                # These CL0 and CD0 values do not vary wrt alpha.
                'CL0' : 0.0,            # CL of the surface at alpha=0
                'CD0' : 0.0,            # CD of the surface at alpha=0

                # Airfoil properties for viscous drag calculation
                'k_lam' : 0.05,         # percentage of chord with laminar
                                        # flow, used for viscous drag
                't_over_c_cp' : t_over_c,      # thickness over chord ratio (NACA0015)
                'c_max_t' : .37,       # chordwise location of maximum (NACA0015)
                                        # thickness
                'with_viscous' : with_viscous,  # if true, compute viscous drag
                'with_wave' : with_wave,     # if true, compute wave drag
                }

    # Create the OpenMDAO problem
    prob = om.Problem()

    # Create an independent variable component that will supply the flow
    # conditions to the problem.
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output('v', val=inputs['v'], units='m/s')
    indep_var_comp.add_output('alpha', val=inputs['alpha'], units='deg')
    indep_var_comp.add_output('Mach_number', val=inputs['Mach_number'])
    indep_var_comp.add_output('re', val=inputs['re'], units='1/m')
    indep_var_comp.add_output('rho', val=inputs['rho'], units='kg/m**3')
    indep_var_comp.add_output('cg', val=np.zeros((3)), units='m')

    # Add this IndepVarComp to the problem model
    prob.model.add_subsystem('prob_vars',
        indep_var_comp,
        promotes=['*'])

    # Create and add a group that handles the geometry for the
    # aerodynamic lifting surface
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(surface['name'], geom_group)

    # Create the aero point group, which contains the actual aerodynamic
    # analyses
    aero_group = AeroPoint(surfaces=[surface])
    point_name = 'aero_point_0'
    prob.model.add_subsystem(point_name, aero_group,
        promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'cg'])

    name = surface['name']

    # Connect the mesh from the geometry component to the analysis point
    prob.model.connect(name + '.mesh', point_name + '.' + name + '.def_mesh')

    # Perform the connections with the modified names within the
    # 'aero_states' group.
    prob.model.connect(name + '.mesh', point_name + '.aero_states.' + name + '_def_mesh')
    prob.model.connect(name + '.t_over_c', point_name + '.' + name + '_perf.' + 't_over_c')

    # Set up and run the model
    prob.setup()
    prob.run_model()
    outputs = {}
    outputs['CL'] = prob['aero_point_0.wing_perf.CL']
    outputs['CD'] = prob['aero_point_0.wing_perf.CD']
    return outputs



if __name__=="__main__":
    unittest.main()