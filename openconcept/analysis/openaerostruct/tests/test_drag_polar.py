import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.analysis.openaerostruct.drag_polar import *

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

        partials = p.check_partials()
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

        partials = p.check_partials(form='central')
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
        
        print(mesh)
        
        assert_near_equal(p.get_val('mesh', units='m'), mesh)

        partials = p.check_partials()
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

        partials = p.check_partials()
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

        partials = p.check_partials()
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

        partials = p.check_partials()
        assert_check_partials(partials, atol=2e-5)
