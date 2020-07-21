import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.analysis.aerodynamics import PolarDrag, Lift, StallSpeed

class PolarDragTestGroup(Group):
    """
    This is a simple analysis group for testing the drag polar component
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")
    def setup(self):
        nn = self.options['num_nodes']
        iv = self.add_subsystem('conditions', IndepVarComp(),promotes_outputs=['*'])
        self.add_subsystem('polardrag', PolarDrag(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])

        iv.add_output('fltcond|CL', val=np.linspace(0,1.5,nn))
        iv.add_output('fltcond|q', val=np.ones(nn)*0.5*1.225*70**2, units='N * m**-2')
        iv.add_output('ac|geom|wing|S_ref', val=30, units='m**2')
        iv.add_output('ac|geom|wing|AR', val=15)
        iv.add_output('CD0', val=0.02)
        iv.add_output('e', val=0.8)

class VectorDragTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(PolarDragTestGroup(num_nodes=5))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_drag_vectorial(self):
        drag_cl0 = 0.5*1.225*70**2 * 30 *(0.02 + 0**2 / np.pi / 0.8 / 15)
        drag_cl1p5 = 0.5*1.225*70**2 * 30 *(0.02 + 1.5**2 / np.pi / 0.8 / 15)
        assert_near_equal(self.prob['drag'][0], drag_cl0, tolerance=1e-8)
        assert_near_equal(self.prob['drag'][-1], drag_cl1p5, tolerance=1e-8)


    def test_partials(self):
        partials = self.prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class ScalarDragTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(PolarDragTestGroup(num_nodes=1))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_drag_scalar(self):
        drag_cl0 = 0.5*1.225*70**2 * 30 *(0.02 + 0**2 / np.pi / 0.8 / 15)
        assert_near_equal(self.prob['drag'], drag_cl0, tolerance=1e-8)

    def test_partials(self):
        partials = self.prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class LiftTestGroup(Group):
    """
    This is a simple analysis group for testing the lift component
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")
    def setup(self):
        nn = self.options['num_nodes']
        iv = self.add_subsystem('conditions', IndepVarComp(),promotes_outputs=['*'])
        self.add_subsystem('lift', Lift(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])

        iv.add_output('fltcond|CL', val=np.linspace(1.5,0,nn))
        iv.add_output('fltcond|q', val=np.ones(nn)*0.5*1.225*70**2, units='N * m**-2')
        iv.add_output('ac|geom|wing|S_ref', val=30, units='m**2')

class VectorLiftTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(LiftTestGroup(num_nodes=5))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_lift_vectorial(self):
        lift_cl0 = 0
        lift_cl1p5 = 0.5*1.225*70**2 * 30 * 1.5
        assert_near_equal(self.prob['lift'][-1], lift_cl0, tolerance=1e-8)
        assert_near_equal(self.prob['lift'][0], lift_cl1p5, tolerance=1e-8)

    def test_partials(self):
        partials = self.prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class ScalarLiftTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(LiftTestGroup(num_nodes=1))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_lift_scalar(self):
        lift_cl1p5 = 0.5*1.225*70**2 * 30 * 1.5
        assert_near_equal(self.prob['lift'], lift_cl1p5, tolerance=1e-8)

    def test_partials(self):
        partials = self.prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)


class StallSpeedTestGroup(Group):
    """
    This is a simple analysis group for testing the stall speed component
    """
    def setup(self):
        iv = self.add_subsystem('conditions', IndepVarComp(),promotes_outputs=['*'])
        self.add_subsystem('stall', StallSpeed(),promotes_inputs=['*'],promotes_outputs=['*'])

        iv.add_output('CLmax', val=2.5)
        iv.add_output('weight', val=1000, units='kg')
        iv.add_output('ac|geom|wing|S_ref', val=30, units='m**2')

class StallSpeedTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(StallSpeedTestGroup())
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_stall_speed(self):
        vstall = np.sqrt(2 * 1000 * 9.80665 / 1.225 / 30 / 2.5)
        assert_near_equal(self.prob['Vstall_eas'], vstall, tolerance=1e-8)

    def test_partials(self):
        partials = self.prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)
