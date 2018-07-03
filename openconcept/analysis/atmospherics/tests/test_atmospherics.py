import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties

class AtmosTestGroup(Group):
    """This computes pressure, temperature, and density for a given altitude at ISA condtions. Also true airspeed from equivalent ~ indicated airspeed
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")
    def setup(self):
        nn = self.options['num_nodes']
        iv = self.add_subsystem('conditions', IndepVarComp())
        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn),promotes_outputs=['*'])
        iv.add_output('h', val=np.linspace(0,30000,nn), units='ft')
        iv.add_output('Ueas', val=np.ones(nn)*120, units='kn')
        self.connect('conditions.h','atmos.fltcond|h')
        self.connect('conditions.Ueas','atmos.fltcond|Ueas')

class VectorAtmosTestCase(unittest.TestCase):
    def test_sea_level(self):
        prob = Problem(AtmosTestGroup(num_nodes=5))
        prob.setup(check=True)
        prob.run_model()
        #check conditions at sea level
        assert_rel_error(self,prob['fltcond|rho'][0],1.225,tolerance=1e-4)
        assert_rel_error(self,prob['fltcond|p'][0],101325,tolerance=1e-2)
        assert_rel_error(self,prob['fltcond|T'][0],288.15,tolerance=1e-2)
        assert_rel_error(self,prob['fltcond|Utrue'][0],61.7333,tolerance=1e-3)
        assert_rel_error(self,prob['fltcond|q'][0],2334.2398,tolerance=1e-3)

        #check conditions at 30kft (1976 standard atmosphere verified at https://www.digitaldutch.com/atmoscalc/)
        assert_rel_error(self,prob['fltcond|rho'][-1],0.458312,tolerance=1e-4)
        assert_rel_error(self,prob['fltcond|p'][-1],30089.6,tolerance=1e-2)
        assert_rel_error(self,prob['fltcond|T'][-1],228.714,tolerance=1e-2)
        assert_rel_error(self,prob['fltcond|Utrue'][-1],61.7333*np.sqrt(1.225/0.458312),tolerance=1e-3)
        assert_rel_error(self,prob['fltcond|q'][-1],2334.2398,tolerance=1e-3)

class ScalarAtmosTestCase(unittest.TestCase):
    def test_sea_level(self):
        prob = Problem(AtmosTestGroup(num_nodes=1))
        prob.setup(check=True)
        prob.run_model()
        # prob.model.list_inputs(units=True,print_arrays=True)
        # prob.model.list_outputs(units=True,print_arrays=True)
        assert_rel_error(self,prob['fltcond|rho'][0],1.225,tolerance=1e-4)
        assert_rel_error(self,prob['fltcond|p'][0],101325,tolerance=1e-2)
        assert_rel_error(self,prob['fltcond|T'][0],288.15,tolerance=1e-2)
        assert_rel_error(self,prob['fltcond|Utrue'][0],61.7333,tolerance=1e-3)
        assert_rel_error(self,prob['fltcond|q'][0],2334.2398,tolerance=1e-3)