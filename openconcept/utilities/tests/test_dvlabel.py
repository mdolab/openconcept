from __future__ import print_function, division, absolute_import

import unittest
import imp
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openconcept.utilities.dvlabel import DVLabel
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

class TestBasic(unittest.TestCase):

    def setUp(self):
        self.nn = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a_to_be_renamed', shape=(self.nn,))
        ivc.add_output(name='b_to_be_renamed', shape=(self.nn,))
        dvlabel = DVLabel([['a_to_be_renamed','a',np.ones(self.nn),None],
                           ['b_to_be_renamed','b',np.ones(self.nn),None]])

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])
        self.p.model.add_subsystem(name='dvlabel',
                                   subsys=dvlabel,
                                   promotes_inputs=['*'],
                                   promotes_outputs=['*'])
        self.p.setup()

        self.p['a_to_be_renamed'] = np.random.rand(self.nn,)
        self.p['b_to_be_renamed'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a_in = self.p['a_to_be_renamed']
        b_in = self.p['b_to_be_renamed']
        a_out = self.p['a']
        b_out = self.p['b']
        assert_near_equal(a_in, a_out,1e-16)
        assert_near_equal(b_in, b_out,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a_to_be_renamed', shape=(self.nn,), units='m/s')
        ivc.add_output(name='b_to_be_renamed', shape=(self.nn,), units='kg')
        dvlabel = DVLabel([['a_to_be_renamed','a',np.ones(self.nn),'m/s'],
                           ['b_to_be_renamed','b',np.ones(self.nn),'lbm']])

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])
        self.p.model.add_subsystem(name='dvlabel',
                                   subsys=dvlabel,
                                   promotes_inputs=['*'],
                                   promotes_outputs=['*'])
        self.p.setup()

        self.p['a_to_be_renamed'] = np.random.rand(self.nn,)
        self.p['b_to_be_renamed'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a_in = self.p['a_to_be_renamed']
        b_in = self.p['b_to_be_renamed']
        a_out = self.p['a']
        b_out = self.p['b']
        assert_near_equal(a_in, a_out,1e-16)
        assert_near_equal(b_in*2.20462, b_out,1e-5)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestScalars(unittest.TestCase):

    def setUp(self):
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a_to_be_renamed', units='m/s')
        ivc.add_output(name='b_to_be_renamed', units='kg')
        dvlabel = DVLabel([['a_to_be_renamed','a',1,'m/s'],
                           ['b_to_be_renamed','b',1,'lbm']])

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])
        self.p.model.add_subsystem(name='dvlabel',
                                   subsys=dvlabel,
                                   promotes_inputs=['*'],
                                   promotes_outputs=['*'])
        self.p.setup()

        self.p['a_to_be_renamed'] = np.random.rand(1)
        self.p['b_to_be_renamed'] = np.random.rand(1)

        self.p.run_model()

    def test_results(self):
        a_in = self.p['a_to_be_renamed']
        b_in = self.p['b_to_be_renamed']
        a_out = self.p['a']
        b_out = self.p['b']
        assert_near_equal(a_in, a_out,1e-16)
        assert_near_equal(b_in*2.20462, b_out,1e-5)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

if __name__ == '__main__':
    unittest.main()