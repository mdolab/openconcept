from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
#from openmdao.components.sum_comp import SumComp
from openconcept.utilities.math.sum_comp import SumComp
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

class TestSummation1x1(unittest.TestCase):
    # this test case is nonsensical but should still pass
    def setUp(self):
        self.nn = 1
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp())
        multi.add_equation('sum_output','sum_input')

        self.p.model.connect('a', 'sum_comp.sum_input')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        out = self.p['sum_comp.sum_output']
        expected = np.sum(a, axis=0)
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSummation1x1AxisNone(unittest.TestCase):
    # this test case is nonsensical but should still pass
    def setUp(self):
        self.nn = 1
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp(axis=None))
        multi.add_equation('sum_output','sum_input')

        self.p.model.connect('a', 'sum_comp.sum_input')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        out = self.p['sum_comp.sum_output']
        expected = np.sum(a, axis=None)
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSummationNx1(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp())
        multi.add_equation('sum_output','sum_input',vec_size=self.nn)

        self.p.model.connect('a', 'sum_comp.sum_input')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        out = self.p['sum_comp.sum_output']
        expected = np.sum(a, axis=0)
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSummationNx1AxisNone(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp(axis=None))
        multi.add_equation('sum_output','sum_input',vec_size=self.nn)

        self.p.model.connect('a', 'sum_comp.sum_input')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        out = self.p['sum_comp.sum_output']
        expected = np.sum(a, axis=None)
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSummationNx3(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.sf = -2
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp())
        multi.add_equation('sum_output','sum_input',vec_size=self.nn,length=self.length,scaling_factor=self.sf)

        self.p.model.connect('a', 'sum_comp.sum_input')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        out = self.p['sum_comp.sum_output']
        expected = self.sf*np.sum(a, axis=0)
        expected = expected.reshape((1,self.length))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSummationNx3Axis1(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.sf = -2
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp(axis=1))
        multi.add_equation('sum_output','sum_input',vec_size=self.nn,length=self.length,scaling_factor=self.sf)

        self.p.model.connect('a', 'sum_comp.sum_input')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        out = self.p['sum_comp.sum_output']
        expected = self.sf*np.sum(a, axis=1)
        expected = expected.reshape((self.nn,))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSummationNx3AxisNone(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.sf = -2
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp(axis=None))
        multi.add_equation('sum_output','sum_input',vec_size=self.nn,length=self.length,scaling_factor=self.sf)

        self.p.model.connect('a', 'sum_comp.sum_input')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        out = self.p['sum_comp.sum_output']
        expected = self.sf*np.sum(a, axis=None)
        expected = expected.reshape((1,))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSummationNx3UnitsMultipleSystems(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length), units='m')
        ivc.add_output(name='b', shape=(self.nn,self.length), units='kg')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a','b'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp())
        multi.add_equation('sum_output1','sum_input_a',vec_size=self.nn,length=self.length, units='m')
        multi.add_equation('sum_output2','sum_input_b',vec_size=self.nn,length=self.length, units='kg')

        self.p.model.connect('a', 'sum_comp.sum_input_a')
        self.p.model.connect('b', 'sum_comp.sum_input_b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)
        self.p['b'] = np.random.rand(self.nn,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out_1 = self.p.get_val('sum_comp.sum_output1', units='km')
        out_2 = self.p.get_val('sum_comp.sum_output2', units='g')


        expected_1 = np.sum(a, axis=0) / 1000
        expected_1 = expected_1.reshape((1,self.length))
        expected_2 = np.sum(b, axis=0) * 1000
        expected_2 = expected_2.reshape((1,self.length))
        assert_near_equal(out_1, expected_1,1e-15)
        assert_near_equal(out_2, expected_2,1e-15)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestBadAxisRaisesError(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp(axis=2))
        multi.add_equation('sum_output','sum_input',vec_size=self.nn,length=self.length)

        self.p.model.connect('a', 'sum_comp.sum_input')

    def test_for_exception(self):
        self.assertRaises(ValueError,self.p.setup)

class TestSummationNx3OnInit(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        multi=self.p.model.add_subsystem(name='sum_comp',
                                   subsys=SumComp('sum_output','sum_input',vec_size=self.nn,length=self.length))

        self.p.model.connect('a', 'sum_comp.sum_input')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        out = self.p['sum_comp.sum_output']
        expected = np.sum(a, axis=0)
        expected = expected.reshape((1,self.length))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestForDocs(unittest.TestCase):

    def test(self):
        """
        A simple example to compute total fuel burn over an aircraft mission
        """
        import numpy as np
        #from openmdao.api import Problem, Group, IndepVarComp
        from openconcept.utilities.math.sum_comp import SumComp
        from openmdao.utils.assert_utils import assert_near_equal

        n = 10
        length = 1
        p = Problem(model=Group())

        ivc = IndepVarComp()
        #the vector represents fuel burns over several mission segments
        ivc.add_output(name='fuel_burn_by_seg', shape=(n,), units='kg')
        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['fuel_burn_by_seg'])

        # construct a summation component here
        # axis=0 sums along the vector
        total = SumComp(axis=0)
        total.add_equation('total_fuel','fuel_burn_by_seg',vec_size=n, units='kg')

        p.model.add_subsystem(name='totalfuelcomp', subsys=total, promotes_inputs=['*'])

        p.setup()

        # create a vector of fuel burns
        p['fuel_burn_by_seg'] = np.random.uniform(low=20,high=30,size=(n,))

        p.run_model()

        # print(p.get_val('totalforcecomp.total_force', units='kN'))

        # Verify the results
        expected_i = np.sum(p['fuel_burn_by_seg'],axis=0)
        expected_i = expected_i.reshape((1,))
        assert_near_equal(p.get_val('totalfuelcomp.total_fuel', units='kg'), expected_i)


if __name__ == '__main__':
    unittest.main()