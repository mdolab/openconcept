from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
#from openmdao.components.multiply_divide_comp import ElementMultiplyDivideComp
from openconcept.utilities.math.multiply_divide_comp import ElementMultiplyDivideComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

class TestElementMultiplyDivideCompScalars(unittest.TestCase):

    def setUp(self):
        self.nn = 1
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp())
        multi.add_equation('multdiv_output',['input_a','input_b'])

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['multiply_divide_comp.multdiv_output']
        expected = a * b
        assert_rel_error(self, out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestElementMultiplyDivideCompNx1(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp())
        multi.add_equation('multdiv_output',['input_a','input_b'],vec_size=self.nn)

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['multiply_divide_comp.multdiv_output']
        expected = a * b
        assert_rel_error(self, out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestElementMultiplyVectorScalar(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', val=3.0)

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp())
        multi.add_equation('multdiv_output',['input_a','input_b'],vec_size=[self.nn, 1])

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['multiply_divide_comp.multdiv_output']
        expected = a * b
        assert_rel_error(self, out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestElementMultiplyDivideCompNx3(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp())
        multi.add_equation('multdiv_output',['input_a','input_b'],vec_size=self.nn,length=3)

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')

        self.p.setup()

        self.p['a'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['b'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['multiply_divide_comp.multdiv_output']
        expected = a * b
        assert_rel_error(self, out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestElementMultiplyDivideMultipleInputs(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp())
        multi.add_equation('multdiv_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3)

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')
        self.p.model.connect('c', 'multiply_divide_comp.input_c')

        self.p.setup()

        self.p['a'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['b'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['c'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p['multiply_divide_comp.multdiv_output']
        expected = a * b * c
        assert_rel_error(self, out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestElementMultiplyDivideDivisionFirst(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp(complex=True))
        multi.add_equation('multdiv_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,divide=[True,True,False])

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')
        self.p.model.connect('c', 'multiply_divide_comp.input_c')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['b'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['c'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p['multiply_divide_comp.multdiv_output']
        expected = 1 / a / b * c
        assert_rel_error(self, out, expected,1e-15)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestElementMultiplyDivideScalingFactor(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp())
        multi.add_equation('multdiv_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,scaling_factor=2)

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')
        self.p.model.connect('c', 'multiply_divide_comp.input_c')

        self.p.setup()

        self.p['a'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['b'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['c'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p['multiply_divide_comp.multdiv_output']
        expected = 2 * a * b * c
        assert_rel_error(self, out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestElementMultiplyDivideUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='kg')
        ivc.add_output(name='b', shape=(self.nn, 3), units='m')
        ivc.add_output(name='c', shape=(self.nn, 3), units='s ** 2')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp(complex=True))
        multi.add_equation('multdiv_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,
                           input_units=['kg','m','s**2'], divide=[False, False, True])

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')
        self.p.model.connect('c', 'multiply_divide_comp.input_c')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['b'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        # use uniform 1 - 2 distribution to avoid div zero
        self.p['c'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p.get_val('multiply_divide_comp.multdiv_output', units='kN')
        expected = a * b / c / 1000
        assert_rel_error(self, out, expected,1e-8)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestElementMultiplyDivideUnits_DivideFirst(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='kg')
        ivc.add_output(name='b', shape=(self.nn, 3), units='m')
        ivc.add_output(name='c', shape=(self.nn, 3), units='s ** 2')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp(complex=True))
        multi.add_equation('multdiv_output',['input_c','input_b','input_a'],vec_size=self.nn,length=3,
                           input_units=['s**2','m','kg'], divide=[True, False, False])

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')
        self.p.model.connect('c', 'multiply_divide_comp.input_c')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        self.p['b'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))
        # use uniform 1 - 2 distribution to avoid div zero
        self.p['c'] = np.random.uniform(low=1,high=2,size=(self.nn, 3))

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        out = self.p.get_val('multiply_divide_comp.multdiv_output', units='kN')
        expected = a * b / c / 1000
        assert_rel_error(self, out, expected,1e-8)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestWrongUnitsCount(unittest.TestCase):

    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp())
        multi.add_equation('multdiv_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,input_units=['kg','ft'])

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')
        self.p.model.connect('c', 'multiply_divide_comp.input_c')


    def test_for_exception(self):
        self.assertRaises(ValueError,self.p.setup)

class TestWrongDivideCount(unittest.TestCase):

    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))
        ivc.add_output(name='c', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        multi=self.p.model.add_subsystem(name='multiply_divide_comp',
                                   subsys=ElementMultiplyDivideComp())
        multi.add_equation('multdiv_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,divide=[False,True])

        self.p.model.connect('a', 'multiply_divide_comp.input_a')
        self.p.model.connect('b', 'multiply_divide_comp.input_b')
        self.p.model.connect('c', 'multiply_divide_comp.input_c')


    def test_for_exception(self):
        self.assertRaises(ValueError,self.p.setup)

class TestForDocs(unittest.TestCase):

    def test(self):
        """
        A simple example to compute inertial forces on four projectiles at
            a number of analysis points (F_inertial = - m*a)
        """
        import numpy as np
        #from openmdao.api import Problem, Group, IndepVarComp
        from openconcept.utilities.math.multiply_divide_comp import ElementMultiplyDivideComp
        from openmdao.utils.assert_utils import assert_rel_error

        n = 5
        length = 4
        p = Problem(model=Group())

        ivc = IndepVarComp()
        #the vector represents forces at 3 analysis points (rows) in 2 dimensional plane (cols)
        ivc.add_output(name='mass', shape=(n,length), units='kg')
        ivc.add_output(name='acceleration', shape=(n,length), units='m / s**2')
        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['mass', 'acceleration'])

        #construct an multi/subtracter here. create a relationship through the add_equation method
        multi = ElementMultiplyDivideComp()
        multi.add_equation('inertial_force',input_names=['mass', 'acceleration'],vec_size=n,length=length, input_units=['kg','m / s**2'], scaling_factor=-1)
        #note the scaling factors. we assume all forces are positive sign upstream

        p.model.add_subsystem(name='inertialforcecomp', subsys=multi, promotes_inputs=['*'])

        p.setup()

        #set thrust to exceed drag, weight to equal lift for this scenario
        p['mass'] = np.ones((n,length)) * 500
        p['acceleration'] = np.random.rand(n,length)


        p.run_model()

        # print(p.get_val('totalforcecomp.total_force', units='kN'))

        # Verify the results
        expected_i = - p['mass'] * p['acceleration'] / 1000
        assert_rel_error(self, p.get_val('inertialforcecomp.inertial_force', units='kN'), expected_i)


if __name__ == '__main__':
    unittest.main()