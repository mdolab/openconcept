from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group
#from openmdao.components.add_subtract_comp import AddSubtractComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

# Test data to load
# aero data =============================
aero = dict()
## clmax data ===========================
clmax = dict()
clmax['flaps30'] = {'value': 1.7}
clmax['flaps10'] = {'value': 1.5}
aero['CLmax'] = clmax
## other aero data ======================
aero['myvector'] = {'value' : np.ones(10), 'units' : 'kg'}

geom = dict()
geom['S_ref'] = {'value' : 20, 'units': 'ft**2'}
geom['noval'] = {'units' : 'ft**2'}
data = dict()
data['aero'] = aero
data['geom'] = geom


class TestOne(unittest.TestCase):

    def setUp(self):
        self.nn = 3
        self.p = Problem(model=Group())
        ivc = DictIndepVarComp(data_dict=data)
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))
        ivc.add_output_from_dict('geom|S_ref')
        ivc.add_output_from_dict('aero|CLmax|flaps30')
        ivc.add_output_from_dict('aero|CLmax|flaps10')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['geom|S_ref']
        expected = 20
        assert_near_equal(out, expected,1e-16)

    def test_units(self):
        out = self.p.get_val('geom|S_ref', units='m**2')
        expected = 20 * 0.3048**2
        assert_near_equal(out, expected,1e-4)

    def test_twodeep(self):
        out = self.p.get_val('aero|CLmax|flaps30')
        expected = 1.7
        assert_near_equal(out, expected,1e-4)
        out = self.p.get_val('aero|CLmax|flaps10')
        expected = 1.5
        assert_near_equal(out, expected,1e-4)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestAddNonexistentVar(unittest.TestCase):
    def setUp(self):
        self.nn = 3
        self.p = Problem(model=Group())
        self.ivc = DictIndepVarComp(data_dict=data)
        self.ivc.add_output(name='a', shape=(self.nn,))
        self.ivc.add_output(name='b', shape=(self.nn,))

    def test_nonexistent_key(self):
        self.assertRaises(KeyError,self.ivc.add_output_from_dict,'geom|S_ref_blah')

    def test_no_value(self):
        self.assertRaises(KeyError,self.ivc.add_output_from_dict,'geom|noval')

# class TestAddSubtractCompNx1(unittest.TestCase):

#     def setUp(self):
#         self.nn = 5

#         self.p = Problem(model=Group())

#         ivc = IndepVarComp()
#         ivc.add_output(name='a', shape=(self.nn,))
#         ivc.add_output(name='b', shape=(self.nn,))

#         self.p.model.add_subsystem(name='ivc',
#                                    subsys=ivc,
#                                    promotes_outputs=['a', 'b'])

#         adder=self.p.model.add_subsystem(name='add_subtract_comp',
#                                    subsys=AddSubtractComp())
#         adder.add_equation('adder_output',['input_a','input_b'],vec_size=self.nn)

#         self.p.model.connect('a', 'add_subtract_comp.input_a')
#         self.p.model.connect('b', 'add_subtract_comp.input_b')

#         self.p.setup()

#         self.p['a'] = np.random.rand(self.nn,)
#         self.p['b'] = np.random.rand(self.nn,)

#         self.p.run_model()

#     def test_results(self):
#         a = self.p['a']
#         b = self.p['b']
#         out = self.p['add_subtract_comp.adder_output']
#         expected = a + b
#         assert_near_equal(out, expected,1e-16)

#     def test_partials(self):
#         partials = self.p.check_partials(method='fd', out_stream=None)
#         assert_check_partials(partials)


# class TestAddSubtractCompNx3(unittest.TestCase):

#     def setUp(self):
#         self.nn = 5

#         self.p = Problem(model=Group())

#         ivc = IndepVarComp()
#         ivc.add_output(name='a', shape=(self.nn, 3))
#         ivc.add_output(name='b', shape=(self.nn, 3))

#         self.p.model.add_subsystem(name='ivc',
#                                    subsys=ivc,
#                                    promotes_outputs=['a', 'b'])

#         adder=self.p.model.add_subsystem(name='add_subtract_comp',
#                                    subsys=AddSubtractComp())
#         adder.add_equation('adder_output',['input_a','input_b'],vec_size=self.nn,length=3)

#         self.p.model.connect('a', 'add_subtract_comp.input_a')
#         self.p.model.connect('b', 'add_subtract_comp.input_b')

#         self.p.setup()

#         self.p['a'] = np.random.rand(self.nn, 3)
#         self.p['b'] = np.random.rand(self.nn, 3)

#         self.p.run_model()

#     def test_results(self):
#         a = self.p['a']
#         b = self.p['b']
#         out = self.p['add_subtract_comp.adder_output']
#         expected = a + b
#         assert_near_equal(out, expected,1e-16)

#     def test_partials(self):
#         partials = self.p.check_partials(method='fd', out_stream=None)
#         assert_check_partials(partials)

# class TestAddSubtractMultipleInputs(unittest.TestCase):

#     def setUp(self):
#         self.nn = 5

#         self.p = Problem(model=Group())

#         ivc = IndepVarComp()
#         ivc.add_output(name='a', shape=(self.nn, 3))
#         ivc.add_output(name='b', shape=(self.nn, 3))
#         ivc.add_output(name='c', shape=(self.nn, 3))

#         self.p.model.add_subsystem(name='ivc',
#                                    subsys=ivc,
#                                    promotes_outputs=['a', 'b','c'])

#         adder=self.p.model.add_subsystem(name='add_subtract_comp',
#                                    subsys=AddSubtractComp())
#         adder.add_equation('adder_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3)

#         self.p.model.connect('a', 'add_subtract_comp.input_a')
#         self.p.model.connect('b', 'add_subtract_comp.input_b')
#         self.p.model.connect('c', 'add_subtract_comp.input_c')

#         self.p.setup()

#         self.p['a'] = np.random.rand(self.nn, 3)
#         self.p['b'] = np.random.rand(self.nn, 3)
#         self.p['c'] = np.random.rand(self.nn, 3)

#         self.p.run_model()

#     def test_results(self):
#         a = self.p['a']
#         b = self.p['b']
#         c = self.p['c']
#         out = self.p['add_subtract_comp.adder_output']
#         expected = a + b + c
#         assert_near_equal(out, expected,1e-16)

#     def test_partials(self):
#         partials = self.p.check_partials(method='fd', out_stream=None)
#         assert_check_partials(partials)

# class TestAddSubtractScalingFactors(unittest.TestCase):

#     def setUp(self):
#         self.nn = 5

#         self.p = Problem(model=Group())

#         ivc = IndepVarComp()
#         ivc.add_output(name='a', shape=(self.nn, 3))
#         ivc.add_output(name='b', shape=(self.nn, 3))
#         ivc.add_output(name='c', shape=(self.nn, 3))

#         self.p.model.add_subsystem(name='ivc',
#                                    subsys=ivc,
#                                    promotes_outputs=['a', 'b','c'])

#         adder=self.p.model.add_subsystem(name='add_subtract_comp',
#                                    subsys=AddSubtractComp())
#         adder.add_equation('adder_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,scaling_factors=[2.,1.,-1])

#         self.p.model.connect('a', 'add_subtract_comp.input_a')
#         self.p.model.connect('b', 'add_subtract_comp.input_b')
#         self.p.model.connect('c', 'add_subtract_comp.input_c')

#         self.p.setup()

#         self.p['a'] = np.random.rand(self.nn, 3)
#         self.p['b'] = np.random.rand(self.nn, 3)
#         self.p['c'] = np.random.rand(self.nn, 3)

#         self.p.run_model()

#     def test_results(self):
#         a = self.p['a']
#         b = self.p['b']
#         c = self.p['c']
#         out = self.p['add_subtract_comp.adder_output']
#         expected = 2*a + b - c
#         assert_near_equal(out, expected,1e-16)

#     def test_partials(self):
#         partials = self.p.check_partials(method='fd', out_stream=None)
#         assert_check_partials(partials)

# class TestAddSubtractUnits(unittest.TestCase):

#     def setUp(self):
#         self.nn = 5

#         self.p = Problem(model=Group())

#         ivc = IndepVarComp()
#         ivc.add_output(name='a', shape=(self.nn, 3), units='ft')
#         ivc.add_output(name='b', shape=(self.nn, 3), units='m')
#         ivc.add_output(name='c', shape=(self.nn, 3), units='m')

#         self.p.model.add_subsystem(name='ivc',
#                                    subsys=ivc,
#                                    promotes_outputs=['a', 'b','c'])

#         adder=self.p.model.add_subsystem(name='add_subtract_comp',
#                                    subsys=AddSubtractComp())
#         adder.add_equation('adder_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3, units='ft')

#         self.p.model.connect('a', 'add_subtract_comp.input_a')
#         self.p.model.connect('b', 'add_subtract_comp.input_b')
#         self.p.model.connect('c', 'add_subtract_comp.input_c')

#         self.p.setup()

#         self.p['a'] = np.random.rand(self.nn, 3)
#         self.p['b'] = np.random.rand(self.nn, 3)
#         self.p['c'] = np.random.rand(self.nn, 3)

#         self.p.run_model()

#     def test_results(self):
#         a = self.p['a']
#         b = self.p['b']
#         c = self.p['c']
#         out = self.p['add_subtract_comp.adder_output']
#         m_to_ft = 3.280839895
#         expected = a + b*m_to_ft + c*m_to_ft
#         assert_near_equal(out, expected,1e-8)

#     def test_partials(self):
#         partials = self.p.check_partials(method='fd', out_stream=None)
#         assert_check_partials(partials)

# class TestWrongScalingFactorCount(unittest.TestCase):

#     def setUp(self):
#         self.nn = 5
#         self.p = Problem(model=Group())

#         ivc = IndepVarComp()
#         ivc.add_output(name='a', shape=(self.nn, 3))
#         ivc.add_output(name='b', shape=(self.nn, 3))
#         ivc.add_output(name='c', shape=(self.nn, 3))

#         self.p.model.add_subsystem(name='ivc',
#                                    subsys=ivc,
#                                    promotes_outputs=['a', 'b','c'])

#         adder=self.p.model.add_subsystem(name='add_subtract_comp',
#                                    subsys=AddSubtractComp())
#         adder.add_equation('adder_output',['input_a','input_b','input_c'],vec_size=self.nn,length=3,scaling_factors=[1,-1])

#         self.p.model.connect('a', 'add_subtract_comp.input_a')
#         self.p.model.connect('b', 'add_subtract_comp.input_b')
#         self.p.model.connect('c', 'add_subtract_comp.input_c')


#     def test_for_exception(self):
#         self.assertRaises(ValueError,self.p.setup)

# class TestForDocs(unittest.TestCase):

#     def test(self):
#         """
#         A simple example to compute the resultant force on an aircraft and demonstrate the AddSubtract component
#         """
#         import numpy as np
#         #from openmdao.api import Problem, Group, IndepVarComp
#         from openconcept.utilities.math.add_subtract_comp import AddSubtractComp
#         from openmdao.utils.assert_utils import assert_near_equal

#         n = 3

#         p = Problem(model=Group())

#         ivc = IndepVarComp()
#         #the vector represents forces at 3 analysis points (rows) in 2 dimensional plane (cols)
#         ivc.add_output(name='thrust', shape=(n,2), units='kN')
#         ivc.add_output(name='drag', shape=(n,2), units='kN')
#         ivc.add_output(name='lift', shape=(n,2), units='kN')
#         ivc.add_output(name='weight', shape=(n,2), units='kN')
#         p.model.add_subsystem(name='ivc',
#                               subsys=ivc,
#                               promotes_outputs=['thrust', 'drag', 'lift', 'weight'])

#         #construct an adder/subtracter here. create a relationship through the add_equation method
#         adder = AddSubtractComp()
#         adder.add_equation('total_force',input_names=['thrust','drag','lift','weight'],vec_size=n,length=2, scaling_factors=[1,-1,1,-1], units='kN')
#         #note the scaling factors. we assume all forces are positive sign upstream

#         p.model.add_subsystem(name='totalforcecomp', subsys=adder)

#         p.model.connect('thrust', 'totalforcecomp.thrust')
#         p.model.connect('drag', 'totalforcecomp.drag')
#         p.model.connect('lift', 'totalforcecomp.lift')
#         p.model.connect('weight', 'totalforcecomp.weight')

#         p.setup()

#         #set thrust to exceed drag, weight to equal lift for this scenario
#         p['thrust'][:,0] = [500, 600, 700]
#         p['drag'][:,0] = [400, 400, 400]
#         p['weight'][:,1] = [1000, 1001, 1002]
#         p['lift'][:,1] = [1000, 1000, 1000]

#         p.run_model()

#         # print(p.get_val('totalforcecomp.total_force', units='kN'))

#         # Verify the results
#         expected_i = np.array([[100, 200, 300], [0, -1, -2]]).T
#         assert_near_equal(p.get_val('totalforcecomp.total_force', units='kN'), expected_i)


if __name__ == '__main__':
    unittest.main()