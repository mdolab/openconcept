from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openconcept.utilities.math.combine_split_comp import VectorConcatenateComp, VectorSplitComp
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

class TestConcatenateScalars(unittest.TestCase):

    def setUp(self):
        self.nn = 1
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp())
        combiner.add_relation('concat_output',['input_a','input_b'],vec_sizes=[1,1])

        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['vector_concat_comp.concat_output']
        expected = np.concatenate((a,b))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestConcatenateNx1(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp())
        combiner.add_relation('concat_output',['input_a','input_b'],vec_sizes=[self.nn,self.nn])

        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['vector_concat_comp.concat_output']
        expected = np.concatenate((a,b))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestConcatenateNx3(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))
        ivc.add_output(name='b', shape=(self.nn,self.length))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp())
        combiner.add_relation('concat_output',['input_a','input_b'],vec_sizes=[self.nn,self.nn],length=self.length)

        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')

        self.p.setup(force_alloc_complex=False)

        self.p['a'] = np.random.rand(self.nn,self.length)
        self.p['b'] = np.random.rand(self.nn,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['vector_concat_comp.concat_output']
        expected = np.concatenate((a,b))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='fd', out_stream=None)
        assert_check_partials(partials)

class TestConcatenateInitMethod(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))
        ivc.add_output(name='b', shape=(self.nn,self.length))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp('concat_output',['input_a','input_b'],vec_sizes=[self.nn,self.nn],length=self.length))
        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)
        self.p['b'] = np.random.rand(self.nn,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        out = self.p['vector_concat_comp.concat_output']
        expected = np.concatenate((a,b))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestConcatenateMultipleSystems(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))
        ivc.add_output(name='b', shape=(self.nn,self.length))
        ivc.add_output(name='c', shape=(3,))
        ivc.add_output(name='d', shape=(4,))


        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c','d'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp())
        combiner.add_relation('concat_output1',['input_a','input_b'],vec_sizes=[self.nn,self.nn],length=self.length)
        combiner.add_relation('concat_output2',['input_c','input_d'],vec_sizes=[3,4])

        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')
        self.p.model.connect('c', 'vector_concat_comp.input_c')
        self.p.model.connect('d', 'vector_concat_comp.input_d')


        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)
        self.p['b'] = np.random.rand(self.nn,self.length)
        self.p['c'] = np.random.rand(3,)
        self.p['d'] = np.random.rand(4,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']
        d = self.p['d']

        out1 = self.p['vector_concat_comp.concat_output1']
        out2 = self.p['vector_concat_comp.concat_output2']

        expected1 = np.concatenate((a,b))
        expected2 = np.concatenate((c,d))
        assert_near_equal(out1, expected1,1e-16)
        assert_near_equal(out2, expected2,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestConcatenateNx3Units(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length), units='m')
        ivc.add_output(name='b', shape=(self.nn,self.length), units='km')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp())
        combiner.add_relation('concat_output',['input_a','input_b'],vec_sizes=[self.nn,self.nn],length=self.length, units='m')

        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)
        self.p['b'] = np.random.rand(self.nn,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        b = b*1000.
        out = self.p['vector_concat_comp.concat_output']
        expected = np.concatenate((a,b))
        assert_near_equal(out, expected,1e-16)

class TestConcatenate3InputsDiffSizesNx1(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))
        ivc.add_output(name='c', shape=(3,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp())
        combiner.add_relation('concat_output',['input_a','input_b','input_c'],vec_sizes=[self.nn,self.nn,3])

        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')
        self.p.model.connect('c', 'vector_concat_comp.input_c')


        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,)
        self.p['b'] = np.random.rand(self.nn,)
        self.p['c'] = np.random.rand(3,)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']

        out = self.p['vector_concat_comp.concat_output']
        expected = np.concatenate((a,b,c))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestConcatenate3InputsDiffSizesNx3(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.length = 3
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,self.length))
        ivc.add_output(name='b', shape=(self.nn,self.length))
        ivc.add_output(name='c', shape=(3,self.length))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp())
        combiner.add_relation('concat_output',['input_a','input_b','input_c'],vec_sizes=[self.nn,self.nn,3],length=self.length)

        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')
        self.p.model.connect('c', 'vector_concat_comp.input_c')


        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn,self.length)
        self.p['b'] = np.random.rand(self.nn,self.length)
        self.p['c'] = np.random.rand(3,self.length)

        self.p.run_model()

    def test_results(self):
        a = self.p['a']
        b = self.p['b']
        c = self.p['c']

        out = self.p['vector_concat_comp.concat_output']
        expected = np.concatenate((a,b,c))
        assert_near_equal(out, expected,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestConcatenateWrongVecSizesInputMismatch(unittest.TestCase):
    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))
        ivc.add_output(name='c', shape=(3,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b','c'])

        combiner=self.p.model.add_subsystem(name='vector_concat_comp',
                                   subsys=VectorConcatenateComp())
        combiner.add_relation('concat_output',['input_a','input_b','input_c'],vec_sizes=[self.nn,self.nn])

        self.p.model.connect('a', 'vector_concat_comp.input_a')
        self.p.model.connect('b', 'vector_concat_comp.input_b')
        self.p.model.connect('c', 'vector_concat_comp.input_c')

    def test_for_exception(self):
        self.assertRaises(ValueError,self.p.setup)

class TestSplitScalars(unittest.TestCase):

    def setUp(self):
        self.nn = 1
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='input_to_split', shape=(self.nn*2,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['input_to_split'])

        splitter=self.p.model.add_subsystem(name='vector_split_comp',
                                   subsys=VectorSplitComp())
        splitter.add_relation(['output_a','output_b'],'input_to_split',vec_sizes=[1,1])

        self.p.model.connect('input_to_split', 'vector_split_comp.input_to_split')
        self.p.setup(force_alloc_complex=True)
        self.p['input_to_split'] = np.random.rand(self.nn*2,)
        self.p.run_model()

    def test_results(self):
        input_to_split = self.p['input_to_split']
        out_a = self.p['vector_split_comp.output_a']
        out_b = self.p['vector_split_comp.output_b']

        expected_a = input_to_split[0]
        expected_b = input_to_split[1]
        assert_near_equal(out_a, expected_a,1e-16)
        assert_near_equal(out_b, expected_b,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSplitNx1(unittest.TestCase):

    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='input_to_split', shape=(self.nn*2,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['input_to_split'])

        splitter=self.p.model.add_subsystem(name='vector_split_comp',
                                   subsys=VectorSplitComp())
        splitter.add_relation(['output_a','output_b'],'input_to_split',vec_sizes=[self.nn,self.nn])

        self.p.model.connect('input_to_split', 'vector_split_comp.input_to_split')
        self.p.setup(force_alloc_complex=True)
        self.p['input_to_split'] = np.random.rand(self.nn*2,)
        self.p.run_model()

    def test_results(self):
        input_to_split = self.p['input_to_split']
        out_a = self.p['vector_split_comp.output_a']
        out_b = self.p['vector_split_comp.output_b']

        expected_a = input_to_split[0:self.nn]
        expected_b = input_to_split[self.nn:2*self.nn]
        assert_near_equal(out_a, expected_a,1e-16)
        assert_near_equal(out_b, expected_b,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSplitNx3(unittest.TestCase):

    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='input_to_split', shape=(self.nn*2,3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['input_to_split'])

        splitter=self.p.model.add_subsystem(name='vector_split_comp',
                                   subsys=VectorSplitComp())
        splitter.add_relation(['output_a','output_b'],'input_to_split',vec_sizes=[self.nn,self.nn],length=3)

        self.p.model.connect('input_to_split', 'vector_split_comp.input_to_split')
        self.p.setup(force_alloc_complex=True)
        self.p['input_to_split'] = np.random.rand(self.nn*2,3)
        self.p.run_model()

    def test_results(self):
        input_to_split = self.p['input_to_split']
        out_a = self.p['vector_split_comp.output_a']
        out_b = self.p['vector_split_comp.output_b']

        expected_a = input_to_split[0:self.nn,:]
        expected_b = input_to_split[self.nn:2*self.nn,:]
        assert_near_equal(out_a, expected_a,1e-16)
        assert_near_equal(out_b, expected_b,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSplitInitMethod(unittest.TestCase):

    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='input_to_split', shape=(self.nn*2,3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['input_to_split'])

        splitter=self.p.model.add_subsystem(name='vector_split_comp',
                                   subsys=VectorSplitComp(['output_a','output_b'],'input_to_split',vec_sizes=[self.nn,self.nn],length=3))
        self.p.model.connect('input_to_split', 'vector_split_comp.input_to_split')
        self.p.setup(force_alloc_complex=True)
        self.p['input_to_split'] = np.random.rand(self.nn*2,3)
        self.p.run_model()

    def test_results(self):
        input_to_split = self.p['input_to_split']
        out_a = self.p['vector_split_comp.output_a']
        out_b = self.p['vector_split_comp.output_b']

        expected_a = input_to_split[0:self.nn,:]
        expected_b = input_to_split[self.nn:2*self.nn,:]
        assert_near_equal(out_a, expected_a,1e-16)
        assert_near_equal(out_b, expected_b,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSplitMultipleSystems(unittest.TestCase):

    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='input_to_split', shape=(self.nn*2+2,3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['input_to_split'])

        splitter=self.p.model.add_subsystem(name='vector_split_comp',
                                   subsys=VectorSplitComp())
        splitter.add_relation(['output_a','output_b','output_c'],'input_to_split',vec_sizes=[self.nn,self.nn,2],length=3)

        self.p.model.connect('input_to_split', 'vector_split_comp.input_to_split')
        self.p.setup(force_alloc_complex=True)
        self.p['input_to_split'] = np.random.rand(self.nn*2+2,3)
        self.p.run_model()

    def test_results(self):
        input_to_split = self.p['input_to_split']
        out_a = self.p['vector_split_comp.output_a']
        out_b = self.p['vector_split_comp.output_b']
        out_c = self.p['vector_split_comp.output_c']

        expected_a = input_to_split[0:self.nn,:]
        expected_b = input_to_split[self.nn:2*self.nn,:]
        expected_c = input_to_split[2*self.nn:2*self.nn+2,:]

        assert_near_equal(out_a, expected_a,1e-16)
        assert_near_equal(out_b, expected_b,1e-16)
        assert_near_equal(out_c, expected_c,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSplitNx3Units(unittest.TestCase):

    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='input_to_split', shape=(self.nn*2,3), units='m')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['input_to_split'])

        splitter=self.p.model.add_subsystem(name='vector_split_comp',
                                   subsys=VectorSplitComp())
        splitter.add_relation(['output_a','output_b'],'input_to_split',vec_sizes=[self.nn,self.nn],length=3, units='m')

        self.p.model.connect('input_to_split', 'vector_split_comp.input_to_split')
        self.p.setup(force_alloc_complex=True)
        self.p['input_to_split'] = np.random.rand(self.nn*2,3)
        self.p.run_model()

    def test_results(self):
        input_to_split = self.p['input_to_split']
        out_a = self.p['vector_split_comp.output_a']
        out_b = self.p['vector_split_comp.output_b']

        expected_a = input_to_split[0:self.nn,:]
        expected_b = input_to_split[self.nn:2*self.nn,:]
        assert_near_equal(out_a, expected_a,1e-16)
        assert_near_equal(out_b, expected_b,1e-16)

    def test_partials(self):
        partials = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials)

class TestSplitWrongVecSizesOutputMismatch(unittest.TestCase):

    def setUp(self):
        self.nn = 5
        self.p = Problem(model=Group())
        ivc = IndepVarComp()
        ivc.add_output(name='input_to_split', shape=(self.nn*2,3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['input_to_split'])

        splitter=self.p.model.add_subsystem(name='vector_split_comp',
                                   subsys=VectorSplitComp())
        splitter.add_relation(['output_a','output_b'],'input_to_split',vec_sizes=[self.nn],length=3)

        self.p.model.connect('input_to_split', 'vector_split_comp.input_to_split')

    def test_for_exception(self):
        self.assertRaises(ValueError,self.p.setup)

class TestForDocs(unittest.TestCase):

    def test(self):
        """
        A simple example to illustrate usage of the concatenate/split components.

        Let's say we have two analysis tools which generate lists of test conditions to analyze.
        For this example, our two tools takeoff analysis and cruise analysis for an airplane.
        We'll generate altitudes and velocity vectors from two (fake) components, then combine them into one
        so we can feed a single vector to some other analysis tool.

        Once the tool is done, we'll split the results back out into takeoff and cruise phases.
        """
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.utils.assert_utils import assert_near_equal
        from openconcept.utilities.math.combine_split_comp import VectorConcatenateComp, VectorSplitComp

        n_takeoff_pts = 3
        n_cruise_pts = 5

        p = Problem(model=Group())

        takeoff_conditions = IndepVarComp()
        takeoff_conditions.add_output(name='velocity', shape=(n_takeoff_pts,2), units='m/s')
        takeoff_conditions.add_output(name='altitude', shape=(n_takeoff_pts,), units='m')

        cruise_conditions = IndepVarComp()
        cruise_conditions.add_output(name='velocity', shape=(n_cruise_pts,2), units='m/s')
        cruise_conditions.add_output(name='altitude', shape=(n_cruise_pts,), units='m')

        p.model.add_subsystem(name='takeoff_conditions',
                              subsys=takeoff_conditions)
        p.model.add_subsystem(name='cruise_conditions',
                              subsys=cruise_conditions)

        combiner=p.model.add_subsystem(name='combiner',
                                            subsys=VectorConcatenateComp())

        combiner.add_relation('velocity',['to_vel','cruise_vel'],
                              vec_sizes=[3,5],length=2, units='m/s')

        combiner.add_relation('altitude',['to_alt','cruise_alt'],
                              vec_sizes=[3,5], units='m')

        p.model.connect('takeoff_conditions.velocity', 'combiner.to_vel')
        p.model.connect('cruise_conditions.velocity', 'combiner.cruise_vel')
        p.model.connect('takeoff_conditions.altitude', 'combiner.to_alt')
        p.model.connect('cruise_conditions.altitude', 'combiner.cruise_alt')

        divider=p.model.add_subsystem(name='divider',
                                            subsys=VectorSplitComp())
        divider.add_relation(['to_vel','cruise_vel'],'velocity',
                              vec_sizes=[3,5],length=2, units='m/s')
        p.model.connect('combiner.velocity','divider.velocity')

        p.setup(force_alloc_complex=True)

        #set thrust to exceed drag, weight to equal lift for this scenario
        p['takeoff_conditions.velocity'][:,0] = [30, 40, 50]
        p['takeoff_conditions.velocity'][:,1] = [0, 0, 0]
        p['cruise_conditions.velocity'][:,0] = [60, 60, 60, 60, 60]
        p['cruise_conditions.velocity'][:,1] = [5, 0, 0, 0, -5]
        p['takeoff_conditions.altitude'][:] = [0, 0, 0]
        p['cruise_conditions.altitude'][:] = [6000,7500,8000,8500,5000]

        p.run_model()

        # Verify the results
        expected_vel = np.array([[30, 0], [40, 0], [50, 0], [60, 5], [60, 0], [60, 0], [60, 0], [60, -5]])
        expected_alt = np.array([0, 0, 0, 6000, 7500, 8000, 8500, 5000])
        expected_split_vel = np.array([[60, 5], [60, 0], [60, 0], [60, 0], [60, -5]])
        assert_near_equal(p.get_val('combiner.velocity', units='m/s'), expected_vel)
        assert_near_equal(p.get_val('combiner.altitude', units='m'), expected_alt)
        assert_near_equal(p.get_val('divider.cruise_vel', units='m/s'), expected_split_vel)

if __name__ == '__main__':
    unittest.main()