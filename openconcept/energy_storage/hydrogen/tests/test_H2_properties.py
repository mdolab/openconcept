import numpy as np
import unittest
from parameterized import parameterized_class
from openmdao.utils.assert_utils import assert_near_equal
import pytest
from openconcept.energy_storage.hydrogen.H2_properties import *

# For some reason need to wrap function handle in list to properly call it
real_gas_funcs = [
    {"gas_func": [gh2_cv]},
    {"gas_func": [gh2_cp]},
    {"gas_func": [gh2_u]},
    {"gas_func": [gh2_h]},
]

sat_funcs = [
    {"sat_func": [lh2_P], "temp_input": True},
    {"sat_func": [lh2_h], "temp_input": True},
    {"sat_func": [lh2_u], "temp_input": True},
    {"sat_func": [lh2_cp], "temp_input": True},
    {"sat_func": [lh2_rho], "temp_input": True},
    {"sat_func": [sat_gh2_rho], "temp_input": True},
    {"sat_func": [sat_gh2_h], "temp_input": True},
    {"sat_func": [sat_gh2_cp], "temp_input": True},
    {"sat_func": [sat_gh2_k], "temp_input": True},
    {"sat_func": [sat_gh2_viscosity], "temp_input": True},
    {"sat_func": [sat_gh2_beta], "temp_input": True},
    {"sat_func": [sat_gh2_T], "temp_input": False},
]

@parameterized_class(real_gas_funcs)
class RealGasPropertyTestCase(unittest.TestCase):
    def test_scalars(self):
        func = self.gas_func[0]
        P = 1e5
        T = 20.1
        out = func(P, T)
        out_P, out_T = func(P, T, deriv=True)

        self.assertTrue(isinstance(out, float))
        self.assertTrue(isinstance(out_P, float))
        self.assertTrue(isinstance(out_T, float))

    def test_vectors(self):
        func = self.gas_func[0]
        n = 3
        P = np.linspace(1e4, 1e6, n)
        T = np.linspace(10, 30, n)
        out = func(P, T)
        out_P, out_T = func(P, T, deriv=True)

        self.assertEqual(out.shape, (n,))
        self.assertEqual(out_P.shape, (n,))
        self.assertEqual(out_T.shape, (n,))

    def test_mix(self):
        func = self.gas_func[0]
        n = 3
        P = 1e5
        T = np.linspace(10, 30, n)
        out = func(P, T)
        out_P, out_T = func(P, T, deriv=True)

        self.assertEqual(out.shape, (n,))
        self.assertEqual(out_P.shape, (n,))
        self.assertEqual(out_T.shape, (n,))

    def test_invalid_vector_shapes(self):
        func = self.gas_func[0]

        with self.assertRaises(ValueError):
            _ = func(np.zeros(3), np.zeros(4))

        with self.assertRaises(ValueError):
            _ = func(np.zeros(4), np.zeros(3))

    def test_derivatives(self):
        func = self.gas_func[0]
        P = np.linspace(1e4, 1e6, 3, dtype=complex)
        T = np.linspace(10, 30, 3, dtype=complex)
        out_P, out_T = func(P, T, deriv=True)

        step = 1e-200

        for i in range(P.size):
            P[i] += step * 1j
            out = func(P, T)
            P[i] -= step * 1j

            assert_near_equal(np.imag(out[i]) / step, np.real(out_P[i]), tolerance=1e-13)

        for i in range(T.size):
            T[i] += step * 1j
            out = func(P, T)
            T[i] -= step * 1j

            assert_near_equal(np.imag(out[i]) / step, np.real(out_T[i]), tolerance=1e-13)


@parameterized_class(sat_funcs)
class SaturatedPropertyTestCase(unittest.TestCase):
    def test_scalar(self):
        func = self.sat_func[0]
        if self.temp_input:
            input = 20.1
        else:
            input = 1e5
        out = func(input)
        out_deriv = func(input, deriv=True)

        self.assertTrue(isinstance(out, float))
        self.assertTrue(isinstance(out_deriv, float))

    def test_derivatives(self):
        func = self.sat_func[0]
        if self.temp_input:
            input = np.linspace(10, 30, 3, dtype=complex)
        else:
            input = np.linspace(1e4, 1e6, 3, dtype=complex)
        out_deriv = func(input, deriv=True)

        step = 1e-200

        for i in range(input.size):
            input[i] += step * 1j
            out = func(input)
            input[i] -= step * 1j

            assert_near_equal(np.imag(out[i]) / step, np.real(out_deriv[i]), tolerance=1e-13)


if __name__=="__main__":
    unittest.main()
