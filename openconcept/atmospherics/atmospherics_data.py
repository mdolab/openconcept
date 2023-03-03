"""
This module provides 1976 Standard Atmosphere constants and calculations.

Adapted from:
J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018
"""

import numpy as np

# tropopause
epsilon = 500
h_trans = 11000
h_lower = h_trans - epsilon
h_upper = h_trans + epsilon
tropopause_matrix = np.array(
    [
        [h_lower**3, h_lower**2, h_lower, 1],
        [h_upper**3, h_upper**2, h_upper, 1],
        [3 * h_lower**2, 2 * h_lower, 1, 0],
        [3 * h_upper**2, 2 * h_upper, 1, 0],
    ]
)

# pressure
tmp1 = 1 - 0.0065 * h_lower / 288.16
tmp2 = np.exp(-9.81 * epsilon / (288 * 216.65))
pressure_rhs = np.array(
    [
        101325 * tmp1**5.2561,
        22632 * tmp2,
        (-101325 * 5.2561 * (0.0065 / 288.16) * tmp1**4.2561),
        (22632 * (-9.81 / (288 * 216.65)) * tmp2),
    ]
)
pressure_coeffs = np.linalg.solve(tropopause_matrix, pressure_rhs)

# temperature
temp_rhs = np.array(
    [
        288.16 - (6.5e-3) * h_lower,
        216.65,
        -6.5e-3,
        0,
    ]
)
temp_coeffs = np.linalg.solve(tropopause_matrix, temp_rhs)


# functions
def get_mask_arrays(h_m):
    tropos_mask = h_m <= h_lower
    strato_mask = h_m > h_upper
    smooth_mask = np.logical_and(~tropos_mask, ~strato_mask)
    return tropos_mask, strato_mask, smooth_mask


def compute_pressures(h_m, tropos_mask, strato_mask, smooth_mask):
    a, b, c, d = pressure_coeffs

    p_Pa = np.zeros(len(h_m), dtype=type(h_m[0]))
    p_Pa += tropos_mask * (101325 * (1 - 0.0065 * h_m / 288.16) ** 5.2561)
    p_Pa += strato_mask * (22632 * np.exp(-9.81 * (h_m - h_trans) / (288 * 216.65)))
    p_Pa += smooth_mask * (a * h_m**3 + b * h_m**2 + c * h_m + d)

    return p_Pa


def compute_pressure_derivs(h_m, tropos_mask, strato_mask, smooth_mask):
    a, b, c, d = pressure_coeffs

    derivs = np.zeros(len(h_m), dtype=type(h_m[0]))
    derivs += tropos_mask * (101325 * 5.2561 * (-0.0065 / 288.16) * (1 - 0.0065 * h_m / 288.16) ** 4.2561)
    derivs += strato_mask * (
        22632 * (-9.81 / (288 * 216.65)) * np.exp(9.81 * 11000 / (288 * 216.65)) * np.exp(-9.81 * h_m / (288 * 216.65))
    )
    derivs += smooth_mask * (3 * a * h_m**2 + 2 * b * h_m + c)

    return derivs


def compute_temps(h_m, tropos_mask, strato_mask, smooth_mask):
    a, b, c, d = temp_coeffs

    temp_K = np.zeros(len(h_m), dtype=type(h_m[0]))
    temp_K += tropos_mask * (288.16 - (6.5e-3) * h_m)
    temp_K += strato_mask * 216.65
    temp_K += smooth_mask * (a * h_m**3 + b * h_m**2 + c * h_m + d)

    return temp_K


def compute_temp_derivs(h_m, tropos_mask, strato_mask, smooth_mask):
    a, b, c, d = temp_coeffs

    derivs = np.zeros(len(h_m), dtype=type(h_m[0]))
    derivs += tropos_mask * (-6.5e-3)
    derivs += smooth_mask * (3 * a * h_m**2 + 2 * b * h_m + c)

    return derivs
