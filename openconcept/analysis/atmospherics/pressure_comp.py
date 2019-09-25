from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent

from .atmospherics_data import get_mask_arrays, compute_pressures, compute_pressure_derivs


class PressureComp(ExplicitComponent):
    '''
    This component computes pressure from altitude.

    Adapted from:
    J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018
    '''


    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_points = self.options['num_nodes']

        self.add_input('h_km', shape=num_points)
        self.add_output('p_MPa', shape=num_points, lower=0.)

        arange = np.arange(num_points)
        self.declare_partials('p_MPa', 'h_km', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        num_points = self.options['num_nodes']

        h_m = inputs['h_km'] * 1e3
        self.tropos_mask, self.strato_mask, self.smooth_mask = get_mask_arrays(h_m)
        p_Pa = compute_pressures(h_m, self.tropos_mask, self.strato_mask, self.smooth_mask)

        outputs['p_MPa'] = p_Pa / 1e6

    def compute_partials(self, inputs, partials):
        num_points = self.options['num_nodes']

        h_m = inputs['h_km'] * 1e3

        derivs = compute_pressure_derivs(h_m, self.tropos_mask, self.strato_mask, self.smooth_mask)

        partials['p_MPa', 'h_km'] = derivs * 1e3 / 1e6
