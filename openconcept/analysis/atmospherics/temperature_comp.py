from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent

from .atmospherics_data import get_mask_arrays, compute_temps, compute_temp_derivs


class TemperatureComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_points = self.options['num_nodes']

        self.add_input('h_km', shape=num_points)
        self.add_output('T_1e2_K', shape=num_points, lower=0.)

        arange = np.arange(num_points)
        self.declare_partials('T_1e2_K', 'h_km', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        num_points = self.options['num_nodes']

        h_m = inputs['h_km'] * 1e3

        self.tropos_mask, self.strato_mask, self.smooth_mask = get_mask_arrays(h_m)
        temp_K = compute_temps(h_m, self.tropos_mask, self.strato_mask, self.smooth_mask)

        outputs['T_1e2_K'] = temp_K / 1e2

    def compute_partials(self, inputs, partials):
        num_points = self.options['num_nodes']

        h_m = inputs['h_km'] * 1e3

        derivs = compute_temp_derivs(h_m, self.tropos_mask, self.strato_mask, self.smooth_mask)

        partials['T_1e2_K', 'h_km'] = derivs * 1e3 / 1e2
