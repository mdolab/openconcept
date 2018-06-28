from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent


R = 287.058


class DensityComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_points = self.options['num_nodes']

        self.add_input('p_MPa', shape=num_points)
        self.add_input('T_1e2_K', shape=num_points)
        self.add_output('rho_kg_m3', shape=num_points)

        arange = np.arange(num_points)
        self.declare_partials('rho_kg_m3', 'p_MPa', rows=arange, cols=arange)
        self.declare_partials('rho_kg_m3', 'T_1e2_K', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        p_Pa = inputs['p_MPa'] * 1e6
        T_K = inputs['T_1e2_K'] * 1e2

        outputs['rho_kg_m3'] = p_Pa / R / T_K

    def compute_partials(self, inputs, partials):
        p_Pa = inputs['p_MPa'] * 1e6
        T_K = inputs['T_1e2_K'] * 1e2

        data = 1.0 / R / T_K
        partials['rho_kg_m3', 'p_MPa'] = data * 1e6

        data = -p_Pa / R / T_K ** 2
        partials['rho_kg_m3', 'T_1e2_K'] = data * 1e2
