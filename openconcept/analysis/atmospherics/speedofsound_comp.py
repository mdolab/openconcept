from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent


gamma = 1.4
R = 287.058


class SpeedOfSoundComp(ExplicitComponent):
    '''
    This component computes speed of sound from temperature.

    Adapted from:
    J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018
    '''


    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_points = self.options['num_nodes']

        self.add_input('T_1e2_K', shape=num_points)
        self.add_output('a_1e2_ms', shape=num_points)

        arange = np.arange(num_points)
        self.declare_partials('a_1e2_ms', 'T_1e2_K', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        T_K = inputs['T_1e2_K'] * 1e2

        outputs['a_1e2_ms'] = np.sqrt(gamma * R * T_K) / 1e2

    def compute_partials(self, inputs, partials):
        T_K = inputs['T_1e2_K'] * 1e2

        data = 0.5 * np.sqrt(gamma * R / T_K)
        partials['a_1e2_ms', 'T_1e2_K'] = data
