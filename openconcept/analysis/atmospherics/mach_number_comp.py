from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent


v_cl = 300 * 0.514444
qc = 0.5 * 1.225 * v_cl ** 2

r = -20.

class MachNumberComp(ExplicitComponent):
    '''
    This component computes Mach number from stagnation Mach number, density, speed of sound, and pressure.

    Adapted from:
    J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018
    '''


    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mode', 'TAS', values=['TAS', 'EAS', 'IAS', 'constant'])

    def setup(self):
        num_points = self.options['num_nodes']

        self.add_input('M0', val=[0.8])
        self.add_input('rho_kg_m3', shape=num_points)
        self.add_input('a_1e2_ms', shape=num_points)
        self.add_input('p_MPa', shape=num_points)
        self.add_output('M', shape=num_points)

        self.M = np.zeros(num_points)
        self.dM_dM0 = np.zeros(num_points)
        self.dM_dp = np.zeros(num_points)
        self.dM_drho = np.zeros(num_points)
        self.dM_da = np.zeros(num_points)

        arange = np.arange(num_points)
        self.arange = arange
        self.declare_partials('M', 'M0', dependent=True)
        self.declare_partials('M', 'rho_kg_m3', rows=arange, cols=arange)
        self.declare_partials('M', 'a_1e2_ms', rows=arange, cols=arange)
        self.declare_partials('M', 'p_MPa', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        num_points = self.options['num_nodes']
        mode = self.options['mode']

        M0 = inputs['M0']
        rho = inputs['rho_kg_m3']
        a_ms = inputs['a_1e2_ms'] * 1e2
        p_Pa = inputs['p_MPa'] * 1e6

        if mode == 'TAS':
            Mcl = np.sqrt(5) * np.sqrt((qc / p_Pa + 1) ** (2. / 7.) - 1)
        elif mode == 'EAS':
            Mcl = v_cl * np.sqrt(1.225 / rho) / a_ms
        elif mode == 'IAS':
            Mcl = v_cl / a_ms
        elif mode == 'constant':
            Mcl = M0 * np.ones(num_points)
        else:
            raise Exception('Unrecognized mode option in Mach component')

        M_max = np.minimum(Mcl, M0)

        outputs['M'] = M_max + 1. / r \
            * np.log(np.exp(r * (Mcl - M_max)) + np.exp(r * (M0 - M_max)))

    def compute_partials(self, inputs, partials):
        num_points = self.options['num_nodes']
        mode = self.options['mode']

        M0 = inputs['M0']
        rho = inputs['rho_kg_m3']
        a_ms = inputs['a_1e2_ms'] * 1e2
        p_Pa = inputs['p_MPa'] * 1e6

        dM_dM0 = self.dM_dM0
        dM_dp = self.dM_dp
        dM_drho = self.dM_drho
        dM_da = self.dM_da

        if mode == 'TAS':
            Mcl = np.sqrt(5) * np.sqrt((qc / p_Pa + 1) ** (2. / 7.) - 1)
            dMcl_dp = 0.5 * np.sqrt(5) / np.sqrt((qc/p_Pa+1)**(2./7.) - 1) \
                * 2./7. * (qc/p_Pa+1)**(-5./7.) * (-qc/p_Pa**2) * 1e6
            dMcl_drho = np.zeros(num_points)
            dMcl_da = np.zeros(num_points)
            dMcl_dM0 = np.zeros(num_points)
        elif mode == 'EAS':
            Mcl = v_cl * np.sqrt(1.225 / rho) / a_ms
            dMcl_dp = np.zeros(num_points)
            dMcl_drho = -0.5 * v_cl * np.sqrt(1.225 / rho**3) / a_ms
            dMcl_da = -v_cl * np.sqrt(1.225 / rho) / a_ms**2 * 1e2
            dMcl_dM0 = np.zeros(num_points)
        elif mode == 'IAS':
            Mcl = v_cl / a_ms
            dMcl_dp = np.zeros(num_points)
            dMcl_drho = np.zeros(num_points)
            dMcl_da = -v_cl / a_ms**2 * 1e2
            dMcl_dM0 = np.zeros(num_points)
        elif mode == 'constant':
            Mcl = M0 * np.ones(num_points)
            dMcl_dp = np.zeros(num_points)
            dMcl_drho = np.zeros(num_points)
            dMcl_da = np.zeros(num_points)
            dMcl_dM0 = np.ones(num_points)

        M_max = np.minimum(Mcl, M0)

        partials['M', 'M0'] = (1. / r \
            / (np.exp(r * (Mcl - M_max)) + np.exp(r * (M0 - M_max))) \
            * (np.exp(r * (Mcl - M_max)) * r * dMcl_dM0 + np.exp(r * (M0 - M_max)) * r)
            ).reshape((num_points, 1))
        partials['M', 'p_MPa'] = 1. / r \
            / (np.exp(r * (Mcl - M_max)) + np.exp(r * (M0 - M_max))) \
            * (np.exp(r * (Mcl - M_max)) * r * dMcl_dp)
        partials['M', 'rho_kg_m3'] = 1. / r \
            / (np.exp(r * (Mcl - M_max)) + np.exp(r * (M0 - M_max))) \
            * (np.exp(r * (Mcl - M_max)) * r * dMcl_drho)
        partials['M', 'a_1e2_ms'] = 1. / r \
            / (np.exp(r * (Mcl - M_max)) + np.exp(r * (M0 - M_max))) \
            * (np.exp(r * (Mcl - M_max)) * r * dMcl_da)
