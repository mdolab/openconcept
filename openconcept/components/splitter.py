from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class PowerSplit(ExplicitComponent):
    """
    A power split mechanism for mechanical or electrical power.

    Inputs
    ------
    power_in : float
        Power fed to the splitter. (vector, W)
    power_rating : float
        Maximum rated power of the split mechanism. (scalar, W)
    power_split_fraction:
        If ``'rule'`` is set to ``'fraction'``, sets percentage of input power directed
        to Output A (minus losses). (vector, dimensionless)
    power_split_amount:
        If ``'rule'`` is set to ``'fixed'``, sets amount of input power to Output A (minus
        losses). (vector, W)

    Outputs
    -------
    power_out_A : float
        Power sent to first output (vector, W)
    power_out_B : float
        Power sent to second output (vector, W)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_weight : float
        Weight of the component (scalar, kg)
    component_sizing_margin : float
        Equal to 1 when fed full rated power (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    rule : str
        Power split control rule to use; either ``'fixed'`` where a set
        amount of power is sent to Output A or ``'fraction'`` where a
        fraction of the total power is sent to Output A
    efficiency : float
        Component efficiency (default 1)
    weight_inc : float
        Weight per unit rated power
        (default 0, kg/W)
    weight_base : float
        Base weight
        (default 0, kg)
    cost_inc : float
        Nonrecurring cost per unit power
        (default 0, USD/W)
    cost_base : float
        Base cost
        (default 0 USD)
    """
    def initialize(self):
        # define control rules
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('rule', default='fraction',
                             desc='Control strategy - fraction or fixed power')

        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=0., desc='kg per input watt')
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=0., desc='$ cost per input watt')
        self.options.declare('cost_base', default=0., desc='$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('power_in', units='W',
                       desc='Input shaft power or incoming electrical load', shape=(nn,))
        self.add_input('power_rating', val=99999999, units='W', desc='Split mechanism power rating')

        rule = self.options['rule']
        if rule == 'fraction':
            self.add_input('power_split_fraction', val=0.5,
                           desc='Fraction of power to output A', shape=(nn,))
        elif rule == 'fixed':
            self.add_input('power_split_amount', units='W',
                           desc='Raw amount of power to output A', shape=(nn,))
        else:
            msg = 'Specify either "fraction" or "fixed" as power split control rule'
            raise ValueError(msg)

        eta = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('power_out_A', units='W', desc='Output power or load to A', shape=(nn,))
        self.add_output('power_out_B', units='W', desc='Output power or load to B', shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out', shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Splitter component cost')
        self.add_output('component_weight', units='kg', desc='Splitter component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power', shape=(nn,))

        if rule == 'fraction':
            self.declare_partials(['power_out_A', 'power_out_B'],
                                  ['power_in', 'power_split_fraction'],
                                  rows=range(nn), cols=range(nn))
        elif rule == 'fixed':
            self.declare_partials(['power_out_A', 'power_out_B'],
                                  ['power_in', 'power_split_amount'],
                                  rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'power_in', val=(1 - eta) * np.ones(nn),
                              rows=range(nn), cols=range(nn))
        self.declare_partials('component_cost', 'power_rating', val=cost_inc)
        self.declare_partials('component_weight', 'power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin', 'power_in',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('component_sizing_margin', 'power_rating')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        rule = self.options['rule']
        eta = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        if rule == 'fraction':
            outputs['power_out_A'] = inputs['power_in'] * inputs['power_split_fraction'] * eta
            outputs['power_out_B'] = inputs['power_in'] * (1 - inputs['power_split_fraction']) * eta
        elif rule == 'fixed':
            # check to make sure enough power is available
            # if inputs['power_in'] < inputs['power_split_amount']:
            not_enough_idx = np.where(inputs['power_in'] < inputs['power_split_amount'])
            po_A = np.zeros(nn)
            po_B = np.zeros(nn)
            po_A[not_enough_idx] = inputs['power_in'][not_enough_idx] * eta
            po_B[not_enough_idx] = np.zeros(nn)[not_enough_idx]
            # else:
            enough_idx = np.where(inputs['power_in'] >= inputs['power_split_amount'])
            po_A[enough_idx] = inputs['power_split_amount'][enough_idx] * eta
            po_B[enough_idx] = (inputs['power_in'][enough_idx] -
                                inputs['power_split_amount'][enough_idx]) * eta
            outputs['power_out_A'] = po_A
            outputs['power_out_B'] = po_B
        outputs['heat_out'] = inputs['power_in'] * (1 - eta)
        outputs['component_cost'] = inputs['power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = inputs['power_in'] / inputs['power_rating']

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        rule = self.options['rule']
        eta = self.options['efficiency']
        if rule == 'fraction':
            J['power_out_A', 'power_in'] = inputs['power_split_fraction'] * eta
            J['power_out_A', 'power_split_fraction'] = inputs['power_in'] * eta
            J['power_out_B', 'power_in'] = (1 - inputs['power_split_fraction']) * eta
            J['power_out_B', 'power_split_fraction'] = -inputs['power_in'] * eta
        elif rule == 'fixed':
            not_enough_idx = np.where(inputs['power_in'] < inputs['power_split_amount'])
            enough_idx = np.where(inputs['power_in'] >= inputs['power_split_amount'])
            # if inputs['power_in'] < inputs['power_split_amount']:
            Jpo_A_pi = np.zeros(nn)
            Jpo_A_ps = np.zeros(nn)
            Jpo_B_pi = np.zeros(nn)
            Jpo_B_ps = np.zeros(nn)
            Jpo_A_pi[not_enough_idx] = eta * np.ones(nn)[not_enough_idx]
            Jpo_A_ps[not_enough_idx] = np.zeros(nn)[not_enough_idx]
            Jpo_B_pi[not_enough_idx] = np.zeros(nn)[not_enough_idx]
            Jpo_B_ps[not_enough_idx] = np.zeros(nn)[not_enough_idx]
            # else:
            Jpo_A_ps[enough_idx] = eta * np.ones(nn)[enough_idx]
            Jpo_A_pi[enough_idx] = np.zeros(nn)[enough_idx]
            Jpo_B_ps[enough_idx] = -eta * np.ones(nn)[enough_idx]
            Jpo_B_pi[enough_idx] = eta * np.ones(nn)[enough_idx]
            J['power_out_A', 'power_in'] = Jpo_A_pi
            J['power_out_A', 'power_split_amount'] = Jpo_A_ps
            J['power_out_B', 'power_in'] = Jpo_B_pi
            J['power_out_B', 'power_split_amount'] = Jpo_B_ps
        J['component_sizing_margin', 'power_in'] = 1 / inputs['power_rating']
        J['component_sizing_margin', 'power_rating'] = - (inputs['power_in'] /
                                                          inputs['power_rating'] ** 2)


class FlowSplit(ExplicitComponent):
    """
    Split incoming flow from one inlet into two outlets at a fractional ratio.

    Inputs
    ------
    mdot_in : float
        Mass flow rate of incoming fluid (vector, kg/s)
    mdot_split_fraction : float
        Fraction of incoming mass flow directed to output A, must be in
        range 0-1 inclusive (vector, dimensionless)
    
    Outputs
    -------
    mdot_out_A : float
        Mass flow rate directed to first output (vector, kg/s)
    mdot_out_B : float
        Mass flow rate directed to second output (vector, kg/s)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
    
    def setup(self):
        nn = self.options['num_nodes']
        rng = np.arange(0, nn)

        self.add_input('mdot_in', units='kg/s', shape=(nn,))
        self.add_input('mdot_split_fraction', units=None, shape=(nn,), val=0.5)

        self.add_output('mdot_out_A', units='kg/s', shape=(nn,))
        self.add_output('mdot_out_B', units='kg/s', shape=(nn,))

        self.declare_partials(['mdot_out_A'], ['mdot_in', 'mdot_split_fraction'], rows=rng, cols=rng)
        self.declare_partials(['mdot_out_B'], ['mdot_in', 'mdot_split_fraction'], rows=rng, cols=rng)
    
    def compute(self, inputs, outputs):
        if np.any(inputs['mdot_split_fraction'] < 0) or np.any(inputs['mdot_split_fraction'] > 1):
            raise RuntimeWarning(f"mdot_split_fraction of {inputs['mdot_split_fraction']} has at least one element out of range [0, 1]")
        outputs['mdot_out_A'] = inputs['mdot_in'] * inputs['mdot_split_fraction']
        outputs['mdot_out_B'] = inputs['mdot_in'] * (1 - inputs['mdot_split_fraction'])

    def compute_partials(self, inputs, J):
        J['mdot_out_A', 'mdot_in'] = inputs['mdot_split_fraction']
        J['mdot_out_A', 'mdot_split_fraction'] = inputs['mdot_in']

        J['mdot_out_B', 'mdot_in'] = 1 - inputs['mdot_split_fraction']
        J['mdot_out_B', 'mdot_split_fraction'] = - inputs['mdot_in']


class FlowCombine(ExplicitComponent):
    """
    Combines two incoming flows into a single outgoing flow and does a weighted average
    of their temperatures based on the mass flow rate of each to compute the outlet temp.

    Inputs
    ------
    mdot_in_A : float
        Mass flow rate of fluid from first inlet, should be nonegative (vector, kg/s)
    mdot_in_B : float
        Mass flow rate of fluid from second inlet, should be nonnegative (vector, kg/s)
    T_in_A : float
        Temperature of fluid from first inlet (vector, K)
    T_in_B : float
        Temperature of fluid from second inlet (vector, K)

    Outputs
    -------
    mdot_out : float
        Outgoing fluid mass flow rate (vector, kg/s)
    T_out : float
        Outgoing fluid temperature (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points (scalar, default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
    
    def setup(self):
        nn = self.options['num_nodes']
        rng = np.arange(0, nn)

        self.add_input('mdot_in_A', units='kg/s', shape=(nn,))
        self.add_input('mdot_in_B', units='kg/s', shape=(nn,))
        self.add_input('T_in_A', units='K', shape=(nn,))
        self.add_input('T_in_B', units='K', shape=(nn,))

        self.add_output('mdot_out', units='kg/s', shape=(nn,))
        self.add_output('T_out', units='K', shape=(nn,))

        self.declare_partials(['mdot_out'], ['mdot_in_A', 'mdot_in_B'], rows=rng, cols=rng)
        self.declare_partials(['T_out'], ['mdot_in_A', 'mdot_in_B', 'T_in_A', 'T_in_B'], rows=rng, cols=rng)
    
    def compute(self, inputs, outputs):
        mdot_A = inputs['mdot_in_A']
        mdot_B = inputs['mdot_in_B']
        outputs['mdot_out'] = mdot_A + mdot_B
        # Weighted average of temperatures for output temperature
        outputs['T_out'] = (mdot_A * inputs['T_in_A'] + mdot_B * inputs['T_in_B']) / (mdot_A + mdot_B)

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        J['mdot_out', 'mdot_in_A'] = np.ones((nn,))
        J['mdot_out', 'mdot_in_B'] = np.ones((nn,))

        mdot_A = inputs['mdot_in_A']
        mdot_B = inputs['mdot_in_B']
        mdot = mdot_A + mdot_B
        T_A = inputs['T_in_A']
        T_B = inputs['T_in_B']
        J['T_out', 'mdot_in_A'] = (mdot * T_A - mdot_A * T_A - mdot_B * T_B) / (mdot**2)
        J['T_out', 'mdot_in_B'] = (mdot * T_B - mdot_A * T_A - mdot_B * T_B) / (mdot**2)
        J['T_out', 'T_in_A'] = mdot_A / mdot
        J['T_out', 'T_in_B'] = mdot_B / mdot