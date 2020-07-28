import openmdao.api as om 
import numpy as np
from openconcept.utilities.math.integrals import NewIntegrator
import warnings 

# OpenConcept PhaseGroup will be used to hold analysis phases with time integration
def find_integrators_in_model(system, abs_namespace, timevars, states):
    durationvar = system._problem_meta['oc_time_var']

    # check if we are a group or not
    if isinstance(system, om.Group):
        for subsys in system._subsystems_allprocs:
            if not abs_namespace:
                next_namespace = subsys.name
            else:
                next_namespace = abs_namespace + '.' + subsys.name
            find_integrators_in_model(subsys, next_namespace, timevars, states)
    else:
        # if the duration variable shows up we need to add its absolute path to timevars
        if isinstance(system, NewIntegrator):
            for varname in system._var_rel_names['input']:
                if varname == durationvar:
                    timevars.append(abs_namespace + '.' + varname)
            for state in system._state_vars.keys():
                states.append(abs_namespace + '.' + state)

class PhaseGroup(om.Group):
    def __init__(self, **kwargs):
        # BB what if user isn't passing num_nodes to the phases?
        num_nodes = kwargs.get('num_nodes', 1)
        super(PhaseGroup, self).__init__(**kwargs)
        self._oc_time_var_name = 'duration'
        self._oc_num_nodes = num_nodes

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=0)

    def _setup_procs(self, pathname, comm, mode, prob_meta):
        # need to pass down the name of the duration variable via prob_meta
        prob_meta.update({'oc_time_var': self._oc_time_var_name})
        prob_meta.update({'oc_num_nodes': self._oc_num_nodes})
        super(PhaseGroup, self)._setup_procs(pathname, comm, mode, prob_meta)
    
    def _configure(self):
        super(PhaseGroup, self)._configure()
        # check child subsys for variables to be integrated and add them all
        timevars = []
        states = []
        find_integrators_in_model(self, '', timevars, states)
        # make connections from duration to integrated vars automatically
        for var_abs_address in timevars:
            self.connect(self._oc_time_var_name, var_abs_address)
        self._oc_states_list = states

class IntegratorGroup(om.Group):
    def __init__(self, **kwargs):
        # BB what if user isn't passing num_nodes to the phases?
        time_units = kwargs.pop('time_units', 's')
        super(IntegratorGroup, self).__init__(**kwargs)
        self._oc_time_units = time_units

    def _setup_procs(self, pathname, comm, mode, prob_meta):
        time_units = self._oc_time_units
        try:
            num_nodes = prob_meta['oc_num_nodes']
        except KeyError:
            raise NameError('Integrator group must be created within an OpenConcept phase')
        self.add_subsystem('ode_integ', NewIntegrator(time_setup='duration', diff_units=time_units, num_nodes=num_nodes))
        super(IntegratorGroup, self)._setup_procs(pathname, comm, mode, prob_meta)

    def _configure(self):
        super(IntegratorGroup, self)._configure()
        for subsys in self._subsystems_allprocs:
            for var in subsys._var_rel_names['output']:
                # check if there are any variables to integrate
                tags = subsys._var_rel2meta[var]['tags']
                if 'integrate' in tags:
                    state_name = None
                    state_units = None
                    state_val = 0.0
                    state_lower = -1e30
                    state_upper = 1e30
                    state_promotes = False
                    # TODO Check for duplicates otherwise generic Openmdao duplicate output/input error raised

                    for tag in tags:
                        split_tag = tag.split(':')
                        if split_tag[0] == 'state_name':
                            state_name = split_tag[-1]
                        elif split_tag[0] == 'state_units':
                            state_units = split_tag[-1]
                        elif split_tag[0] == 'state_val':
                            state_val = eval(split_tag[-1])
                        elif split_tag[0] == 'state_lower':
                            state_lower = float(split_tag[-1])
                        elif split_tag[0] == 'state_upper':
                            state_upper = float(split_tag[-1])
                        elif split_tag[0] == 'state_promotes':
                            state_promotes = eval(split_tag[-1])
                    if state_name is None:
                        raise ValueError('Must provide a state_name tag for integrated variable '+subsys.name+'.'+var)
                    if state_units is None:
                        warnings.warn('OpenConcept integration variable '+subsys.name+'.'+var+' '+'has no units specified. This can be dangerous.')
                    self.ode_integ.add_integrand(state_name, rate_name=var, val=state_val,
                                       units=state_units, lower=state_lower, upper=state_upper)
                    # make the rate connection
                    self.connect(subsys.name+'.'+var, 'ode_integ'+'.'+var)
                    if state_promotes:
                        self.ode_integ._var_promotes['output'].append(state_name)

class TrajectoryGroup(om.Group):
    def __init__(self, **kwargs):
        super(TrajectoryGroup, self).__init__(**kwargs)
        self._oc_phases_to_link = []

    def _configure(self):
        super(TrajectoryGroup, self)._configure()
        for linkage in self._oc_phases_to_link:
            self._link_phases(linkage[0], linkage[1], linkage[2])
        
    def _link_phases(self, phase1, phase2, states_to_skip=[]):
        # find all the states in each phase
        # if they appear in both phase1 and phase2, connect them
        #   unless the state is in states_to_skip
        # if they do not appear in both, do nothing or maybe raise an error message
        # print a report of states linked
        phase1_states = phase1._oc_states_list
        phase2_states = phase2._oc_states_list
        for state in phase1_states:
            if state in phase2_states and not state in states_to_skip:
                state_phase1_abs_name = phase1.name + '.' + state + '_final'
                state_phase2_abs_name = phase2.name + '.' + state + '_initial'
                self.connect(state_phase1_abs_name, state_phase2_abs_name)

    def link_phases(self, phase1, phase2, states_to_skip=[]):
        # need to cache this because the data we need isn't ready yet
        if not isinstance(phase1, PhaseGroup) or not isinstance(phase2, PhaseGroup):
            raise ValueError('link_phases phase arguments must be OpenConcept PhaseGroup objects')
        self._oc_phases_to_link.append((phase1, phase2, states_to_skip))