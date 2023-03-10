import openmdao.api as om
import numpy as np
from openconcept.utilities import Integrator, AddSubtractComp, ElementMultiplyDivideComp

import warnings


# OpenConcept PhaseGroup will be used to hold analysis phases with time integration
def find_integrators_in_model(system, abs_namespace, timevars, states):
    durationvar = system._problem_meta["oc_time_var"]

    # check if we are a group or not
    if isinstance(system, om.Group):
        for subsys, _ in system._subsystems_allprocs.values():
            if not abs_namespace:
                next_namespace = subsys.name
            else:
                next_namespace = abs_namespace + "." + subsys.name
            find_integrators_in_model(subsys, next_namespace, timevars, states)
    else:
        # if the duration variable shows up we need to add its absolute path to timevars
        if isinstance(system, Integrator):
            for varname in system._var_rel_names["input"]:
                if varname == durationvar:
                    timevars.append(abs_namespace + "." + varname)
            for state in system._state_vars.keys():
                state_options = system._state_vars[state]
                state_tuple = (
                    abs_namespace + "." + state_options["name"],
                    abs_namespace + "." + state_options["start_name"],
                    abs_namespace + "." + state_options["end_name"],
                )
                states.append(state_tuple)


class PhaseGroup(om.Group):
    def __init__(self, **kwargs):
        # BB what if user isn't passing num_nodes to the phases?
        num_nodes = kwargs.get("num_nodes", 1)
        super(PhaseGroup, self).__init__(**kwargs)
        self._oc_time_var_name = "duration"
        self._oc_num_nodes = num_nodes

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int, lower=0)

    def _setup_procs(self, pathname, comm, mode, prob_meta):
        # need to pass down the name of the duration variable via prob_meta
        prob_meta.update({"oc_time_var": self._oc_time_var_name})
        prob_meta.update({"oc_num_nodes": self._oc_num_nodes})
        super(PhaseGroup, self)._setup_procs(pathname, comm, mode, prob_meta)

    def configure(self):
        # check child subsys for variables to be integrated and add them all
        timevars = []
        states = []
        find_integrators_in_model(self, "", timevars, states)
        self._setup_var_data()

        # make connections from duration to integrated vars automatically
        time_prom_addresses_already_connected = []
        for var_abs_address in timevars:
            if self.pathname:
                var_abs_address = self.pathname + "." + var_abs_address
            var_prom_address = self._var_abs2prom["input"][var_abs_address]
            if (
                var_prom_address != self._oc_time_var_name
                and var_prom_address not in time_prom_addresses_already_connected
            ):
                self.connect(self._oc_time_var_name, var_prom_address)
                time_prom_addresses_already_connected.append(var_prom_address)
        self._oc_states_list = states


class IntegratorGroup(om.Group):
    def __init__(self, **kwargs):
        # BB what if user isn't passing num_nodes to the phases?
        time_units = kwargs.pop("time_units", "s")
        super(IntegratorGroup, self).__init__(**kwargs)
        self._oc_time_units = time_units
        self._n_auto_comps = 0

    def promote_add(
        self,
        sources,
        prom_name,
        promoted_sources=[],
        factors=None,
        vec_size=1,
        units=None,
        length=1,
        val=1.0,
        res_units=None,
        desc="",
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=None,
    ):
        """Helper function called during setup"""
        add_index = self._n_auto_comps
        self._n_auto_comps += 1
        adder = AddSubtractComp()
        n_inputs = len(sources) + len(promoted_sources)
        if factors is None:
            factors = np.ones(n_inputs)
        adder.add_equation(
            output_name=prom_name,
            input_names=["_temp" + str(i) for i in range(n_inputs)],
            scaling_factors=factors,
            vec_size=vec_size,
            units=units,
            length=length,
            val=val,
            res_units=res_units,
            desc=desc,
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            res_ref=res_ref,
        )
        adder_name = "add" + str(add_index)
        prom_in = []
        for i, promoted_source in enumerate(promoted_sources):
            prom_in.append(("_temp" + str(i + len(sources)), promoted_source))

        self.add_subsystem(adder_name, adder, promotes_inputs=prom_in, promotes_outputs=["*"])
        for i, source in enumerate(sources):
            self.connect(source, adder_name + "._temp" + str(i))

    def promote_mult(
        self,
        source,
        prom_name,
        factor,
        vec_size=1,
        units=None,
        length=1,
        val=1.0,
        res_units=None,
        desc="",
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=None,
        divide=None,
        input_units=None,
        tags=None,
    ):
        """Helper function called during setup"""
        mult_index = self._n_auto_comps
        self._n_auto_comps += 1
        mult = ElementMultiplyDivideComp(
            output_name=prom_name,
            input_names=["_temp"],
            scaling_factor=factor,
            input_units=[units],
            vec_size=vec_size,
            length=length,
            val=val,
            res_units=res_units,
            desc=desc,
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            res_ref=res_ref,
            divide=divide,
            tags=tags,
        )
        mult_name = "mult" + str(mult_index)
        self.add_subsystem(mult_name, mult, promotes_outputs=["*"])
        self.connect(source, mult_name + "._temp")

    def _setup_procs(self, pathname, comm, mode, prob_meta):
        time_units = self._oc_time_units
        self._under_dymos = False
        self._under_openconcept = False
        try:
            num_nodes = prob_meta["oc_num_nodes"]
            self._under_openconcept = True
        except KeyError:
            # TODO test_if_under_dymos
            if not self._under_dymos:
                raise NameError("Integrator group must be created within an OpenConcept phase or Dymos trajectory")

        if self._under_openconcept:
            self.add_subsystem(
                "ode_integ",
                Integrator(time_setup="duration", method="simpson", diff_units=time_units, num_nodes=num_nodes),
            )

        super(IntegratorGroup, self)._setup_procs(pathname, comm, mode, prob_meta)

    def configure(self):
        self._setup_var_data()
        parent_meta_dict = self.get_io_metadata(iotypes="output", tags="integrate")
        for subsys, _ in self._subsystems_allprocs.values():
            # exclude any group subclasses, they don't have real outputs
            if not isinstance(subsys, om.Group):
                # find any variables that have an 'integrate tag
                child_meta_dict = subsys.get_io_metadata(iotypes="output", tags="integrate")
                for var in child_meta_dict.keys():
                    tags = child_meta_dict[var]["tags"]
                    state_name = None
                    state_units = None
                    state_val = 0.0
                    state_lower = -1e30
                    state_upper = 1e30
                    state_promotes = False
                    # TODO Check for duplicates otherwise generic Openmdao duplicate output/input error raised

                    for tag in tags:
                        split_tag = tag.split(":")
                        if split_tag[0] == "state_name":
                            state_name = split_tag[-1]
                        elif split_tag[0] == "state_units":
                            state_units = split_tag[-1]
                        elif split_tag[0] == "state_val":
                            state_val = eval(split_tag[-1])
                        elif split_tag[0] == "state_lower":
                            state_lower = float(split_tag[-1])
                        elif split_tag[0] == "state_upper":
                            state_upper = float(split_tag[-1])
                        elif split_tag[0] == "state_promotes":
                            state_promotes = eval(split_tag[-1])
                    if state_name is None:
                        raise ValueError(
                            "Must provide a state_name tag for integrated variable " + subsys.name + "." + var
                        )
                    if state_units is None:
                        warnings.warn(
                            "OpenConcept integration variable "
                            + subsys.name
                            + "."
                            + var
                            + " "
                            + "has no units specified. This can be dangerous.",
                            stacklevel=2,
                        )
                    self.ode_integ.add_integrand(
                        state_name,
                        rate_name=var,
                        val=state_val,
                        units=state_units,
                        lower=state_lower,
                        upper=state_upper,
                    )
                    # make the rate connection
                    rate_var_abs_address = subsys.name + "." + var
                    # if self.pathname:
                    #     rate_var_abs_address = self.pathname + '.' + rate_var_abs_address
                    rate_var_prom_address = parent_meta_dict[rate_var_abs_address]["prom_name"]
                    self.connect(rate_var_prom_address, "ode_integ" + "." + var)
                    if state_promotes:
                        self.ode_integ._var_promotes["output"].append((state_name, None))
                        self.ode_integ._var_promotes["output"].append((state_name + "_final", None))
                        self.ode_integ._var_promotes["input"].append((state_name + "_initial", None))


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
        self._setup_var_data()
        for state_tuple in phase1_states:
            if state_tuple[0] in [state_tuple_2[0] for state_tuple_2 in phase2_states]:
                phase1_abs_name = phase1.name + "." + state_tuple[0]
                phase1_end_abs_name = phase1.name + "." + state_tuple[2]  # final
                phase2_start_abs_name = phase2.name + "." + state_tuple[1]  # initial
                if self.pathname:
                    phase1_abs_name = self.pathname + "." + phase1_abs_name
                    phase1_end_abs_name = self.pathname + "." + phase1_end_abs_name
                    phase2_start_abs_name = self.pathname + "." + phase2_start_abs_name

                phase1_prom_name = self._var_abs2prom["output"][phase1_abs_name]
                if phase1_prom_name.startswith(phase1.name):  # only modify the text if it starts with the prefix
                    state_prom_name = phase1_prom_name.replace(phase1.name + ".", "", 1)
                else:
                    state_prom_name = phase1_prom_name
                phase1_end_prom_name = self._var_abs2prom["output"][phase1_end_abs_name]
                phase2_start_prom_name = self._var_abs2prom["input"][phase2_start_abs_name]
                if not (state_tuple[0] in states_to_skip):
                    if not (state_prom_name in states_to_skip):
                        self.connect(phase1_end_prom_name, phase2_start_prom_name)

    def link_phases(self, phase1, phase2, states_to_skip=[]):
        # need to cache this because the data we need isn't ready yet
        if not isinstance(phase1, PhaseGroup) or not isinstance(phase2, PhaseGroup):
            raise ValueError("link_phases phase arguments must be OpenConcept PhaseGroup objects")
        self._oc_phases_to_link.append((phase1, phase2, states_to_skip))
