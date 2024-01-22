import unittest
from openmdao.utils.assert_utils import assert_near_equal
from openconcept.thermal import ImplicitCompressibleDuct_ExternalHX
import openmdao.api as om
import warnings

try:
    import pycycle

    pyc_major_version = int(pycycle.__version__.split(".")[0])
    if pyc_major_version >= 4:
        HAS_PYCYCLE = True
        import pycycle.api as pyc

        class PyCycleDuct(pyc.Cycle):
            """
            This is tested with pycycle master as of 10 March 2021
            (commit 2cca1bdee53f6f5cd0d340ee2a12b62d49f7721d)
            """

            def setup(self):
                design = self.options["design"]

                USE_TABULAR = False
                if USE_TABULAR:
                    self.options["thermo_method"] = "TABULAR"
                    self.options["thermo_data"] = pyc.AIR_JETA_TAB_SPEC
                else:
                    self.options["thermo_method"] = "CEA"
                    self.options["thermo_data"] = pyc.species_data.janaf

                self.add_subsystem("fc", pyc.FlightConditions())
                # ram_recovery | ram_recovery
                # MN | area
                self.add_subsystem("inlet", pyc.Inlet())
                # dPqP | s_dPqP
                # Q_dot | Q_dot
                # MN | area
                self.add_subsystem("duct", pyc.Duct())
                # Ps_exhaust
                # dPqP
                self.add_subsystem("nozz", pyc.Nozzle(lossCoef="Cfg"))
                self.add_subsystem("perf", pyc.Performance(num_nozzles=1, num_burners=0))

                balance = om.BalanceComp()
                if design:
                    self.add_subsystem("iv", om.IndepVarComp("nozzle_area", 60.0, units="inch**2"))
                    balance.add_balance("W", units="kg/s", eq_units="inch**2", val=2.0, lower=0.05, upper=10.0)
                    self.add_subsystem("balance", balance)
                    self.connect("iv.nozzle_area", "balance.rhs:W")
                    self.connect("nozz.Throat:stat:area", "balance.lhs:W")
                else:
                    balance.add_balance("W", units="kg/s", eq_units="inch**2", val=2.0, lower=0.05, upper=10.0)
                    self.add_subsystem("balance", balance)
                    self.connect("nozz.Throat:stat:area", "balance.lhs:W")

                self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I")
                self.pyc_connect_flow("inlet.Fl_O", "duct.Fl_I")
                self.pyc_connect_flow("duct.Fl_O", "nozz.Fl_I")

                self.connect("fc.Fl_O:stat:P", "nozz.Ps_exhaust")
                self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
                self.connect("duct.Fl_O:tot:P", "perf.Pt3")
                self.connect("inlet.F_ram", "perf.ram_drag")
                self.connect("nozz.Fg", "perf.Fg_0")

                self.connect("balance.W", "fc.W")

                newton = self.nonlinear_solver = om.NewtonSolver()
                newton.options["atol"] = 1e-12
                newton.options["rtol"] = 1e-12
                newton.options["iprint"] = 2
                newton.options["maxiter"] = 10
                newton.options["solve_subsystems"] = True
                newton.options["max_sub_solves"] = 10
                newton.options["reraise_child_analysiserror"] = False

                newton.linesearch = om.BoundsEnforceLS()
                newton.linesearch.options["bound_enforcement"] = "scalar"
                # newton.linesearch.options['print_bound_enforce'] = True
                # newton.linesearch.options['iprint'] = -1
                self.linear_solver = om.DirectSolver(assemble_jac=True)
                super().setup()

        def viewer(prob, pt):
            """
            print a report of all the relevant cycle properties
            """

            fs_names = ["fc.Fl_O", "inlet.Fl_O", "duct.Fl_O", "nozz.Fl_O"]
            fs_full_names = [f"{pt}.{fs}" for fs in fs_names]
            pyc.print_flow_station(prob, fs_full_names)

            # pyc.print_compressor(prob, [f'{pt}.fan'])

            pyc.print_nozzle(prob, [f"{pt}.nozz"])

            summary_data = (
                prob[pt + ".fc.Fl_O:stat:MN"],
                prob[pt + ".fc.alt"],
                prob[pt + ".inlet.Fl_O:stat:W"],
                prob[pt + ".perf.Fn"],
                prob[pt + ".perf.Fg"],
                prob[pt + ".inlet.F_ram"],
                prob[pt + ".perf.OPR"],
            )

            print("----------------------------------------------------------------------------")
            print("                              POINT:", pt)
            print("----------------------------------------------------------------------------")
            print("                       PERFORMANCE CHARACTERISTICS")
            print("    Mach      Alt       W      Fn      Fg    Fram     OPR ")
            print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f" % summary_data)

        class MPDuct(pyc.MPCycle):
            def setup(self):
                self.pyc_add_pnt("design", PyCycleDuct(design=True, thermo_method="CEA"))

                # define the off-design conditions we want to run
                self.od_pts = []
                # self.od_pts = ['off_design']
                # self.od_MNs = [0.8,]
                # self.od_alts = [10000,]

                # for i, pt in enumerate(self.od_pts):
                #     self.pyc_add_pnt(pt, PyCycleDuct(design=False))

                #     self.set_input_defaults(pt+'.fc.MN', val=self.od_MNs[i])
                #     self.set_input_defaults(pt+'.fc.alt', val=self.od_alts, units='m')

                # self.pyc_use_default_des_od_conns()

                # self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')
                super().setup()

        def print_truth(name, value, list_output=True):
            if list_output:
                print(name + ": " + str(value))

        def check_params_match_pycycle(prob, list_output=True, case_name=""):
            list_of_vals = []
            if list_output:
                print("=========" + case_name + "=============")
            mdot_pyc = prob.get_val("pyduct.design.fc.Fl_O:stat:W", units="kg/s")
            mdot_oc = prob.get_val("oc.mdot", units="kg/s")
            assert_near_equal(mdot_oc, mdot_pyc, tolerance=1e-4)
            list_of_vals.append(mdot_pyc[0])
            print_truth("mass flow", mdot_pyc, list_output)

            fnet_pyc = prob.get_val("pyduct.design.perf.Fn", units="N")
            fnet_oc = prob.get_val("oc.force.F_net", units="N")
            assert_near_equal(fnet_oc, fnet_pyc, tolerance=1e-4)
            list_of_vals.append(fnet_pyc[0])

            print_truth("net force", fnet_pyc, list_output)

            # compare the flow conditions at each station
            oc_stations = ["inlet", "sta1", "sta3", "nozzle"]
            state_units = ["K", "Pa", "kg/m**3", None, "m/s", "K", "Pa", "inch**2"]
            for i, pyc_station in enumerate(["fc", "inlet", "duct", "nozz"]):
                oc_station = oc_stations[i]
                if list_output:
                    print("--------" + pyc_station + "-------------")
                if oc_station == "nozzle" or oc_station == "inlet":
                    oc_states = ["T", "p", "rho", "M", "a", "Tt", "pt", "area"]
                else:
                    oc_states = ["T", "p", "rho", "M", "a", "Tt_out", "pt_out", "area"]
                for j, pyc_state in enumerate(
                    ["stat:T", "stat:P", "stat:rho", "stat:MN", "stat:Vsonic", "tot:T", "tot:P", "stat:area"]
                ):
                    oc_state = oc_states[j]
                    if oc_station == "inlet" and oc_state in ["rho", "area"]:
                        continue
                    state_pyc = prob.get_val(
                        "pyduct.design." + pyc_station + ".Fl_O:" + pyc_state, units=state_units[j]
                    )
                    state_oc = prob.get_val("oc." + oc_station + "." + oc_state, units=state_units[j])
                    assert_near_equal(state_oc, state_pyc, tolerance=5e-4)
                    list_of_vals.append(state_pyc[0])
                    print_truth(oc_state, state_pyc, list_output)
            print(list_of_vals)

    else:
        HAS_PYCYCLE = False


except ImportError:
    HAS_PYCYCLE = False


def run_problem(
    ram_recovery=1.0,
    dPqP=0.0,
    heat_in=0.0,
    cfg=0.98,
    oc_use_dpqp=False,
    list_output=True,
    oc_areas=None,
    oc_delta_p=0.0,
):
    prob = om.Problem()
    model = prob.model = om.Group()

    iv = model.add_subsystem("iv", om.IndepVarComp())
    iv.add_output("area_2", val=408, units="inch**2")

    # add the pycycle duct
    if HAS_PYCYCLE:
        mp_duct = model.add_subsystem("pyduct", MPDuct())
        prob.model.connect("pyduct.design.fc.Fl_O:stat:T", "fltcond|T")
        prob.model.connect("pyduct.design.fc.Fl_O:stat:P", "fltcond|p")
        prob.model.connect("pyduct.design.fc.Fl_O:stat:V", "fltcond|Utrue")
    else:
        iv.add_output("fltcond|T", val=223.15013852435118, units="K")
        iv.add_output("fltcond|p", val=26436.23048846425, units="Pa")
        iv.add_output("fltcond|Utrue", val=0.8 * 299.57996571373235, units="m/s")
        prob.model.connect("iv.fltcond|T", "fltcond|T")
        prob.model.connect("iv.fltcond|p", "fltcond|p")
        prob.model.connect("iv.fltcond|Utrue", "fltcond|Utrue")

    # add the openconcept duct
    oc = model.add_subsystem(
        "oc",
        ImplicitCompressibleDuct_ExternalHX(num_nodes=1, cfg=cfg),
        promotes_inputs=[("p_inf", "fltcond|p"), ("T_inf", "fltcond|T"), ("Utrue", "fltcond|Utrue")],
    )

    newton = oc.nonlinear_solver = om.NewtonSolver()
    newton.options["atol"] = 1e-12
    newton.options["rtol"] = 1e-12
    newton.options["iprint"] = -1
    newton.options["maxiter"] = 10
    newton.options["solve_subsystems"] = True
    newton.options["max_sub_solves"] = 10
    newton.options["reraise_child_analysiserror"] = False
    newton.linesearch = om.BoundsEnforceLS()
    newton.linesearch.options["bound_enforcement"] = "scalar"
    oc.linear_solver = om.DirectSolver(assemble_jac=True)

    prob.model.connect("iv.area_2", ["oc.area_2", "oc.area_3"])

    # iv.add_output('cp', val=1002.93, units='J/kg/K')
    # iv.add_output('pressure_recovery_1', val=np.ones((nn,)))
    # iv.add_output('loss_factor_1', val=0.0)
    # iv.add_output('delta_p_2', val=np.ones((nn,))*0., units='Pa')
    # iv.add_output('heat_in_2', val=np.ones((nn,))*0., units='W')
    # iv.add_output('pressure_recovery_2', val=np.ones((nn,)))
    # iv.add_output('pressure_recovery_3', val=np.ones((nn,)))

    prob.setup()
    prob.set_val("oc.area_1", val=64, units="inch**2")
    prob.set_val("oc.convergence_hack", val=0.0, units="Pa")
    prob.set_val("oc.area_nozzle_in", val=60.0, units="inch**2")
    prob.set_val("oc.inlet.totalpressure.eta_ram", val=ram_recovery)

    if HAS_PYCYCLE:
        # Define the design point
        prob.set_val("pyduct.design.fc.alt", 10000, units="m")
        prob.set_val("pyduct.design.fc.MN", 0.8)
        prob.set_val("pyduct.design.inlet.MN", 0.6)
        prob.set_val("pyduct.design.inlet.ram_recovery", ram_recovery)
        prob.set_val("pyduct.design.duct.MN", 0.08)
        prob.set_val("pyduct.design.duct.dPqP", dPqP)
        prob.set_val("pyduct.design.duct.Q_dot", heat_in, units="kW")
        prob.set_val("pyduct.design.nozz.Cfg", cfg, units=None)

        # Set initial guesses for balances
        prob["pyduct.design.balance.W"] = 8.0
        prob.model.pyduct.design.nonlinear_solver.options["atol"] = 1e-6
        prob.model.pyduct.design.nonlinear_solver.options["rtol"] = 1e-6

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=-1, depth=2)

    # do a first run to get the duct areas from pycycle
    prob.run_model()

    if HAS_PYCYCLE:
        # set areas based on pycycle design point
        prob.set_val(
            "oc.area_1", val=prob.get_val("pyduct.design.inlet.Fl_O:stat:area", units="inch**2"), units="inch**2"
        )
        prob.set_val(
            "iv.area_2", val=prob.get_val("pyduct.design.duct.Fl_O:stat:area", units="inch**2"), units="inch**2"
        )
    else:
        prob.set_val("oc.area_1", val=oc_areas[0], units="inch**2")
        prob.set_val("iv.area_2", val=oc_areas[1], units="inch**2")

    prob.set_val("oc.sta3.heat_in", val=heat_in, units="kW")
    if oc_use_dpqp:
        prob.set_val("oc.sta3.pressure_recovery", val=(1 - dPqP), units=None)
    else:
        if HAS_PYCYCLE:
            delta_p = prob.get_val("pyduct.design.inlet.Fl_O:tot:P", units="Pa") - prob.get_val(
                "pyduct.design.nozz.Fl_O:tot:P", units="Pa"
            )
        else:
            delta_p = oc_delta_p
        prob.set_val("oc.sta3.delta_p", -delta_p, units="Pa")

    prob.run_model()

    if list_output and HAS_PYCYCLE:
        prob.model.list_outputs(units=True, excludes=["*chem_eq*", "*props*"])
        # prob.model.list_outputs(includes=['*oc.*force*','*perf*','*mdot*'], units=True)
        print(prob.get_val("pyduct.design.inlet.Fl_O:stat:W", units="kg/s"))
        print(prob.get_val("pyduct.design.perf.Fn", units="N"))

        for pt in ["design"] + mp_duct.od_pts:
            print("\n", "#" * 10, pt, "#" * 10)
            viewer(prob, "pyduct." + pt)
    elif list_output:
        prob.model.list_outputs(units=True)
    return prob


def check_params_match_known(prob, known_vals):
    mdot_oc = prob.get_val("oc.mdot", units="kg/s")
    assert_near_equal(mdot_oc, known_vals.pop(0), tolerance=1e-4)

    fnet_oc = prob.get_val("oc.force.F_net", units="N")
    assert_near_equal(fnet_oc, known_vals.pop(0), tolerance=1e-4)

    # compare the flow conditions at each station
    oc_stations = ["inlet", "sta1", "sta3", "nozzle"]
    state_units = ["K", "Pa", "kg/m**3", None, "m/s", "K", "Pa", "inch**2"]
    for oc_station in oc_stations:
        if oc_station == "nozzle" or oc_station == "inlet":
            oc_states = ["T", "p", "rho", "M", "a", "Tt", "pt", "area"]
        else:
            oc_states = ["T", "p", "rho", "M", "a", "Tt_out", "pt_out", "area"]
        for j, oc_state in enumerate(oc_states):
            if oc_station == "inlet" and oc_state in ["rho", "area"]:
                continue
            state_oc = prob.get_val("oc." + oc_station + "." + oc_state, units=state_units[j])
            assert_near_equal(state_oc, known_vals.pop(0), tolerance=5e-4)


if not HAS_PYCYCLE:

    class TestOCDuct(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            self.list_output = False
            warnings.warn(
                "pycycle >= 3.0 must be installed to run reg tests using pycycle. Using cached values", stacklevel=2
            )
            super(TestOCDuct, self).__init__(*args, **kwargs)

        def test_baseline(self):
            prob = run_problem(
                heat_in=0.0, oc_use_dpqp=False, list_output=False, oc_areas=[68.6660253970519, 419.64492400826833]
            )
            known_vals = [
                3.8288293812130427,
                -18.352679756968048,
                223.15013852435112,
                26436.230488463945,
                0.8,
                299.57996571373224,
                251.78817691084615,
                40308.54640098064,
                234.83838055394062,
                31597.160969709043,
                0.46872831137173837,
                0.6,
                307.3153368247888,
                251.78817691084615,
                40308.54640098064,
                68.6660253970519,
                251.4656086903193,
                40128.37627313919,
                0.5559237001577775,
                0.08,
                317.9885117338898,
                251.78817691084618,
                40308.54640098064,
                419.64492400826833,
                223.1501387055282,
                26436.23069828319,
                0.4127096009002737,
                0.7999999971456052,
                299.57996583521236,
                251.7881769108463,
                40308.54640098064,
                60.00000081736329,
            ]
            check_params_match_known(prob, known_vals)

        def test_delta_p(self):
            prob = run_problem(
                heat_in=5.0,
                oc_use_dpqp=False,
                list_output=False,
                oc_areas=[63.46924960435457, 409.4433340158199],
                oc_delta_p=2015.4273200490352,
            )
            known_vals = [
                3.539056269581751,
                -64.14356947383646,
                223.15013852435118,
                26436.23048846425,
                0.8,
                299.57996571373235,
                251.78817691084626,
                40308.54640098054,
                234.83838055394045,
                31597.160969708944,
                0.46872831137173715,
                0.6,
                307.3153368247886,
                251.78817691084626,
                40308.54640098054,
                63.46924960435457,
                252.87220799749818,
                38121.95962142752,
                0.5251898405386665,
                0.08,
                318.87459959919477,
                253.19656586001645,
                38293.1190809315,
                409.4433340158199,
                227.71856595088528,
                26436.235641634095,
                0.40443001001100554,
                0.7469937048327284,
                302.62735666768816,
                253.19656586001642,
                38293.1190809315,
                60.00003918869304,
            ]
            check_params_match_known(prob, known_vals)

        def test_heat_addition(self):
            prob = run_problem(
                dPqP=0.0,
                heat_in=5.0,
                oc_use_dpqp=False,
                list_output=False,
                oc_areas=[68.48861194954175, 419.6465614913434],
            )
            known_vals = [
                3.818936776878921,
                -15.98286892624089,
                223.15013852435112,
                26436.230488463945,
                0.8,
                299.57996571373224,
                251.78817691084615,
                40308.546400980646,
                234.8383805539406,
                31597.16096970898,
                0.4687283113717375,
                0.6,
                307.31533682478863,
                251.78817691084615,
                40308.546400980646,
                68.48861194954175,
                252.76912333327795,
                40128.37841113136,
                0.5530568656891555,
                0.08,
                318.8097475436091,
                253.0933500013548,
                40308.546400980646,
                419.6465614913434,
                224.30746290344237,
                26436.23069828319,
                0.4105802076038991,
                0.800001851306077,
                300.3549387801823,
                253.0933500013548,
                40308.546400980646,
                60.00000076028102,
            ]
            check_params_match_known(prob, known_vals)

        def test_dpqp(self):
            prob = run_problem(
                dPqP=0.05,
                heat_in=5.0,
                oc_use_dpqp=True,
                list_output=False,
                oc_areas=[63.46924960435457, 409.4433340158199],
            )
            known_vals = [
                3.539056269581751,
                -64.14356947383646,
                223.15013852435118,
                26436.23048846425,
                0.8,
                299.57996571373235,
                251.78817691084626,
                40308.54640098054,
                234.83838055394045,
                31597.160969708944,
                0.46872831137173715,
                0.6,
                307.3153368247886,
                251.78817691084626,
                40308.54640098054,
                63.46924960435457,
                252.87220799749818,
                38121.95962142752,
                0.5251898405386665,
                0.08,
                318.87459959919477,
                253.19656586001645,
                38293.1190809315,
                409.4433340158199,
                227.71856595088528,
                26436.235641634095,
                0.40443001001100554,
                0.7469937048327284,
                302.62735666768816,
                253.19656586001642,
                38293.1190809315,
                60.00003918869304,
            ]
            check_params_match_known(prob, known_vals)

        def test_cfg(self):
            prob = run_problem(
                dPqP=0.05,
                heat_in=5.0,
                oc_use_dpqp=True,
                cfg=0.95,
                list_output=False,
                oc_areas=[63.46924960435457, 409.4433340158199],
            )
            known_vals = [
                3.539056269581751,
                -88.14485507224843,
                223.15013852435118,
                26436.23048846425,
                0.8,
                299.57996571373235,
                251.78817691084626,
                40308.54640098054,
                234.83838055394045,
                31597.160969708944,
                0.46872831137173715,
                0.6,
                307.3153368247886,
                251.78817691084626,
                40308.54640098054,
                63.46924960435457,
                252.87220799749818,
                38121.95962142752,
                0.5251898405386665,
                0.08,
                318.87459959919477,
                253.19656586001645,
                38293.1190809315,
                409.4433340158199,
                227.71856595088528,
                26436.235641634095,
                0.40443001001100554,
                0.7469937048327284,
                302.62735666768816,
                253.19656586001642,
                38293.1190809315,
                60.00003918869304,
            ]
            check_params_match_known(prob, known_vals)

else:

    class TestOCDuct(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            self.list_output = False
            super(TestOCDuct, self).__init__(*args, **kwargs)

        def test_baseline(self):
            prob = run_problem(dPqP=0.0, heat_in=0.0, oc_use_dpqp=False, list_output=False)
            check_params_match_pycycle(prob, list_output=self.list_output, case_name="baseline")

        def test_heat_addition(self):
            prob = run_problem(dPqP=0.0, heat_in=5.0, oc_use_dpqp=False, list_output=False)
            check_params_match_pycycle(prob, list_output=self.list_output, case_name="heat_add")

        def test_delta_p(self):
            prob = run_problem(dPqP=0.05, heat_in=5.0, oc_use_dpqp=False, list_output=False)
            check_params_match_pycycle(prob, list_output=self.list_output, case_name="delta_p")

        def test_dpqp(self):
            prob = run_problem(dPqP=0.05, heat_in=5.0, oc_use_dpqp=True, list_output=False)
            check_params_match_pycycle(prob, list_output=self.list_output, case_name="dpqp")

        def test_cfg(self):
            prob = run_problem(dPqP=0.05, heat_in=5.0, oc_use_dpqp=True, cfg=0.95, list_output=False)
            check_params_match_pycycle(prob, list_output=self.list_output, case_name="cfg")


if __name__ == "__main__":
    unittest.main()
