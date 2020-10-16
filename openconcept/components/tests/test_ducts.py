from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.components.ducts import ImplicitCompressibleDuct_ExternalHX
import openmdao.api as om 
import warnings

try:
    import pycycle
    pyc_version = float(pycycle.__version__)
    if pyc_version >= 3.0:
        HAS_PYCYCLE = True
        import pycycle.api as pyc
    
        class PyCycleDuct(pyc.Cycle):

            def initialize(self):
                self.options.declare('design', types=bool, default=True)

            def setup(self):

                thermo_spec = pyc.species_data.janaf
                design = self.options['design']

                self.pyc_add_element('fc', pyc.FlightConditions(thermo_data=thermo_spec,
                                                        elements=pyc.AIR_MIX))
                # ram_recovery | ram_recovery
                # MN | area
                self.pyc_add_element('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
                # dPqP | s_dPqP
                # Q_dot | Q_dot
                # MN | area
                self.pyc_add_element('duct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
                # Ps_exhaust
                # dPqP
                self.pyc_add_element('nozz', pyc.Nozzle(lossCoef='Cfg',thermo_data=thermo_spec, elements=pyc.AIR_MIX))
                self.pyc_add_element('perf', pyc.Performance(num_nozzles=1, num_burners=0))


                balance = om.BalanceComp()
                if design:
                    self.add_subsystem('iv', om.IndepVarComp('nozzle_area', 60., units='inch**2'))
                    balance.add_balance('W', units='kg/s', eq_units='inch**2', val=2., lower=0.05, upper=10.)
                    self.add_subsystem('balance', balance)
                    self.connect('iv.nozzle_area', 'balance.rhs:W')
                    self.connect('nozz.Throat:stat:area', 'balance.lhs:W')
                else:
                    balance.add_balance('W', units='kg/s', eq_units='inch**2', val=2., lower=0.05, upper=10.)
                    self.add_subsystem('balance', balance)
                    self.connect('nozz.Throat:stat:area', 'balance.lhs:W')


                self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
                self.pyc_connect_flow('inlet.Fl_O', 'duct.Fl_I')
                self.pyc_connect_flow('duct.Fl_O', 'nozz.Fl_I')


                self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
                self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
                self.connect('duct.Fl_O:tot:P', 'perf.Pt3')
                self.connect('inlet.F_ram', 'perf.ram_drag')
                self.connect('nozz.Fg', 'perf.Fg_0')

                self.connect('balance.W', 'fc.W')

                newton = self.nonlinear_solver = om.NewtonSolver()
                newton.options['atol'] = 1e-12
                newton.options['rtol'] = 1e-12
                newton.options['iprint'] = 2
                newton.options['maxiter'] = 10
                newton.options['solve_subsystems'] = True
                newton.options['max_sub_solves'] = 10
                newton.options['reraise_child_analysiserror'] = False

                newton.linesearch = om.BoundsEnforceLS()
                newton.linesearch.options['bound_enforcement'] = 'scalar'
                # newton.linesearch.options['print_bound_enforce'] = True
                # newton.linesearch.options['iprint'] = -1
                self.linear_solver = om.DirectSolver(assemble_jac=True)
            
        def viewer(prob, pt):
            """
            print a report of all the relevant cycle properties
            """

            fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'duct.Fl_O', 'nozz.Fl_O']
            fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
            pyc.print_flow_station(prob, fs_full_names)

            # pyc.print_compressor(prob, [f'{pt}.fan'])

            pyc.print_nozzle(prob, [f'{pt}.nozz'])

            summary_data = (prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'], 
                            prob[pt+'.perf.Fn'], prob[pt+'.perf.Fg'], prob[pt+'.inlet.F_ram'],
                            prob[pt+'.perf.OPR'])

            print("----------------------------------------------------------------------------")
            print("                              POINT:", pt)
            print("----------------------------------------------------------------------------")
            print("                       PERFORMANCE CHARACTERISTICS")
            print("    Mach      Alt       W      Fn      Fg    Fram     OPR ")
            print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f" %summary_data)
    
        class MPDuct(pyc.MPCycle):

            def setup(self):
                design = self.pyc_add_pnt('design', PyCycleDuct(design=True))
                
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

        def run_problem(ram_recovery=1.0, dPqP=0.0, heat_in=0.0, cfg=0.98, oc_use_dpqp=False, list_output=True):
            prob = om.Problem()
            model = prob.model = om.Group()

            iv = model.add_subsystem('iv', om.IndepVarComp())
            iv.add_output('area_2', val=408, units='inch**2')

            # add the pycycle duct
            mp_duct = model.add_subsystem('pyduct', MPDuct())

            # add the openconcept duct
            # TODO pass thru cfg
            oc = model.add_subsystem('oc', ImplicitCompressibleDuct_ExternalHX(num_nodes=1, cfg=cfg), 
                                     promotes_inputs=[('p_inf','fltcond|p'),('T_inf','fltcond|T'),('Utrue','fltcond|Utrue')])

            newton = oc.nonlinear_solver = om.NewtonSolver()
            newton.options['atol'] = 1e-12
            newton.options['rtol'] = 1e-12
            newton.options['iprint'] = -1
            newton.options['maxiter'] = 10
            newton.options['solve_subsystems'] = True
            newton.options['max_sub_solves'] = 10
            newton.options['reraise_child_analysiserror'] = False
            newton.linesearch = om.BoundsEnforceLS()
            newton.linesearch.options['bound_enforcement'] = 'scalar'
            oc.linear_solver = om.DirectSolver(assemble_jac=True)

            prob.model.connect('iv.area_2', ['oc.area_2','oc.area_3'])
            prob.model.connect('pyduct.design.fc.Fl_O:stat:T','fltcond|T')
            prob.model.connect('pyduct.design.fc.Fl_O:stat:P','fltcond|p')
            prob.model.connect('pyduct.design.fc.Fl_O:stat:V','fltcond|Utrue')

            # iv.add_output('cp', val=1002.93, units='J/kg/K')
            # iv.add_output('pressure_recovery_1', val=np.ones((nn,)))
            # iv.add_output('loss_factor_1', val=0.0)
            # iv.add_output('delta_p_2', val=np.ones((nn,))*0., units='Pa')
            # iv.add_output('heat_in_2', val=np.ones((nn,))*0., units='W')
            # iv.add_output('pressure_recovery_2', val=np.ones((nn,)))
            # iv.add_output('pressure_recovery_3', val=np.ones((nn,)))


            prob.setup()
            prob.set_val('oc.area_1', value=64, units='inch**2')
            prob.set_val('oc.convergence_hack', value=0.0, units='Pa')
            prob.set_val('oc.area_nozzle_in', value=60.0, units='inch**2')
            prob.set_val('oc.inlet.totalpressure.eta_ram', value=ram_recovery)

            #Define the design point
            prob.set_val('pyduct.design.fc.alt', 10000, units='m')
            prob.set_val('pyduct.design.fc.MN', 0.8)
            prob.set_val('pyduct.design.inlet.MN', 0.6)
            prob.set_val('pyduct.design.inlet.ram_recovery', ram_recovery)
            prob.set_val('pyduct.design.duct.MN', 0.08)
            prob.set_val('pyduct.design.duct.dPqP', dPqP)
            prob.set_val('pyduct.design.duct.Q_dot', heat_in, units='kW')
            prob.set_val('pyduct.design.nozz.Cfg', cfg, units=None)

            # Set initial guesses for balances
            prob['pyduct.design.balance.W'] = 8.
            
            # for i, pt in enumerate(mp_duct.od_pts):
            
            #     # initial guesses
            #     prob['pyduct.off_design.inlet.ram_recovery'] = 0.96
            #     prob['pyduct.off_design.duct.dPqP'] = 0.05
            #     prob.set_val('pyduct.off_design.duct.Q_dot', 5., units='kW') 
            #     prob['pyduct.off_design.balance.W'] = 8.
            #     prob['pyduct.off_design.nozz.Cfg'] = 0.98
                # prob.model.pyduct.off_design.nonlinear_solver.options['atol'] = 1e-6
                # prob.model.pyduct.off_design.nonlinear_solver.options['rtol'] = 1e-6

            prob.set_solver_print(level=-1)
            prob.set_solver_print(level=-1, depth=2)
            prob.model.pyduct.design.nonlinear_solver.options['atol'] = 1e-6
            prob.model.pyduct.design.nonlinear_solver.options['rtol'] = 1e-6
            # do a first run to get the duct areas from pycycle
            prob.run_model()
            # set areas based on pycycle design point
            prob.set_val('oc.area_1', value=prob.get_val('pyduct.design.inlet.Fl_O:stat:area', units='inch**2'), units='inch**2')
            prob.set_val('iv.area_2', value=prob.get_val('pyduct.design.duct.Fl_O:stat:area', units='inch**2'), units='inch**2')
            prob.set_val('oc.sta3.heat_in', value=heat_in, units='kW')
            if oc_use_dpqp:
                prob.set_val('oc.sta3.pressure_recovery', value=(1-dPqP), units=None)
            else:
                delta_p = prob.get_val('pyduct.design.inlet.Fl_O:tot:P', units='Pa') - prob.get_val('pyduct.design.nozz.Fl_O:tot:P', units='Pa')
                prob.set_val('oc.sta3.delta_p', -delta_p, units='Pa')

            prob.run_model()
            if list_output:
                prob.model.list_outputs(units=True, excludes=['*chem_eq*','*props*'])
                # prob.model.list_outputs(includes=['*oc.*force*','*perf*','*mdot*'], units=True)
                print(prob.get_val('pyduct.design.inlet.Fl_O:stat:W', units='kg/s'))
                print(prob.get_val('pyduct.design.perf.Fn', units='N'))

                for pt in ['design']+mp_duct.od_pts:
                    print('\n', '#'*10, pt, '#'*10)
                    viewer(prob, 'pyduct.'+pt)

            return prob

        def print_truth(name, value, list_output=True):
            if list_output:
                print(name+': '+str(value))

        def check_params_match(prob, list_output=True):
            mdot_pyc = prob.get_val('pyduct.design.fc.Fl_O:stat:W', units='kg/s')
            mdot_oc = prob.get_val('oc.mdot', units='kg/s')
            assert_near_equal(mdot_oc, mdot_pyc, tolerance=1e-4)
            print_truth('mass flow', mdot_pyc, list_output)

            fnet_pyc = prob.get_val('pyduct.design.perf.Fn', units='N')
            fnet_oc = prob.get_val('oc.force.F_net', units='N')
            assert_near_equal(fnet_pyc, fnet_pyc, tolerance=1e-4)
            print_truth('net force', fnet_pyc, list_output)

            # compare the flow conditions at each station
            oc_stations = ['inlet','sta1','sta3','nozzle']
            state_units = ['K','Pa','kg/m**3',None,'m/s','K','Pa']
            for i, pyc_station in enumerate(['fc','inlet', 'duct', 'nozz']):
                oc_station = oc_stations[i]
                if list_output:
                    print('--------'+pyc_station+'-------------')
                if oc_station == 'nozzle' or oc_station == 'inlet':
                    oc_states = ['T','p','rho','M','a','Tt','pt']
                else:
                    oc_states = ['T','p','rho','M','a','Tt_out','pt_out']
                for j, pyc_state in enumerate(['stat:T','stat:P','stat:rho','stat:MN','stat:Vsonic','tot:T','tot:P']):
                    oc_state = oc_states[j]
                    if oc_station == 'inlet' and oc_state == 'rho':
                        continue
                    state_pyc = prob.get_val('pyduct.design.'+pyc_station+'.Fl_O:'+pyc_state, units=state_units[j])
                    state_oc = prob.get_val('oc.'+oc_station+'.'+oc_state, units=state_units[j])
                    assert_near_equal(state_oc, state_pyc, tolerance=5e-4)
                    print_truth(oc_state, state_pyc, list_output)
    else:
        HAS_PYCYCLE = False
    

except:
    HAS_PYCYCLE = False

if not HAS_PYCYCLE:
    # TODO define run_problem without pycycle
    pass
else:
    class TestOCDuct(unittest.TestCase):
        def test_baseline(self):
            prob = run_problem(dPqP=0.0, heat_in=0.0, oc_use_dpqp=False, list_output=False)
            check_params_match(prob, list_output=False)
        
        def test_heat_addition(self):
            prob = run_problem(dPqP=0.0, heat_in=5.0, oc_use_dpqp=False, list_output=False)
            check_params_match(prob, list_output=False)

        def test_delta_p(self):
            prob = run_problem(dPqP=0.05, heat_in=5.0, oc_use_dpqp=False, list_output=False)
            check_params_match(prob, list_output=False)

        def test_dpqp(self):
            prob = run_problem(dPqP=0.05, heat_in=5.0, oc_use_dpqp=True, list_output=False)
            check_params_match(prob, list_output=False)

        def test_cfg(self):
            prob = run_problem(dPqP=0.05, heat_in=5.0, oc_use_dpqp=True, cfg=0.95, list_output=False)
            check_params_match(prob, list_output=False)

if __name__ == "__main__":
    if HAS_PYCYCLE:
        # run_problem(dPqP=0.05, oc_use_dpqp=True)
        unittest.main()
    else:
        warnings.warn('pycycle >= 3.0 must be installed to run reg tests using pycycle. Skipping compressible duct tests')