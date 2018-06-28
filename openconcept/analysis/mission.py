from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, DirectSolver, NewtonSolver, ScipyKrylov
import numpy as np
import scipy.sparse as sp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.math.simpson_integration import simpson_integral, simpson_partials, simpson_integral_every_node, simpson_partials_every_node
from openconcept.utilities.math.sum_comp import SumComp
from openconcept.analysis.aerodynamics import PolarDrag
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent, BalanceComp, ArmijoGoldsteinLS, NonlinearBlockGS
from openconcept.utilities.dvlabel import DVLabel

class ComputeDesignMissionResiduals(ExplicitComponent):
    def initialize(self):
        self.options.declare('include_takeoff',default=False,desc='Flag yes if you want to include takeoff fuel')
    def setup(self):
        include_takeoff = self.options['include_takeoff']
        self.add_input('ac:weights:MTOW',val=2000,units='kg')
        self.add_input('mission:total_fuel',val=180,units='kg')

        self.add_input('OEW',val=1500,units='kg')
        self.add_input('mission:payload',val=200,units='kg')
        self.add_input('ac:weights:W_fuel_max',val=400,units='kg')
        self.add_output('mission:fuel_capacity_margin',units='kg')
        self.add_output('mission:MTOW_margin',units='kg')
        self.declare_partials('mission:fuel_capacity_margin','mission:total_fuel',val=-1)
        self.declare_partials('mission:fuel_capacity_margin','ac:weights:W_fuel_max' , val=1)
        self.declare_partials('mission:MTOW_margin','ac:weights:MTOW',val=1)
        self.declare_partials(['mission:MTOW_margin'],['mission:total_fuel','OEW','mission:payload'],val=-1)
        if include_takeoff:
            self.add_input('takeoff:total_fuel',val=1,units='kg')
            self.add_output('fuel_burn',units='kg')
            self.declare_partials('mission:MTOW_margin','takeoff:total_fuel',val=-1)
            self.declare_partials('fuel_burn',['takeoff:total_fuel','mission:total_fuel'],val=1)
            self.declare_partials('mission:fuel_capacity_margin','takeoff:total_fuel',val=-1)

    def compute(self, inputs, outputs):
        include_takeoff = self.options['include_takeoff']
        if include_takeoff:
            outputs['mission:fuel_capacity_margin'] = inputs['ac:weights:W_fuel_max']-inputs['mission:total_fuel'] - inputs['takeoff:total_fuel']
            outputs['mission:MTOW_margin'] = inputs['ac:weights:MTOW'] - inputs['mission:total_fuel'] - inputs['takeoff:total_fuel'] - inputs['OEW'] - inputs['mission:payload']
            outputs['fuel_burn'] = inputs['mission:total_fuel'] + inputs['takeoff:total_fuel']
            if inputs['mission:total_fuel'] < -1e-4 or inputs['takeoff:total_fuel'] < -1e-4:
                raise ValueError('You have negative total fuel flows for some flight phase')
        else:
            outputs['mission:fuel_capacity_margin'] = inputs['ac:weights:W_fuel_max']-inputs['mission:total_fuel']
            outputs['mission:MTOW_margin'] = inputs['ac:weights:MTOW'] - inputs['mission:total_fuel'] - inputs['OEW'] - inputs['mission:payload']

class ComputeDesignMissionResidualsBattery(ComputeDesignMissionResiduals):
    def setup(self):
        super(ComputeDesignMissionResidualsBattery,self).setup()
        include_takeoff = self.options['include_takeoff']
        self.add_input('ac:weights:W_battery',val=0,units='kg')
        self.add_input('battery_max_energy',val=1,units='MJ')
        self.add_input('mission:total_battery_energy',val=0,units='MJ')
        self.add_output('mission:battery_margin',units='MJ')
        self.declare_partials('mission:MTOW_margin','ac:weights:W_battery',val=-1)
        self.declare_partials('mission:battery_margin','battery_max_energy',val=1)
        self.declare_partials('mission:battery_margin','mission:total_battery_energy',val=-1)
        if include_takeoff:
            self.add_input('takeoff:total_battery_energy',val=0,units='MJ')
            self.add_output('battery_energy_used',units='MJ')
            self.declare_partials('battery_energy_used',['mission:total_battery_energy','takeoff:total_battery_energy'],val=1)
            self.declare_partials('mission:battery_margin','takeoff:total_battery_energy',val=-1)
    def compute(self, inputs, outputs):
        super(ComputeDesignMissionResidualsBattery,self).compute(inputs, outputs)
        include_takeoff = self.options['include_takeoff']
        if include_takeoff:
            outputs['battery_energy_used'] = inputs['mission:total_battery_energy'] + inputs['takeoff:total_battery_energy']
            outputs['mission:battery_margin'] = inputs['battery_max_energy'] - inputs['mission:total_battery_energy'] - inputs['takeoff:total_battery_energy']
            outputs['mission:MTOW_margin'] = inputs['ac:weights:MTOW'] - inputs['mission:total_fuel'] - inputs['takeoff:total_fuel'] - inputs['ac:weights:W_battery'] - inputs['OEW'] - inputs['mission:payload']

        else:
            outputs['mission:battery_margin'] = inputs['battery_max_energy'] - inputs['mission:total_battery_energy']
            outputs['mission:MTOW_margin'] = inputs['ac:weights:MTOW'] - inputs['mission:total_fuel'] -inputs['ac:weights:W_battery'] - inputs['OEW'] - inputs['mission:payload']
        #print('Battery margin:'+str(outputs['mission:battery_margin']))


class MissionFlightConditions(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        self.add_input('mission:climb:vs', val=5, units='m / s', desc='Vertical speed in the climb segment')
        self.add_input('mission:descent:vs', val=-2.5, units='m / s', desc='Vertical speed in the descent segment (should be neg)')
        self.add_input('mission:climb:Ueas', val=90, units='m / s', desc='Indicated airspeed during climb')
        self.add_input('mission:cruise:Ueas', val=100, units='m / s', desc='Cruise airspeed (indicated)')
        self.add_input('mission:descent:Ueas', val=80, units='m / s', desc='Descent airspeed (indicated)')
        self.add_input('mission:takeoff:h', val=0, units='m',desc='Airport altitude')
        self.add_input('mission:cruise:h', val=8000, units='m', desc='Cruise altitude')

        self.add_output('fltcond:mission:Ueas', units='m / s', desc='indicated airspeed at each timepoint',shape=(3*nn,))
        self.add_output('fltcond:mission:h', units='m', desc='altitude at each timepoint',shape=(3*nn,))
        self.add_output('fltcond:mission:vs', units='m / s', desc='vectorial representation of vertical speed',shape=(3*nn,))
        self.add_output('mission:climb:time', units ='s', desc='Time from ground level to cruise')
        self.add_output('mission:descent:time', units='s', desc='Time to descend to ground from cruise')
        self.add_output('mission:climb:dt',units='s', desc='Timestep in climb phase')
        self.add_output('mission:descent:dt', units='s', desc='Timestep in descent phase')

        #the climb speeds only have influence over their respective mission segments
        self.declare_partials(['fltcond:mission:Ueas'],['mission:climb:Ueas'],rows=np.arange(0,nn),cols=np.ones(nn)*0,val=np.ones(nn))
        self.declare_partials(['fltcond:mission:Ueas'],['mission:cruise:Ueas'],rows=np.arange(nn,2*nn),cols=np.ones(nn)*0,val=np.ones(nn))
        self.declare_partials(['fltcond:mission:Ueas'],['mission:descent:Ueas'],rows=np.arange(2*nn,3*nn),cols=np.ones(nn)*0,val=np.ones(nn))
        hcruisepartials = np.concatenate([np.linspace(0.0,1.0,nn),np.ones(nn),np.linspace(1.0,0.0,nn)])
        hgroundpartials = np.concatenate([np.linspace(1.0,0.0,nn),np.linspace(0.0,1.0,nn)])
        #the influence of each parameter linearly varies from 0 to 1 and vice versa on climb and descent. The partials are different lengths on purpose - no influence of ground on the mid-mission points so no partial derivative
        self.declare_partials(['fltcond:mission:h'],['mission:cruise:h'],rows=range(3*nn),cols=np.zeros(3*nn),val=hcruisepartials)
        self.declare_partials(['fltcond:mission:h'],['mission:takeoff:h'],rows=np.concatenate([np.arange(0,nn),np.arange(2*nn,3*nn)]),cols=np.zeros(2*nn),val=hgroundpartials)
        self.declare_partials(['fltcond:mission:vs'],['mission:climb:vs'],rows=range(nn),cols=np.zeros(nn),val=np.ones(nn))
        self.declare_partials(['fltcond:mission:vs'],['mission:descent:vs'],rows=np.arange(2*nn,3*nn),cols=np.zeros(nn),val=np.ones(nn))
        self.declare_partials(['mission:climb:time'],['mission:takeoff:h','mission:cruise:h','mission:climb:vs'])
        self.declare_partials(['mission:descent:time'],['mission:takeoff:h','mission:cruise:h','mission:descent:vs'])
        self.declare_partials(['mission:climb:dt'],['mission:takeoff:h','mission:cruise:h','mission:climb:vs'])
        self.declare_partials(['mission:descent:dt'],['mission:takeoff:h','mission:cruise:h','mission:descent:vs'])



    def compute(self,inputs,outputs):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = n_int_per_seg*2 + 1
        hvec_climb = np.linspace(inputs['mission:takeoff:h'],inputs['mission:cruise:h'],nn)
        hvec_desc = np.linspace(inputs['mission:cruise:h'],inputs['mission:takeoff:h'],nn)
        hvec_cruise = np.ones(nn)*inputs['mission:cruise:h']
        outputs['fltcond:mission:h'] = np.concatenate([hvec_climb,hvec_cruise,hvec_desc])
        debug = np.concatenate([np.ones(nn)*inputs['mission:climb:Ueas'],np.ones(nn)*inputs['mission:cruise:Ueas'],np.ones(nn)*inputs['mission:descent:Ueas']])
        outputs['fltcond:mission:Ueas'] = np.concatenate([np.ones(nn)*inputs['mission:climb:Ueas'],np.ones(nn)*inputs['mission:cruise:Ueas'],np.ones(nn)*inputs['mission:descent:Ueas']])
        outputs['fltcond:mission:vs'] = np.concatenate([np.ones(nn)*inputs['mission:climb:vs'],np.ones(nn)*0.0,np.ones(nn)*inputs['mission:descent:vs']])
        outputs['mission:climb:time'] = (inputs['mission:cruise:h']-inputs['mission:takeoff:h'])/inputs['mission:climb:vs']
        outputs['mission:descent:time'] = (inputs['mission:takeoff:h']-inputs['mission:cruise:h'])/inputs['mission:descent:vs']
        outputs['mission:climb:dt'] = (inputs['mission:cruise:h']-inputs['mission:takeoff:h'])/inputs['mission:climb:vs']/(nn-1)
        outputs['mission:descent:dt'] =  (inputs['mission:takeoff:h']-inputs['mission:cruise:h'])/inputs['mission:descent:vs']/(nn-1)

    def compute_partials(self, inputs, J):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = n_int_per_seg*2 + 1
        J['mission:climb:time','mission:cruise:h'] = 1/inputs['mission:climb:vs']
        J['mission:climb:time','mission:takeoff:h'] = -1/inputs['mission:climb:vs']
        J['mission:climb:time','mission:climb:vs'] = -(inputs['mission:cruise:h']-inputs['mission:takeoff:h'])/(inputs['mission:climb:vs']**2)
        J['mission:descent:time','mission:cruise:h'] = -1/inputs['mission:descent:vs']
        J['mission:descent:time','mission:takeoff:h'] = 1/inputs['mission:descent:vs']
        J['mission:descent:time','mission:descent:vs'] = -(inputs['mission:takeoff:h']-inputs['mission:cruise:h'])/(inputs['mission:descent:vs']**2)
        J['mission:climb:dt','mission:cruise:h'] = 1/inputs['mission:climb:vs']/(nn-1)
        J['mission:climb:dt','mission:takeoff:h'] = -1/inputs['mission:climb:vs']/(nn-1)
        J['mission:climb:dt','mission:climb:vs'] = -(inputs['mission:cruise:h']-inputs['mission:takeoff:h'])/(inputs['mission:climb:vs']**2)/(nn-1)
        J['mission:descent:dt','mission:cruise:h'] = -1/inputs['mission:descent:vs']/(nn-1)
        J['mission:descent:dt','mission:takeoff:h'] = 1/inputs['mission:descent:vs']/(nn-1)
        J['mission:descent:dt','mission:descent:vs'] = -(inputs['mission:takeoff:h']-inputs['mission:cruise:h'])/(inputs['mission:descent:vs']**2)/(nn-1)

class MissionGroundspeeds(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        self.add_input('fltcond:mission:vs', units='m/s',shape=(3*nn,))
        self.add_input('fltcond:mission:Utrue', units='m/s',shape=(3*nn,))
        self.add_output('mission:groundspeed', units='m/s',shape=(3*nn,))
        self.add_output('fltcond:mission:cosgamma', shape=(3*nn,), desc='Cosine of the flight path angle')
        self.add_output('fltcond:mission:singamma', shape=(3*nn,), desc='sin of the flight path angle' )
        self.declare_partials(['mission:groundspeed','fltcond:mission:cosgamma','fltcond:mission:singamma'],['fltcond:mission:vs','fltcond:mission:Utrue'],rows=range(3*nn),cols=range(3*nn))

    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #compute the groundspeed on climb and desc
        groundspeed =  np.sqrt(inputs['fltcond:mission:Utrue']**2-inputs['fltcond:mission:vs']**2)
        outputs['mission:groundspeed'] = groundspeed
        outputs['fltcond:mission:singamma'] = inputs['fltcond:mission:vs'] / inputs['fltcond:mission:Utrue']
        outputs['fltcond:mission:cosgamma'] = groundspeed / inputs['fltcond:mission:Utrue']

    def compute_partials(self, inputs, J):
        groundspeed =  np.sqrt(inputs['fltcond:mission:Utrue']**2-inputs['fltcond:mission:vs']**2)
        J['mission:groundspeed','fltcond:mission:vs'] = (1/2) / np.sqrt(inputs['fltcond:mission:Utrue']**2-inputs['fltcond:mission:vs']**2) * (-2) * inputs['fltcond:mission:vs']
        J['mission:groundspeed','fltcond:mission:Utrue'] = (1/2) / np.sqrt(inputs['fltcond:mission:Utrue']**2-inputs['fltcond:mission:vs']**2) * 2 * inputs['fltcond:mission:Utrue']
        J['fltcond:mission:singamma','fltcond:mission:vs'] = 1 / inputs['fltcond:mission:Utrue']
        J['fltcond:mission:singamma','fltcond:mission:Utrue'] = - inputs['fltcond:mission:vs'] / inputs['fltcond:mission:Utrue'] ** 2
        J['fltcond:mission:cosgamma','fltcond:mission:vs'] = J['mission:groundspeed','fltcond:mission:vs'] / inputs['fltcond:mission:Utrue']
        J['fltcond:mission:cosgamma','fltcond:mission:Utrue'] = (J['mission:groundspeed','fltcond:mission:Utrue'] * inputs['fltcond:mission:Utrue'] - groundspeed) / inputs['fltcond:mission:Utrue']**2

class MissionClimbDescentRanges(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        self.add_input('mission:groundspeed', units='m/s',shape=(3*nn,))
        self.add_input('mission:climb:time', units='s')
        self.add_input('mission:descent:time', units='s')
        self.add_output('mission:descent:range',units='m')
        self.add_output('mission:climb:range',units='m')
        self.declare_partials(['mission:climb:range'],['mission:groundspeed'],rows=np.ones(nn)*0,cols=range(nn))
        self.declare_partials(['mission:descent:range'],['mission:groundspeed'],rows=np.ones(nn)*0,cols=np.arange(2*nn,3*nn))
        self.declare_partials(['mission:climb:range'],['mission:climb:time'])
        self.declare_partials(['mission:descent:range'],['mission:descent:time'])

    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)

        groundspeed = inputs['mission:groundspeed']
        #compute distance traveled during climb and desc using Simpson's rule
        dt_climb = inputs['mission:climb:time'] / (nn-1)
        dt_desc = inputs['mission:descent:time'] / (nn-1)
        simpsons_vec = np.ones(nn)
        simpsons_vec[1:nn-1:2] = 4
        simpsons_vec[2:nn-1:2] = 2
        outputs['mission:climb:range'] = np.sum(simpsons_vec*groundspeed[0:nn])*dt_climb/3
        outputs['mission:descent:range'] = np.sum(simpsons_vec*groundspeed[2*nn:3*nn])*dt_desc/3

    def compute_partials(self, inputs, J):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)

        groundspeed = inputs['mission:groundspeed']
        simpsons_vec = np.ones(nn)
        simpsons_vec[1:nn-1:2] = 4
        simpsons_vec[2:nn-1:2] = 2

        J['mission:climb:range','mission:climb:time'] = np.sum(simpsons_vec*groundspeed[0:nn])/3/(nn-1)
        J['mission:descent:range','mission:descent:time'] = np.sum(simpsons_vec*groundspeed[2*nn:3*nn])/3/(nn-1)
        J['mission:climb:range','mission:groundspeed'] = simpsons_vec * inputs['mission:climb:time'] / (nn-1) / 3
        J['mission:descent:range','mission:groundspeed'] = simpsons_vec * inputs['mission:descent:time'] / (nn-1) / 3

class MissionTimings(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)

        self.add_input('mission:groundspeed',units='m/s',shape=(3*nn,))
        self.add_input('mission:climb:range', units='m')
        self.add_input('mission:descent:range', units='m')
        self.add_input('mission:range',units='m')
        self.add_output('mission:cruise:range',units='m')
        self.add_output('mission:cruise:time',units='s')
        self.add_output('mission:cruise:dt',units="s")
        self.declare_partials(['mission:cruise:range'],['mission:climb:range'],val=-1.0)
        self.declare_partials(['mission:cruise:range'],['mission:descent:range'],val=-1.0)
        self.declare_partials(['mission:cruise:range'],['mission:range'],val=1.0)
        self.declare_partials(['mission:cruise:time'],['mission:groundspeed'],rows=np.zeros(nn),cols=np.arange(nn,2*nn))
        self.declare_partials(['mission:cruise:time'],['mission:climb:range','mission:descent:range','mission:range'])
        self.declare_partials(['mission:cruise:dt'],['mission:groundspeed'],rows=np.zeros(nn),cols=np.arange(nn,2*nn))
        self.declare_partials(['mission:cruise:dt'],['mission:climb:range','mission:descent:range','mission:range'])

    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)

        simpsons_vec = np.ones(nn)
        simpsons_vec[1:nn-1:2] = 4
        simpsons_vec[2:nn-1:2] = 2

        #compute the cruise distance
        r_cruise = inputs['mission:range'] - inputs['mission:climb:range'] - inputs['mission:descent:range']
        if r_cruise < 0:
            raise ValueError('Cruise calculated to be less than 0. Change climb and descent rates and airspeeds or increase range')
        dt_cruise = 3*r_cruise/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])
        t_cruise = dt_cruise*(nn-1)

        outputs['mission:cruise:time'] = t_cruise
        outputs['mission:cruise:range'] = r_cruise
        outputs['mission:cruise:dt'] = dt_cruise

    def compute_partials(self, inputs, J):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)

        simpsons_vec = np.ones(nn)
        simpsons_vec[1:nn-1:2] = 4
        simpsons_vec[2:nn-1:2] = 2

        #compute the cruise distance
        r_cruise = inputs['mission:range'] - inputs['mission:climb:range'] - inputs['mission:descent:range']
        J['mission:cruise:time','mission:groundspeed'] = -3*r_cruise/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])**2 * (nn-1) * (simpsons_vec)
        J['mission:cruise:time','mission:climb:range'] = -3/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])*(nn-1)
        J['mission:cruise:time','mission:descent:range'] = -3/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])*(nn-1)
        J['mission:cruise:time','mission:range'] = 3/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])*(nn-1)

        J['mission:cruise:dt','mission:groundspeed'] = -3*r_cruise/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])**2  * (simpsons_vec)
        J['mission:cruise:dt','mission:climb:range'] = -3/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])
        J['mission:cruise:dt','mission:descent:range'] = -3/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])
        J['mission:cruise:dt','mission:range'] = 3/np.sum(simpsons_vec*inputs['mission:groundspeed'][nn:2*nn])

class MissionSegmentFuelBurns(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")


    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)

        self.add_input('mission:fuel_flow',units='kg/s',shape=(3*nn,))
        self.add_input('mission:climb:dt', units='s')
        self.add_input('mission:descent:dt', units='s')
        self.add_input('mission:cruise:dt', units='s')

        self.add_output('mission:segment_fuel',units='kg',shape=(3*(nn-1)))
        #use dummy inputs for dt and q, just want the shapes
        wrt_q, wrt_dt = simpson_partials_every_node(np.ones(3),np.ones(3*nn),n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        self.declare_partials(['mission:segment_fuel'],['mission:fuel_flow'],rows=wrt_q[0],cols=wrt_q[1])
        self.declare_partials(['mission:segment_fuel'],['mission:climb:dt'],rows=wrt_dt[0][0],cols=wrt_dt[1][0])
        self.declare_partials(['mission:segment_fuel'],['mission:cruise:dt'],rows=wrt_dt[0][1],cols=wrt_dt[1][1])
        self.declare_partials(['mission:segment_fuel'],['mission:descent:dt'],rows=wrt_dt[0][2],cols=wrt_dt[1][2])

    def compute(self,inputs,outputs):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        ff = inputs['mission:fuel_flow']
        dts =  [inputs['mission:climb:dt'], inputs['mission:cruise:dt'],inputs['mission:descent:dt']]
        int_ff, delta_ff = simpson_integral_every_node(dts,ff,n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        outputs['mission:segment_fuel'] = delta_ff

    def compute_partials(self,inputs,J):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        ff = inputs['mission:fuel_flow']
        dts =  [inputs['mission:climb:dt'], inputs['mission:cruise:dt'],inputs['mission:descent:dt']]

        wrt_q, wrt_dt = simpson_partials_every_node(dts,ff,n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        J['mission:segment_fuel','mission:fuel_flow'] = wrt_q[2]
        J['mission:segment_fuel','mission:climb:dt'] =  wrt_dt[2][0]
        J['mission:segment_fuel','mission:cruise:dt'] = wrt_dt[2][1]
        J['mission:segment_fuel','mission:descent:dt'] = wrt_dt[2][2]

class MissionSegmentBatteryEnergyUsed(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)

        self.add_input('mission:battery_load',units='kW',shape=(3*nn,))
        self.add_input('mission:climb:dt', units='s')
        self.add_input('mission:descent:dt', units='s')
        self.add_input('mission:cruise:dt', units='s')

        self.add_output('mission:segment_battery_energy_used',units='kW*s',shape=(3*(nn-1)))
        #use dummy inputs for dt and q, just want the shapes
        wrt_q, wrt_dt = simpson_partials_every_node(np.ones(3),np.ones(3*nn),n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        self.declare_partials(['mission:segment_battery_energy_used'],['mission:battery_load'],rows=wrt_q[0],cols=wrt_q[1])
        self.declare_partials(['mission:segment_battery_energy_used'],['mission:climb:dt'],rows=wrt_dt[0][0],cols=wrt_dt[1][0])
        self.declare_partials(['mission:segment_battery_energy_used'],['mission:cruise:dt'],rows=wrt_dt[0][1],cols=wrt_dt[1][1])
        self.declare_partials(['mission:segment_battery_energy_used'],['mission:descent:dt'],rows=wrt_dt[0][2],cols=wrt_dt[1][2])

    def compute(self,inputs,outputs):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        ff = inputs['mission:battery_load']
        dts =  [inputs['mission:climb:dt'], inputs['mission:cruise:dt'],inputs['mission:descent:dt']]
        int_ff, delta_ff = simpson_integral_every_node(dts,ff,n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        outputs['mission:segment_battery_energy_used'] = delta_ff

    def compute_partials(self,inputs,J):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        ff = inputs['mission:battery_load']
        dts =  [inputs['mission:climb:dt'], inputs['mission:cruise:dt'],inputs['mission:descent:dt']]

        wrt_q, wrt_dt = simpson_partials_every_node(dts,ff,n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        J['mission:segment_battery_energy_used','mission:battery_load'] = wrt_q[2]
        J['mission:segment_battery_energy_used','mission:climb:dt'] =  wrt_dt[2][0]
        J['mission:segment_battery_energy_used','mission:cruise:dt'] = wrt_dt[2][1]
        J['mission:segment_battery_energy_used','mission:descent:dt'] = wrt_dt[2][2]


class MissionSegmentWeights(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        self.add_input('mission:segment_fuel',units='kg',shape=(3*(nn-1),))
        self.add_input('mission:weight_initial',units='kg')
        self.add_output('mission:weights',units='kg',shape=(3*nn,))

        n_seg = 3
        jacmat = np.tril(np.ones((n_seg*(nn-1),n_seg*(nn-1))))
        jacmat = np.insert(jacmat,0,np.zeros(n_seg*(nn-1)),axis=0)
        for i in range(1,n_seg):
            duplicate_row = jacmat[nn*i-1,:]
            jacmat = np.insert(jacmat,nn*i,duplicate_row,axis=0)

        self.declare_partials(['mission:weights'],['mission:segment_fuel'],val=sp.csr_matrix(jacmat))
        self.declare_partials(['mission:weights'],['mission:weight_initial'],rows=range(3*nn),cols=np.zeros(3*nn),val=np.ones(3*nn))

    def compute(self,inputs,outputs):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #first endpoint needs to be the takeoff weight; insert a zero to make them the same length
        n_seg = 3
        segweights = np.insert(inputs['mission:segment_fuel'],0,0)
        weights = np.cumsum(segweights)
        for i in range(1,n_seg):
            duplicate_row = weights[i*nn-1]
            weights = np.insert(weights,i*nn,duplicate_row)
        outputs['mission:weights'] = np.ones(3*nn)*inputs['mission:weight_initial'] + weights

class MissionSegmentCL(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        n_seg = 3
        arange = np.arange(0,n_seg*nn)
        self.add_input('mission:weights',units='kg', shape=(n_seg*nn,))
        self.add_input('fltcond:mission:q',units='N * m**-2', shape=(n_seg*nn,))
        self.add_input('ac:geom:wing:S_ref',units='m **2')
        self.add_input('fltcond:mission:cosgamma', shape=(n_seg*nn,))
        self.add_output('fltcond:mission:CL',shape=(n_seg*nn,))


        self.declare_partials(['fltcond:mission:CL'],['mission:weights','fltcond:mission:q',"fltcond:mission:cosgamma"],rows=arange,cols=arange)
        self.declare_partials(['fltcond:mission:CL'],['ac:geom:wing:S_ref'],rows=arange,cols=np.zeros(n_seg*nn))

    def compute(self,inputs,outputs):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #first endpoint needs to be the takeoff weight; insert a zero to make them the same length
        n_seg = 3

        g = 9.80665 #m/s^2
        outputs['fltcond:mission:CL'] = inputs['fltcond:mission:cosgamma']*g*inputs['mission:weights']/inputs['fltcond:mission:q']/inputs['ac:geom:wing:S_ref']

    def compute_partials(self,inputs,J):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #first endpoint needs to be the takeoff weight; insert a zero to make them the same length
        n_seg = 3

        g = 9.80665 #m/s^2
        J['fltcond:mission:CL','mission:weights'] = inputs['fltcond:mission:cosgamma']*g/inputs['fltcond:mission:q']/inputs['ac:geom:wing:S_ref']
        J['fltcond:mission:CL','fltcond:mission:q'] = - inputs['fltcond:mission:cosgamma']*g*inputs['mission:weights'] / inputs['fltcond:mission:q']**2 / inputs['ac:geom:wing:S_ref']
        J['fltcond:mission:CL','ac:geom:wing:S_ref'] = - inputs['fltcond:mission:cosgamma']*g*inputs['mission:weights'] / inputs['fltcond:mission:q'] / inputs['ac:geom:wing:S_ref']**2
        J['fltcond:mission:CL','fltcond:mission:cosgamma'] = g*inputs['mission:weights']/inputs['fltcond:mission:q']/inputs['ac:geom:wing:S_ref']


class ThrustResidual(ImplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        n_seg = 3
        arange = np.arange(0,n_seg*nn)
        self.add_input('drag', units='N',shape=(n_seg*nn,))
        self.add_input('fltcond:mission:singamma', shape=(n_seg*nn,))
        self.add_input('mission:weights',units='kg', shape=(n_seg*nn,))
        self.add_input('mission:thrust', units='N', shape=(n_seg*nn,))
        self.add_output('throttle', shape=(n_seg*nn,),lower=np.zeros(n_seg*nn),upper=np.ones(n_seg*nn)*2)
        self.declare_partials(['throttle'],['drag'],val=-sp.eye(nn*n_seg))
        self.declare_partials(['throttle'],['mission:thrust'],val=sp.eye(nn*n_seg))
        self.declare_partials(['throttle'], ['fltcond:mission:singamma','mission:weights'],rows=arange,cols=arange)

    def apply_nonlinear(self, inputs, outputs, residuals):
        g = 9.80665 #m/s^2
        debug_nonlinear = False
        if debug_nonlinear:
            print('Thrust: ' + str(inputs['mission:thrust']))
            print('Drag: '+ str(inputs['drag']))
            print('mgsingamma: ' + str(inputs['mission:weights']*g*inputs['fltcond:mission:singamma']))
            print('Throttle: ' + str(outputs['throttle']))

        residuals['throttle'] = inputs['mission:thrust'] - inputs['drag'] - inputs['mission:weights']*g*inputs['fltcond:mission:singamma']


    def linearize(self, inputs, outputs, partials):
        g = 9.80665 #m/s^2
        partials['throttle','mission:weights'] = -g*inputs['fltcond:mission:singamma']
        partials['throttle','fltcond:mission:singamma'] = -g*inputs['mission:weights']

class ExplicitThrustResidual(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        n_seg = 3
        arange = np.arange(0,n_seg*nn)
        self.add_input('drag', units='N',shape=(n_seg*nn,))
        self.add_input('fltcond:mission:singamma', shape=(n_seg*nn,))
        self.add_input('mission:weights',units='kg', shape=(n_seg*nn,))
        self.add_input('mission:thrust', units='N', shape=(n_seg*nn,))
        self.add_output('thrust_residual', shape=(n_seg*nn,), units='N')
        self.declare_partials(['thrust_residual'],['drag'],rows=arange,cols=arange,val=-np.ones(nn*n_seg))
        self.declare_partials(['thrust_residual'],['mission:thrust'],rows=arange,cols=arange,val=np.ones(nn*n_seg))
        self.declare_partials(['thrust_residual'], ['fltcond:mission:singamma','mission:weights'],rows=arange,cols=arange)

    def compute(self, inputs, outputs):
        g = 9.80665 #m/s^2
        debug_nonlinear = False
        if debug_nonlinear:
            print('Thrust: ' + str(inputs['mission:thrust']))
            print('Drag: '+ str(inputs['drag']))
            print('mgsingamma: ' + str(inputs['mission:weights']*g*inputs['fltcond:mission:singamma']))
            print('Throttle: ' + str(outputs['throttle']))

        outputs['thrust_residual'] = inputs['mission:thrust'] - inputs['drag'] - inputs['mission:weights']*g*inputs['fltcond:mission:singamma']


    def compute_partials(self, inputs, J):
        g = 9.80665 #m/s^2
        J['thrust_residual','mission:weights'] = -g*inputs['fltcond:mission:singamma']
        J['thrust_residual','fltcond:mission:singamma'] = -g*inputs['mission:weights']

class MissionNoReserves(Group):
    """This analysis group calculates fuel burns, weights, and segment times

    """

    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")
        self.options.declare('track_battery',default=False, desc="Flip to true if you want to track battery state")
        self.options.declare('use_newton',default=False,desc="Use newton solver, i.e. use an implicit component to drive the throttle settings")
    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn_tot = (2*n_int_per_seg+1)*3 #climb, cruise, descent
        #Create holders for control and flight condition parameters
        track_battery = self.options['track_battery']
        use_newton = self.options['use_newton']

        dvlist = [['fltcond:mission:q','fltcond:q',100*np.ones(nn_tot),'Pa'],
                  ['ac:aero:polar:CD0_cruise','CD0',0.005,None],
                  ['ac:aero:polar:e','e',0.95,None]]
        self.add_subsystem('dvs',DVLabel(dvlist),promotes_inputs=["*"],promotes_outputs=["*"])

        groundspeeds = self.add_subsystem('gs',MissionGroundspeeds(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["fltcond:*"],promotes_outputs=["mission:groundspeed","fltcond:*"])
        ranges = self.add_subsystem('ranges',MissionClimbDescentRanges(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["mission:groundspeed","mission:*time"],promotes_outputs=["mission:climb:range","mission:descent:range"])
        timings = self.add_subsystem('timings',MissionTimings(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["mission:*range","mission:groundspeed"],promotes_outputs=["mission:cruise:range","mission:cruise:time","mission:cruise:dt"])

        fbs = self.add_subsystem('fuelburn',MissionSegmentFuelBurns(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["mission:fuel_flow","mission:*dt"],promotes_outputs=["mission:segment_fuel"])
        if track_battery:
            energy = self.add_subsystem('battery',MissionSegmentBatteryEnergyUsed(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["mission:battery_load",'mission*dt'],promotes_outputs=["mission:segment_battery_energy_used"])
        wts = self.add_subsystem('weights',MissionSegmentWeights(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["mission:segment_fuel","mission:weight_initial"],promotes_outputs=["mission:weights"])
        CLs = self.add_subsystem('CLs',MissionSegmentCL(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["mission:weights","fltcond:mission:q",'fltcond:mission:cosgamma',"ac:geom:*"],promotes_outputs=["fltcond:mission:CL"])
        drag = self.add_subsystem('drag',PolarDrag(num_nodes=nn_tot),promotes_inputs=["fltcond:q","ac:geom:*","CD0","e"],promotes_outputs=["drag"])
        self.connect('fltcond:mission:CL','drag.fltcond:CL')
        if use_newton:
            td = self.add_subsystem('thrust',ThrustResidual(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["fltcond:mission:singamma","drag","mission:weights*","mission:thrust"])
        else:
            td = self.add_subsystem('thrust_resid',ExplicitThrustResidual(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["fltcond:mission:singamma","drag","mission:weights*","mission:thrust"])
        totals = SumComp(axis=None)
        totals.add_equation(output_name='mission:total_fuel',input_name='mission:segment_fuel',units='kg',scaling_factor=-1,vec_size=nn_tot-3)
        if track_battery:
            totals.add_equation(output_name='mission:total_battery_energy',input_name='mission:segment_battery_energy_used',units='MJ',vec_size=nn_tot-3)
        self.add_subsystem(name='totals',subsys=totals,promotes_inputs=['*'],promotes_outputs=['*'])


if __name__ == "__main__":

    from openconcept.analysis.mission import MissionAnalysisTest
    from openconcept.examples.simple_turboprop import TurbopropPropulsionSystem

    import matplotlib.pyplot as plt
    prob = Problem()

    prob.model= MissionAnalysisTest(num_integration_intervals_per_seg=3,propmodel=TurbopropPropulsionSystem)
    prob.model.nonlinear_solver= NewtonSolver()
    prob.model.linear_solver = DirectSolver()
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-7
    prob.model.nonlinear_solver.options['rtol'] = 1e-7
#    prob.model.nonlinear_solver.options['max_sub_solves'] = 1

    prob.driver = ScipyOptimizeDriver()
    prob.model.add_design_var('mission:cruise:h', lower=1000, upper=30000)
    prob.model.add_design_var('mission:climb:vs',lower=500,upper=3000)
    prob.model.add_design_var('mission:climb:Ueas',lower=85,upper=300)
    prob.model.add_design_var('mission:cruise:Ueas',lower=150,upper=300)
    prob.model.add_constraint('thrust.throttle',upper=np.ones(15)*0.95)
    prob.model.add_objective('mission:total_fuel')

    prob.setup()
    prob['thrust.throttle'] = np.ones(21)*0.7
    #prob['thrust.eng_throttle'] = np.ones(15)

    prob.run_model()
    #prob.run_driver()
    print(prob['mission:cruise:h'])

    # # print "------Prop 1-------"
    # print('Thrust: ' + str(prob['propmodel.prop1.thrust']))
    # plt.plot(prob['propmodel.prop1.thrust'])
    # plt.show()

    # print('Weight: ' + str(prob['propmodel.prop1.component_weight']))
    dtclimb = prob['mission:climb:dt']
    dtcruise = prob['mission:cruise:dt']
    dtdesc = prob['mission:descent:dt']
    n_int = 3
    timevec = np.concatenate([np.linspace(0,2*n_int*dtclimb,2*n_int+1),np.linspace(2*n_int*dtclimb,2*n_int*dtclimb+2*n_int*dtcruise,2*n_int+1),np.linspace(2*n_int*(dtclimb+dtcruise),2*n_int*(dtclimb+dtcruise+dtdesc),2*n_int+1)])
    plots = True
    if plots:
        print('Flight conditions')
        plt.figure(1)
        plt.plot(timevec, prob['conds.fltcond:mission:Ueas'],'b.')
        plt.plot(timevec, prob['atmos.trueairspeed.fltcond:mission:Utrue'],'b-')
        plt.plot(timevec, prob['gs.mission:groundspeed'],'g-')
        plt.title('Equivalent and true airspeed vs gs')

        print('Propulsion conditions')
        plt.figure(2)
        plt.plot(timevec, prob['thrust'])
        plt.title('Thrust')

        plt.figure(3)
        plt.plot(timevec, prob['mission:fuel_flow'])
        plt.title('Fuel flow')

        plt.figure(4)
        # plt.plot(np.delete(timevec,[0,20,41]),np.cumsum(prob['mission:segment_fuel']))
        plt.plot(timevec,prob['mission:weights'])
        plt.title('Weight')

        plt.figure(5)
        plt.plot(timevec,prob['fltcond:mission:CL'])
        plt.title('CL')

        plt.figure(6)
        plt.plot(timevec,prob['aero_drag'])
        plt.title('Drag')

        plt.figure(7)
        plt.plot(timevec,prob['propmodel.eng1.shaft_power_out'])
        plt.title('Shaft power')
        plt.show()
    print('Total fuel flow (totalizer):' + str(prob['mission:total_fuel']))
    print('Total fuel flow:' + str(np.sum(prob['mission:segment_fuel'])))


    #prob.model.list_inputs()
    #prob.model.list_outputs()
    #prob.check_partials(compact_print=True)
    #prob.check_totals(compact_print=True)
