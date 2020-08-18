from __future__ import division
import numpy as np
import openmdao.api as om
import openconcept
from openconcept.utilities.surrogates.cached_kriging_surrogate import KrigingSurrogate

def N3Opt(num_nodes=1, plot=False):
    """
    A geared turbofan based on NASA's N+3 architecture

    Inputs
    ------
    throttle: float
        Engine throttle. Controls power and fuel flow.
        Produces 100% of rated power at throttle = 1.
        Should be in range 0 to 1 or slightly above 1.
        (vector, dimensionless)
    fltcond|h: float
        Altitude
        (vector, dimensionless)
    fltcond|M: float
        Mach number
        (vector, dimensionless)

    Outputs
    -------
    thrust : float
        Thrust developed by the engine (vector, lbf)
    fuel_flow : float
        Fuel flow consumed (vector, lbm/s)
    T4 : float
        Turbine inlet temperature (vector, Rankine)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    """
    file_root = openconcept.__path__[0] + r'/components/empirical_data/n+3/'
    thrustdata = np.load(file_root + r'/power_off/thrust.npy')
    fuelburndata = np.load(file_root + r'/power_off/wf.npy')
    t4data = np.load(file_root + r'/power_off/t4.npy')
    altdata = np.load(file_root + r'/power_off/alt.npy')
    machdata = np.load(file_root + r'/power_off/mach.npy')
    throttledata = np.load(file_root + r'/power_off/throttle.npy')

    krigedata = []
    for ialt in range(8):
        for jmach in range(7):
            for kthrot in range(11):
                thrustijk = thrustdata[ialt, jmach, kthrot]
                if thrustijk > 0.0:
                    krigedata.append(np.array([throttledata[ialt, jmach, kthrot].copy(), 
                                               altdata[ialt, jmach, kthrot].copy(), 
                                               machdata[ialt, jmach ,kthrot].copy(), 
                                               thrustijk.copy(), 
                                               fuelburndata[ialt, jmach, kthrot].copy(), 
                                               t4data[ialt, jmach, kthrot].copy()]))

    a = np.array(krigedata)
    comp = om.MetaModelUnStructuredComp(vec_size=num_nodes)
    comp.add_input('throttle', np.ones((num_nodes,))*1., training_data=a[:,0], units=None)
    comp.add_input('fltcond|h', np.ones((num_nodes,))*0., training_data=a[:,1], units='ft')
    comp.add_input('fltcond|M', np.ones((num_nodes,))*0.3, training_data=a[:,2], units=None)

    comp.add_output('thrust', np.ones((num_nodes,))*10000.,
                    training_data=a[:,3], units='lbf',
                    surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='n3_poweroff_thrust.pkl'))
    comp.add_output('fuel_flow', np.ones((num_nodes,))*3.0,
                    training_data=a[:,4], units='lbm/s',
                    surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='n3_poweroff_fuelburn.pkl'))
    # comp.add_output('T4', np.ones((num_nodes,))*3000.,
    #                 training_data=a[:,5], units='R',
    #                 surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='n3_poweroff_T4.pkl'))
    comp.options['default_surrogate'] = KrigingSurrogate(lapack_driver='gesvd', cache_trained_model=True)

    if plot:
        import matplotlib.pyplot as plt
        prob = om.Problem()
        prob.model.add_subsystem('comp', comp)
        prob.setup()

        machs = np.linspace(0.2, 0.8, 25)
        alts = np.linspace(0.0, 35000., 25)
        machs, alts = np.meshgrid(machs, alts)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob['comp.throttle'] = 1.0
                prob['comp.fltcond|h'] = alts[i,j]
                prob['comp.fltcond|M'] = machs[i,j]
                prob.run_model()
                pred[i,j,0] = prob['comp.thrust'][0].copy()
                pred[i,j,1] = prob['comp.fuel_flow'][0].copy()
        plt.figure()
        plt.xlabel('Mach')
        plt.ylabel('Altitude')
        plt.title('SFC (lb / hr lb) OM')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, (pred[:,:,1] / pred[:,:,0])*60*60)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Mach')
        plt.ylabel('Altitude')
        plt.title('Fuel Flow (lb/s)')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, pred[:,:,1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Mach')
        plt.ylabel('Altitude')
        plt.title('Thrust (lb)')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, pred[:,:,0], levels=20)
        plt.colorbar()
        plt.show()

        throttles = np.linspace(0.00, 1.0, 25)
        alts = np.linspace(0.0, 35000., 25)
        throttles, alts = np.meshgrid(throttles, alts)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob['comp.throttle'] = throttles[i,j]
                prob['comp.fltcond|h'] = alts[i,j]
                prob['comp.fltcond|M'] = 0.3
                prob.run_model()
                pred[i,j,0] = prob['comp.thrust'][0].copy()
                pred[i,j,1] = prob['comp.fuel_flow'][0].copy()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Altitude')
        plt.title('SFC (lb / hr lb) OM')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(throttles, alts, (pred[:,:,1] / pred[:,:,0])*60*60)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Altitude')
        plt.title('Fuel Flow (lb/s)')
        # plt.contourf(throttles, alts, pred[:,:,0])
        plt.contourf(throttles, alts, pred[:,:,1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Altitude')
        plt.title('Thrust (lb)')
        # plt.contourf(throttles, alts, pred[:,:,0])
        plt.contourf(throttles, alts, pred[:,:,0], levels=20)
        plt.colorbar()
        plt.show()
    return comp

def N3Hybrid(num_nodes=1, plot=False):
    """
    Computes fuel savings attributable to hybridization

    Inputs
    ------
    throttle: float
        Engine throttle. Controls power and fuel flow.
        Produces 100% of rated power at throttle = 1.
        Should be in range 0 to 1 or slightly above 1.
        (vector, dimensionless)
    fltcond|h: float
        Altitude
        (vector, dimensionless)
    fltcond|M: float
        Mach number
        (vector, dimensionless)

    Outputs
    -------
    thrust : float
        Thrust developed by the engine (vector, lbf)
    fuel_flow : float
        Fuel flow consumed (vector, lbm/s)
    T4 : float
        Turbine inlet temperature (vector, Rankine)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    """
    file_root = openconcept.__path__[0] + r'/components/empirical_data/n+3/'
    thrustdata = np.load(file_root + r'/power_off/thrust.npy')
    fuelburndata_0 = np.load(file_root + r'/power_off/wf.npy')
    smwdata_0 = np.load(file_root + r'/power_off/SMW.npy')

    fuelburndata_500 = np.load(file_root + r'/power_on_500kW/wf.npy')
    smwdata_500 = np.load(file_root + r'/power_on_500kW/SMW.npy')
    fuelburndata_1000 = np.load(file_root + r'/power_on_1MW/wf.npy')
    smwdata_1000 = np.load(file_root + r'/power_on_1MW/SMW.npy')

    altdata = np.load(file_root + r'/power_off/alt.npy')
    machdata = np.load(file_root + r'/power_off/mach.npy')
    throttledata = np.load(file_root + r'/power_off/throttle.npy')

    krigedata = []
    # do the base case
    for ialt in range(8):
        for jmach in range(7):
            for kthrot in range(11):
                fuelburnijk = fuelburndata_0[ialt, jmach, kthrot]
                if fuelburnijk > 0.0:
                    krigedata.append(np.array([throttledata[ialt, jmach, kthrot].copy(), 
                                               altdata[ialt, jmach, kthrot].copy(), 
                                               machdata[ialt, jmach ,kthrot].copy(),
                                               0.0,  
                                               thrustdata[ialt, jmach, kthrot].copy(), 
                                               fuelburnijk.copy(),
                                               smwdata_0[ialt, jmach, kthrot].copy()]))
    # do the 500kW case
    for ialt in range(8):
        for jmach in range(7):
            for kthrot in range(11):
                fuelburnijk = fuelburndata_500[ialt, jmach, kthrot]
                if fuelburnijk > 0.0:
                    krigedata.append(np.array([throttledata[ialt, jmach, kthrot].copy(), 
                                               altdata[ialt, jmach, kthrot].copy(), 
                                               machdata[ialt, jmach ,kthrot].copy(),
                                               500.0,  
                                               thrustdata[ialt, jmach, kthrot].copy(), 
                                               fuelburnijk.copy(),
                                               smwdata_500[ialt, jmach, kthrot].copy()]))

    # do the 1MW case
    for ialt in range(8):
        for jmach in range(7):
            for kthrot in range(11):
                fuelburnijk = fuelburndata_1000[ialt, jmach, kthrot]
                if fuelburnijk > 0.0:
                    krigedata.append(np.array([throttledata[ialt, jmach, kthrot].copy(), 
                                               altdata[ialt, jmach, kthrot].copy(), 
                                               machdata[ialt, jmach ,kthrot].copy(),
                                               1000.0,  
                                               thrustdata[ialt, jmach, kthrot].copy(), 
                                               fuelburnijk.copy(),
                                               smwdata_1000[ialt, jmach, kthrot].copy()]))

    a = np.array(krigedata)
    comp = om.MetaModelUnStructuredComp(vec_size=num_nodes)
    comp.add_input('throttle', np.ones((num_nodes,))*1., training_data=a[:,0], units=None)
    comp.add_input('fltcond|h', np.ones((num_nodes,))*0., training_data=a[:,1], units='ft')
    comp.add_input('fltcond|M', np.ones((num_nodes,))*0.3, training_data=a[:,2], units=None)
    comp.add_input('hybrid_power', np.zeros((num_nodes,)), training_data=a[:,3], units='kW')

    comp.add_output('thrust', np.ones((num_nodes,))*10000.,
                    training_data=a[:,4], units='lbf',
                    surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='n3_hybrid_thrust.pkl'))
    comp.add_output('fuel_flow', np.ones((num_nodes,))*3.0,
                    training_data=a[:,5], units='lbm/s',
                    surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='n3_hybrid_fuelflow.pkl'))
    comp.add_output('surge_margin', np.ones((num_nodes,))*3.0,
                    training_data=a[:,6], units=None,
                    surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='n3_hybrid_smw.pkl'))
    comp.options['default_surrogate'] = KrigingSurrogate(lapack_driver='gesvd', cache_trained_model=True)

    if plot:
        import matplotlib.pyplot as plt
        prob = om.Problem()
        prob.model.add_subsystem('comp', comp)
        prob.setup()

        machs = np.linspace(0.2, 0.8, 25)
        alts = np.linspace(0.0, 35000., 25)
        machs, alts = np.meshgrid(machs, alts)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob.set_val('comp.hybrid_power', 1000., 'kW')
                prob['comp.throttle'] = 1.0
                prob['comp.fltcond|h'] = alts[i,j]
                prob['comp.fltcond|M'] = machs[i,j]
                prob.run_model()
                pred[i,j,0] = prob['comp.thrust'][0].copy()
                pred[i,j,1] = prob['comp.fuel_flow'][0].copy()
        plt.figure()
        plt.xlabel('Mach')
        plt.ylabel('Altitude')
        plt.title('SFC (lb / hr lb) OM')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, (pred[:,:,1] / pred[:,:,0])*60*60)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Mach')
        plt.ylabel('Altitude')
        plt.title('Fuel Flow (lb/s)')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, pred[:,:,1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Mach')
        plt.ylabel('Altitude')
        plt.title('Thrust (lb)')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, pred[:,:,0], levels=20)
        plt.colorbar()
        plt.show()

        throttles = np.linspace(0.1, 1.0, 25)
        alts = np.linspace(0.0, 35000., 25)
        throttles, alts = np.meshgrid(throttles, alts)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob.set_val('comp.hybrid_power', 0., 'kW')
                prob['comp.throttle'] = throttles[i,j]
                prob['comp.fltcond|h'] = alts[i,j]
                prob['comp.fltcond|M'] = 0.5
                prob.run_model()
                pred[i,j,0] = prob['comp.thrust'][0].copy()
                pred[i,j,1] = prob['comp.fuel_flow'][0].copy()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Altitude')
        plt.title('SFC (lb / hr lb) OM')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(throttles, alts, (pred[:,:,1] / pred[:,:,0])*60*60)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Altitude')
        plt.title('Fuel Flow (lb/s)')
        # plt.contourf(throttles, alts, pred[:,:,0])
        plt.contourf(throttles, alts, pred[:,:,1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Altitude')
        plt.title('Thrust (lb)')
        # plt.contourf(throttles, alts, pred[:,:,0])
        plt.contourf(throttles, alts, pred[:,:,0], levels=20)
        plt.colorbar()
        plt.show()

        powers = np.linspace(0, 1000, 25)
        throttles = np.linspace(0.3, 1.0, 25)
        powers, throttles = np.meshgrid(powers, throttles)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob['comp.hybrid_power'] = powers[i,j]
                prob['comp.throttle'] = throttles[i,j]
                prob.set_val('comp.fltcond|h', 33000.0, units='ft')
                prob['comp.fltcond|M'] = 0.8
                prob.run_model()
                pred[i,j,0] = prob['comp.thrust'][0].copy()
                pred[i,j,1] = prob['comp.fuel_flow'][0].copy()
                pred[i,j,2] = prob['comp.surge_margin'][0].copy()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Hybrid Power (kW)')
        plt.title('SFC (lb / hr lb) OM')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(throttles, powers, (pred[:,:,1] / pred[:,:,0])*60*60)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Hybrid Power (kW)')
        plt.title('Fuel Flow (lb/s)')
        # plt.contourf(throttles, powers, pred[:,:,0])
        plt.contourf(throttles, powers, pred[:,:,1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Hybrid Power (kW)')
        plt.title('Thrust (lb)')
        # plt.contourf(throttles, powers, pred[:,:,0])
        plt.contourf(throttles, powers, pred[:,:,0], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Throttle')
        plt.ylabel('Hybrid Power (kW)')
        plt.title('Surge margin')
        # plt.contourf(throttles, powers, pred[:,:,0])
        plt.contourf(throttles, powers, pred[:,:,2], levels=20)
        plt.colorbar()
        plt.show()
    return comp