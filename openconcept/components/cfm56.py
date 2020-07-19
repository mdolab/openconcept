from __future__ import division
import numpy as np
import openmdao.api as om
import openconcept
import matplotlib.pyplot as plt
from openconcept.utilities.surrogates.cached_kriging_surrogate import KrigingSurrogate

def CFM56(num_nodes=1, plot=False):
    """
    A simple turboshaft which generates shaft power consumes fuel.

    This model assumes constant power specific fuel consumption (PSFC).

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
    file_root = openconcept.__path__[0] + r'/components/empirical_data/cfm56/'
    thrustdata = np.load(file_root + 'cfm56thrust.npy')
    fuelburndata = np.load(file_root + 'cfm56wf.npy')
    t4data = np.load(file_root + r'cfm56t4.npy')
    krigedata = []
    for ialt, altitude in enumerate(np.array([35, 30, 25, 20, 15, 10, 5, 0])*1000.):
        for jmach, mach in enumerate(np.array([8, 7, 6, 5, 4, 3, 2, 1, 0.])*0.1):
            for kthrot, throttle in enumerate(np.array([10, 9, 8, 7, 6, 5, 4, 3, 2])*0.1):
                thrustijk = thrustdata[ialt, jmach, kthrot]
                if thrustijk > 0.0:
                    krigedata.append(np.array([throttle, altitude, mach, thrustijk.copy(), fuelburndata[ialt, jmach, kthrot].copy(), t4data[ialt, jmach, kthrot].copy()]))

    a = np.array(krigedata)
    comp = om.MetaModelUnStructuredComp(vec_size=num_nodes)
    comp.add_input('throttle', np.ones((num_nodes,))*1., training_data=a[:,0], units=None)
    comp.add_input('fltcond|h', np.ones((num_nodes,))*0., training_data=a[:,1], units='ft')
    comp.add_input('fltcond|M', np.ones((num_nodes,))*0.3, training_data=a[:,2], units=None)

    comp.add_output('thrust', np.ones((num_nodes,))*10000.,
                    training_data=a[:,3], units='lbf',
                    surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='cfm56thrust.pkl'))
    comp.add_output('fuel_flow', np.ones((num_nodes,))*3.0,
                    training_data=a[:,4], units='lbm/s',
                    surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='cfm56fuelburn.pkl'))
    comp.add_output('T4', np.ones((num_nodes,))*3000.,
                    training_data=a[:,5], units='R',
                    surrogate=KrigingSurrogate(cache_trained_model=True, cached_model_filename='cfm56T4.pkl'))
    comp.options['default_surrogate'] = KrigingSurrogate(lapack_driver='gesvd', cache_trained_model=True)

    if plot:
        prob = om.Problem()
        prob.model.add_subsystem('comp', comp)
        prob.setup()

        machs = np.linspace(0.0, 0.8, 25)
        alts = np.linspace(0.0, 35000., 25)
        machs, alts = np.meshgrid(machs, alts)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob['comp.throttle'] = 0.95
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
        plt.contourf(machs, alts, (pred[:,:,1] / pred[:,:,0])*60*60, levels=np.linspace(0.3,1.2,20))
        plt.colorbar()
        plt.figure()
        plt.xlabel('Mach')
        plt.ylabel('Altitude')
        plt.title('Thrust (lb)')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, pred[:,:,1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel('Mach')
        plt.ylabel('Altitude')
        plt.title('Fuel Flow (lb)')
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, pred[:,:,0], levels=20)
        plt.colorbar()
        plt.show()
    return comp
