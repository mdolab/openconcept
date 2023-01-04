import numpy as np
import openmdao.api as om
import openconcept


def N3Hybrid(num_nodes=1, plot=False):
    """
    Returns OpenMDAO component for thrust deck
    for optimized N+3 GTF with hybridization.

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
    hybrid_power : float
        Shaft power added to LP shaft
        (vector, kW)

    Outputs
    -------
    thrust : float
        Thrust developed by the engine (vector, lbf)
    fuel_flow : float
        Fuel flow consumed (vector, lbm/s)
    surge_margin : float
        Surge margin (vector, percent)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    """
    file_root = openconcept.__path__[0] + r"/propulsion/empirical_data/n+3_hybrid/"
    thrustdata = np.load(file_root + r"/power_off/thrust.npy")
    fuelburndata_0 = np.load(file_root + r"/power_off/wf.npy")
    smwdata_0 = np.load(file_root + r"/power_off/SMW.npy")

    fuelburndata_500 = np.load(file_root + r"/power_on_500kW/wf.npy")
    smwdata_500 = np.load(file_root + r"/power_on_500kW/SMW.npy")
    fuelburndata_1000 = np.load(file_root + r"/power_on_1MW/wf.npy")
    smwdata_1000 = np.load(file_root + r"/power_on_1MW/SMW.npy")

    altdata = np.load(file_root + r"/power_off/alt.npy")
    machdata = np.load(file_root + r"/power_off/mach.npy")
    throttledata = np.load(file_root + r"/power_off/throttle.npy")

    krigedata = []
    # do the base case
    for ialt in range(8):
        for jmach in range(7):
            for kthrot in range(11):
                fuelburnijk = fuelburndata_0[ialt, jmach, kthrot]
                if fuelburnijk > 0.0:
                    krigedata.append(
                        np.array(
                            [
                                throttledata[ialt, jmach, kthrot].copy(),
                                altdata[ialt, jmach, kthrot].copy(),
                                machdata[ialt, jmach, kthrot].copy(),
                                0.0,
                                thrustdata[ialt, jmach, kthrot].copy(),
                                fuelburnijk.copy(),
                                smwdata_0[ialt, jmach, kthrot].copy(),
                            ]
                        )
                    )
    # do the 500kW case
    for ialt in range(8):
        for jmach in range(7):
            for kthrot in range(11):
                fuelburnijk = fuelburndata_500[ialt, jmach, kthrot]
                if fuelburnijk > 0.0:
                    krigedata.append(
                        np.array(
                            [
                                throttledata[ialt, jmach, kthrot].copy(),
                                altdata[ialt, jmach, kthrot].copy(),
                                machdata[ialt, jmach, kthrot].copy(),
                                500.0,
                                thrustdata[ialt, jmach, kthrot].copy(),
                                fuelburnijk.copy(),
                                smwdata_500[ialt, jmach, kthrot].copy(),
                            ]
                        )
                    )

    # do the 1MW case
    for ialt in range(8):
        for jmach in range(7):
            for kthrot in range(11):
                fuelburnijk = fuelburndata_1000[ialt, jmach, kthrot]
                if fuelburnijk > 0.0:
                    krigedata.append(
                        np.array(
                            [
                                throttledata[ialt, jmach, kthrot].copy(),
                                altdata[ialt, jmach, kthrot].copy(),
                                machdata[ialt, jmach, kthrot].copy(),
                                1000.0,
                                thrustdata[ialt, jmach, kthrot].copy(),
                                fuelburnijk.copy(),
                                smwdata_1000[ialt, jmach, kthrot].copy(),
                            ]
                        )
                    )

    a = np.array(krigedata)
    comp = om.MetaModelUnStructuredComp(vec_size=num_nodes)
    comp.add_input("throttle", np.ones((num_nodes,)) * 1.0, training_data=a[:, 0], units=None)
    comp.add_input("fltcond|h", np.ones((num_nodes,)) * 0.0, training_data=a[:, 1], units="ft")
    comp.add_input("fltcond|M", np.ones((num_nodes,)) * 0.3, training_data=a[:, 2], units=None)
    comp.add_input("hybrid_power", np.zeros((num_nodes,)), training_data=a[:, 3], units="kW")

    comp.add_output(
        "thrust",
        np.ones((num_nodes,)) * 10000.0,
        training_data=a[:, 4],
        units="lbf",
        surrogate=om.KrigingSurrogate(training_cache=file_root + r"n3_hybrid_thrust_trained.zip"),
    )
    comp.add_output(
        "fuel_flow",
        np.ones((num_nodes,)) * 3.0,
        training_data=a[:, 5],
        units="lbm/s",
        surrogate=om.KrigingSurrogate(training_cache=file_root + r"n3_hybrid_fuelflow_trained.zip"),
    )
    comp.add_output(
        "surge_margin",
        np.ones((num_nodes,)) * 3.0,
        training_data=a[:, 6],
        units=None,
        surrogate=om.KrigingSurrogate(training_cache=file_root + r"n3_hybrid_smw_trained.zip"),
    )
    comp.options["default_surrogate"] = om.KrigingSurrogate(lapack_driver="gesvd")

    if plot:
        import matplotlib.pyplot as plt

        prob = om.Problem()
        prob.model.add_subsystem("comp", comp)
        prob.setup()

        machs = np.linspace(0.2, 0.8, 25)
        alts = np.linspace(0.0, 35000.0, 25)
        machs, alts = np.meshgrid(machs, alts)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob.set_val("comp.hybrid_power", 1000.0, "kW")
                prob["comp.throttle"] = 1.0
                prob["comp.fltcond|h"] = alts[i, j]
                prob["comp.fltcond|M"] = machs[i, j]
                prob.run_model()
                pred[i, j, 0] = prob["comp.thrust"][0].copy()
                pred[i, j, 1] = prob["comp.fuel_flow"][0].copy()
        plt.figure()
        plt.xlabel("Mach")
        plt.ylabel("Altitude")
        plt.title("SFC (lb / hr lb) OM")
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, (pred[:, :, 1] / pred[:, :, 0]) * 60 * 60)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Mach")
        plt.ylabel("Altitude")
        plt.title("Fuel Flow (lb/s)")
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, pred[:, :, 1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Mach")
        plt.ylabel("Altitude")
        plt.title("Thrust (lb)")
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(machs, alts, pred[:, :, 0], levels=20)
        plt.colorbar()
        plt.show()

        throttles = np.linspace(0.1, 1.0, 25)
        alts = np.linspace(0.0, 35000.0, 25)
        throttles, alts = np.meshgrid(throttles, alts)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob.set_val("comp.hybrid_power", 0.0, "kW")
                prob["comp.throttle"] = throttles[i, j]
                prob["comp.fltcond|h"] = alts[i, j]
                prob["comp.fltcond|M"] = 0.5
                prob.run_model()
                pred[i, j, 0] = prob["comp.thrust"][0].copy()
                pred[i, j, 1] = prob["comp.fuel_flow"][0].copy()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Altitude")
        plt.title("SFC (lb / hr lb) OM")
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(throttles, alts, (pred[:, :, 1] / pred[:, :, 0]) * 60 * 60)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Altitude")
        plt.title("Fuel Flow (lb/s)")
        # plt.contourf(throttles, alts, pred[:,:,0])
        plt.contourf(throttles, alts, pred[:, :, 1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Altitude")
        plt.title("Thrust (lb)")
        # plt.contourf(throttles, alts, pred[:,:,0])
        plt.contourf(throttles, alts, pred[:, :, 0], levels=20)
        plt.colorbar()
        plt.show()

        powers = np.linspace(0, 1000, 25)
        throttles = np.linspace(0.1, 1.0, 25)
        powers, throttles = np.meshgrid(powers, throttles)
        pred = np.zeros((25, 25, 3))
        for i in range(25):
            for j in range(25):
                prob["comp.hybrid_power"] = powers[i, j]
                prob["comp.throttle"] = throttles[i, j]
                prob.set_val("comp.fltcond|h", 33000.0, units="ft")
                prob["comp.fltcond|M"] = 0.8
                prob.run_model()
                pred[i, j, 0] = prob["comp.thrust"][0].copy()
                pred[i, j, 1] = prob["comp.fuel_flow"][0].copy()
                pred[i, j, 2] = prob["comp.surge_margin"][0].copy()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Hybrid Power (kW)")
        plt.title("SFC (lb / hr lb) OM")
        # plt.contourf(machs, alts, pred[:,:,0])
        plt.contourf(throttles, powers, (pred[:, :, 1] / pred[:, :, 0]) * 60 * 60)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Hybrid Power (kW)")
        plt.title("Fuel Flow (lb/s)")
        # plt.contourf(throttles, powers, pred[:,:,0])
        plt.contourf(throttles, powers, pred[:, :, 1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Hybrid Power (kW)")
        plt.title("Thrust (lb)")
        # plt.contourf(throttles, powers, pred[:,:,0])
        plt.contourf(throttles, powers, pred[:, :, 0], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Hybrid Power (kW)")
        plt.title("Surge margin")
        # plt.contourf(throttles, powers, pred[:,:,0])
        plt.contourf(throttles, powers, pred[:, :, 2], levels=20)
        plt.colorbar()
        plt.show()
    return comp


def N3(num_nodes=1, plot=False):
    """
    Returns OpenMDAO component for thrust deck
    for optimized N+3 GTF without hybridization.

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
    surge_margin : float
        Surge margin (vector, percent)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    """
    file_root = openconcept.__path__[0] + r"/propulsion/empirical_data/n+3/"
    thrustdata = np.load(file_root + r"/power_off/thrust.npy")
    fuelburndata_0 = np.load(file_root + r"/power_off/wf.npy")
    smwdata_0 = np.load(file_root + r"/power_off/SMW.npy")
    altdata = np.load(file_root + r"/power_off/alt.npy")
    machdata = np.load(file_root + r"/power_off/mach.npy")
    throttledata = np.load(file_root + r"/power_off/throttle.npy")

    krigedata = []
    for ialt in range(8):
        for jmach in range(7):
            for kthrot in range(11):
                fuelburnijk = fuelburndata_0[ialt, jmach, kthrot]
                if fuelburnijk > 0.0:
                    krigedata.append(
                        np.array(
                            [
                                throttledata[ialt, jmach, kthrot].copy(),
                                altdata[ialt, jmach, kthrot].copy(),
                                machdata[ialt, jmach, kthrot].copy(),
                                thrustdata[ialt, jmach, kthrot].copy(),
                                fuelburnijk.copy(),
                                smwdata_0[ialt, jmach, kthrot].copy(),
                            ]
                        )
                    )

    a = np.array(krigedata)
    comp = om.MetaModelUnStructuredComp(vec_size=num_nodes)
    comp.add_input("throttle", np.ones((num_nodes,)) * 1.0, training_data=a[:, 0], units=None)
    comp.add_input("fltcond|h", np.ones((num_nodes,)) * 0.0, training_data=a[:, 1], units="ft")
    comp.add_input("fltcond|M", np.ones((num_nodes,)) * 0.3, training_data=a[:, 2], units=None)

    comp.add_output(
        "thrust",
        np.ones((num_nodes,)) * 10000.0,
        training_data=a[:, 3],
        units="lbf",
        surrogate=om.KrigingSurrogate(training_cache=file_root + r"n3_thrust_trained.zip"),
    )
    comp.add_output(
        "fuel_flow",
        np.ones((num_nodes,)) * 3.0,
        training_data=a[:, 4],
        units="lbm/s",
        surrogate=om.KrigingSurrogate(training_cache=file_root + r"n3_fuelflow_trained.zip"),
    )
    comp.add_output(
        "surge_margin",
        np.ones((num_nodes,)) * 3.0,
        training_data=a[:, 5],
        units=None,
        surrogate=om.KrigingSurrogate(training_cache=file_root + r"n3_smw_trained.zip"),
    )
    comp.options["default_surrogate"] = om.KrigingSurrogate(lapack_driver="gesvd")

    if plot:
        import matplotlib.pyplot as plt

        prob = om.Problem()
        prob.model.add_subsystem("comp", comp)
        prob.setup()

        nmachs = 7
        nalts = 8

        machs = np.linspace(0.2, 0.8, nmachs)
        alts = np.linspace(0.0, 35000.0, nalts)
        machs, alts = np.meshgrid(machs, alts, indexing="ij")
        pred = np.zeros((nmachs, nalts, 3))
        for i in range(nmachs):
            for j in range(nalts):
                prob["comp.throttle"] = 1.0
                prob["comp.fltcond|h"] = alts[i, j]
                prob["comp.fltcond|M"] = machs[i, j]
                prob.run_model()
                pred[i, j, 0] = prob["comp.thrust"][0].copy()
                pred[i, j, 1] = prob["comp.fuel_flow"][0].copy()
        plt.figure()
        plt.xlabel("Mach")
        plt.ylabel("Altitude")
        plt.title("SFC (lb / hr lb) OM")
        plt.contourf(machs, alts, (pred[:, :, 1] / pred[:, :, 0]) * 60 * 60)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Mach")
        plt.ylabel("Altitude")
        plt.title("Fuel Flow (lb/s)")
        plt.contourf(machs, alts, pred[:, :, 1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Mach")
        plt.ylabel("Altitude")
        plt.title("Thrust (lb)")
        plt.contourf(machs, alts, pred[:, :, 0], levels=20)
        plt.colorbar()
        plt.show()

        nthrottles = 10
        throttles = np.linspace(0.1, 1.0, nthrottles)
        alts = np.linspace(0.0, 35000.0, nalts)
        throttles, alts = np.meshgrid(throttles, alts, indexing="ij")
        pred = np.zeros((nthrottles, nalts, 3))
        for i in range(nthrottles):
            for j in range(nalts):
                prob["comp.throttle"] = throttles[i, j]
                prob["comp.fltcond|h"] = alts[i, j]
                prob["comp.fltcond|M"] = 0.5
                prob.run_model()
                pred[i, j, 0] = prob["comp.thrust"][0].copy()
                pred[i, j, 1] = prob["comp.fuel_flow"][0].copy()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Altitude")
        plt.title("SFC (lb / hr lb) OM")
        plt.contourf(throttles, alts, (pred[:, :, 1] / pred[:, :, 0]) * 60 * 60)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Altitude")
        plt.title("Fuel Flow (lb/s)")
        plt.contourf(throttles, alts, pred[:, :, 1], levels=20)
        plt.colorbar()
        plt.figure()
        plt.xlabel("Throttle")
        plt.ylabel("Altitude")
        plt.title("Thrust (lb)")
        plt.contourf(throttles, alts, pred[:, :, 0], levels=20)
        plt.colorbar()
        plt.show()
    return comp


def compare_thrust_decks():
    import matplotlib.pyplot as plt

    prob = om.Problem()
    from openconcept.propulsion import N3, N3Hybrid

    prob.model.add_subsystem("n3", N3(num_nodes=1))
    prob.model.add_subsystem("n3hybrid", N3Hybrid(num_nodes=1))
    bal = prob.model.add_subsystem("bal", om.BalanceComp())
    bal.add_balance("n3hybrid_throttle", lower=0.05, upper=1.1, val=1.0)
    prob.model.connect("n3.thrust", "bal.rhs:n3hybrid_throttle")
    prob.model.connect("n3hybrid.thrust", "bal.lhs:n3hybrid_throttle")
    prob.model.connect("bal.n3hybrid_throttle", "n3hybrid.throttle")
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.setup()

    nmachs = 7
    nalts = 8

    machs = np.linspace(0.2, 0.8, nmachs)
    alts = np.linspace(0.0, 35000.0, nalts)
    machs, alts = np.meshgrid(machs, alts, indexing="ij")
    predn3 = np.zeros((nmachs, nalts, 3))
    predn3hybrid = np.zeros((nmachs, nalts, 3))

    for i in range(nmachs):
        for j in range(nalts):
            prob["n3.throttle"] = 1.0
            prob["n3.fltcond|h"] = alts[i, j]
            prob["n3.fltcond|M"] = machs[i, j]
            prob["n3hybrid.fltcond|h"] = alts[i, j]
            prob["n3hybrid.fltcond|M"] = machs[i, j]
            prob.run_model()
            predn3[i, j, 0] = prob["n3.thrust"][0].copy()
            predn3[i, j, 1] = prob["n3.fuel_flow"][0].copy()
            predn3hybrid[i, j, 0] = prob["n3hybrid.thrust"][0].copy()
            predn3hybrid[i, j, 1] = prob["n3hybrid.fuel_flow"][0].copy()
    plt.figure()
    plt.xlabel("Mach")
    plt.ylabel("Altitude")
    plt.title("SFC (lb / hr lb) OM")
    plt.contourf(machs, alts, (predn3[:, :, 1] / predn3[:, :, 0]) / (predn3hybrid[:, :, 1] / predn3hybrid[:, :, 0]))
    plt.colorbar()
    plt.figure()
    plt.xlabel("Mach")
    plt.ylabel("Altitude")
    plt.title("Fuel Flow (lb/s)")
    plt.contourf(machs, alts, predn3[:, :, 1] / predn3hybrid[:, :, 1], levels=20)
    plt.colorbar()
    plt.figure()
    plt.xlabel("Mach")
    plt.ylabel("Altitude")
    plt.title("Thrust (lb)")
    plt.contourf(machs, alts, predn3[:, :, 0] / predn3hybrid[:, :, 0], levels=20)
    plt.colorbar()
    plt.show()

    nthrottles = 10
    throttles = np.linspace(0.1, 1.0, nthrottles)
    alts = np.linspace(0.0, 35000.0, nalts)
    throttles, alts = np.meshgrid(throttles, alts, indexing="ij")
    predn3 = np.zeros((nthrottles, nalts, 3))
    predn3hybrid = np.zeros((nthrottles, nalts, 3))
    for i in range(nthrottles):
        for j in range(nalts):
            prob["n3.throttle"] = throttles[i, j]
            prob["n3.fltcond|h"] = alts[i, j]
            prob["n3.fltcond|M"] = 0.5
            prob["n3hybrid.fltcond|h"] = alts[i, j]
            prob["n3hybrid.fltcond|M"] = 0.5
            prob.run_model()
            predn3[i, j, 0] = prob["n3.thrust"][0].copy()
            predn3[i, j, 1] = prob["n3.fuel_flow"][0].copy()
            predn3hybrid[i, j, 0] = prob["n3hybrid.thrust"][0].copy()
            predn3hybrid[i, j, 1] = prob["n3hybrid.fuel_flow"][0].copy()
    plt.figure()
    plt.xlabel("Throttle")
    plt.ylabel("Altitude")
    plt.title("SFC (lb / hr lb) OM")
    plt.contourf(throttles, alts, (predn3[:, :, 1] / predn3[:, :, 0]) / (predn3hybrid[:, :, 1] / predn3hybrid[:, :, 0]))
    plt.colorbar()
    plt.figure()
    plt.xlabel("Throttle")
    plt.ylabel("Altitude")
    plt.title("Fuel Flow ratio (N3 / N3hybrid) at equal thrust, M=0.8")
    plt.contourf(throttles, alts, predn3[:, :, 1] / predn3hybrid[:, :, 1], levels=10)
    plt.colorbar()
    plt.figure()
    plt.xlabel("Throttle")
    plt.ylabel("Altitude")
    plt.title("Thrust (lb)")
    plt.contourf(throttles, alts, predn3[:, :, 0] / predn3hybrid[:, :, 0], levels=20)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    compare_thrust_decks()
