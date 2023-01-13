import numpy as np


def gh2_cv(P, T):
    """
    Compute specific heat of hydrogen gas at constant volume.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    It is based on data collected at pressures between 100-750 kPa and 20-150 K.
    See page 100 of the thesis for more details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa)
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
        Specific heat at constant volume of gaseous hydrogen (J/(kg-K))
    """
    # Convert pressure to MPa
    P *= 1e-6

    cv = 56.0992565764207

    H1_1 = np.tanh((0.3990654435825 + 0.275976950953691 * P + -0.0364541001467881 * T))
    H1_2 = np.tanh((-4.86826687998567 + -2.62940079620686 * P + 0.216404289884235 * T))
    H1_3 = np.tanh((1.12548975067467 + 0.269571399186571 * P + -0.0349250996335972 * T))
    H1_4 = np.tanh((-5.57528770123298 + -3.1347851189393 * P + 0.276378205247149 * T))
    H1_5 = np.tanh((-0.768555052808218 + -0.415187001771689 * P + 0.0303660811195935 * T))
    H1_6 = np.tanh((0.271020942398391 + -1.67288064390926 * P + 0.0673489649704582 * T))
    H1_7 = np.tanh((-1.10142968268782 + 0.0083174017874869 * P + 0.0108662672630239 * T))
    H1_8 = np.tanh((1.90064900956529 + 1.17456681024276 * P + -0.0902336799991085 * T))
    H1_9 = np.tanh((-0.673064219823259 + -0.943525497821755 * P + 0.0734160926798799 * T))
    H1_10 = np.tanh((-1.6061497898911 + -1.99739736255923 * P + 0.108199114343877 * T))
    H1_11 = np.tanh((1.73534543350612 + 0.388202467775416 * P + -0.0616088045146524 * T))
    H1_12 = np.tanh((2.70166390032434 + 1.3326759135354 * P + -0.118211564739528 * T))
    H1_13 = np.tanh((-9.32984593347698 + 9.42488755971789 * P + 0.0784916020511085 * T))
    H1_14 = np.tanh((1.2207418073194 + 1.13189377766378 * P + -0.0841043542652104 * T))
    H1_15 = np.tanh((2.58665026763982 + 1.84291010959674 * P + -0.13970364770731 * T))

    cv += -70.3037835837473 * H1_1
    cv += -12.804152874995 * H1_10
    cv += 8.56885046215908 * H1_11
    cv += 16.8828056394326 * H1_12
    cv += 0.000773883721821057 * H1_13
    cv += -11.6665802642623 * H1_14
    cv += -43.8044401310621 * H1_15
    cv += -3.65980723192549 * H1_2
    cv += 16.4087880702894 * H1_3
    cv += -3.5451526649205 * H1_4
    cv += 1.59237668513045 * H1_5
    cv += 16.3356631788945 * H1_6
    cv += 2.87332235554376 * H1_7
    cv += -32.0695317395356 * H1_8
    cv += -162.92972771138 * H1_9

    # Convert from J/(g-K) to J/(kg-K)
    cv *= 1e3

    return cv


def gh2_cp(P, T):
    """
    Compute specific heat of hydrogen gas at constant pressure.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    It is based on data collected at pressures between 100-750 kPa and 20-150 K.
    See page 100 of the thesis for more details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa)
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
        Specific heat at constant pressure of gaseous hydrogen (J/(kg-K))
    """
    # Convert pressure to MPa
    P *= 1e-6

    cp = -163.178217071559

    H1_1 = np.tanh((0.614735260887242 + 1.13159708689763 * P + -0.0551837204578247 * T))
    H1_2 = np.tanh((-0.400739830511314 + -0.0140493655427582 * P + -0.000581742517018706 * T))
    H1_3 = np.tanh((-1.25439265281633 + -0.584213145968296 * P + 0.100382842687856 * T))
    H1_4 = np.tanh((0.613718515601412 + 1.1389617257106 * P + -0.0766902288518594 * T))
    H1_5 = np.tanh((0.783863063691068 + 1.12786935775064 * P + -0.0580710892849312 * T))
    H1_6 = np.tanh((3.0862845706326 + 6.11836677910538 * P + -0.28171194186548 * T))
    H1_7 = np.tanh((-4.01267608558022 + -1.74531315193444 * P + 0.13245397217098 * T))
    H1_8 = np.tanh((-4.73597371609159 + -2.69493649415449 * P + 0.241924487447467 * T))
    H1_9 = np.tanh((-3.13544345816356 + -2.55750416599446 * P + 0.149566251986045 * T))
    H1_10 = np.tanh((-3.57473472514032 + -4.95468689159338 * P + 0.281591654527403 * T))
    H1_11 = np.tanh((4.41722437760112 + 3.38001091079686 * P + -0.242556425504075 * T))
    H1_12 = np.tanh((-0.398001325855848 + 0.0072937040171696 * P + 0.000277104672759379 * T))
    H1_13 = np.tanh((-0.31379942783082 + 0.576128323850331 * P + 0.0347928080405698 * T))
    H1_14 = np.tanh((7.34692537702712 + 1.16723768888899 * P + -0.317917699002537 * T))
    H1_15 = np.tanh((-0.816554474050802 + 0.784988883943678 * P + 0.0377876586456731 * T))

    cp += 346.455492384544 * H1_1
    cp += -131.305889878852 * H1_10
    cp += -87.5906286526561 * H1_11
    cp += -881.40560167543 * H1_12
    cp += -87.1001293092837 * H1_13
    cp += -2.24612605221145 * H1_14
    cp += 40.3225563600741 * H1_15
    cp += -521.915978427339 * H1_2
    cp += 137.870380603119 * H1_3
    cp += 452.163820434345 * H1_4
    cp += -374.519768532003 * H1_5
    cp += -41.7239727718235 * H1_6
    cp += 2.51241979089427 * H1_7
    cp += -55.4472053157707 * H1_8
    cp += 23.7176680036494 * H1_9

    # Convert from J/(g-K) to J/(kg-K)
    cp *= 1e3

    return cp


def gh2_u(P, T):
    """
    Compute internal energy of hydrogen gas.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    It is based on data collected at pressures between 100-750 kPa and 20-150 K.
    See page 100 of the thesis for more details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa)
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
        Internal energy of gaseous hydrogen (J/kg)
    """
    # Convert pressure to MPa
    P *= 1e-6

    u = 673.611983193655

    H1_1 = np.tanh((0.233539747755315 + 0.948951297504785 * P + -0.0499745330828293 * T))
    H1_2 = np.tanh((2.05471470504367 + 2.13093505892378 * P + -0.119572061260962 * T))
    H1_3 = np.tanh((0.398892133197 + 0.587703772663913 * P + -0.00336363426965824 * T))
    H1_4 = np.tanh((0.143237719165029 + 0.992347038797521 * P + -0.0143964824374372 * T))
    H1_5 = np.tanh((-3.96616131435837 + -2.74657042229242 * P + 0.186453818992668 * T))
    H1_6 = np.tanh((1.44405264544866 + -0.645489048931109 * P + -0.0104476083443945 * T))
    H1_7 = np.tanh((2.02006732389655 + 3.10126879273836 * P + -0.132381179154892 * T))
    H1_8 = np.tanh((0.455915191247191 + -0.295895155010423 * P + -0.00270733720731719 * T))
    H1_9 = np.tanh((-1.42121486333193 + 0.135982980631872 * P + 0.0473542542463267 * T))
    H1_10 = np.tanh((-0.662630708916202 + -0.206113999100671 * P + 0.00414455453741901 * T))
    H1_11 = np.tanh((0.56740916395517 + 1.67371525829477 * P + -0.0218895513528701 * T))
    H1_12 = np.tanh((0.237517777100042 + 0.1589500143037 * P + -0.0106519272801902 * T))
    H1_13 = np.tanh((1.54182270996619 + 0.754447306182754 * P + -0.0541064232047049 * T))
    H1_14 = np.tanh((2.62921984539398 + 1.72079624611917 * P + -0.0971068814297633 * T))
    H1_15 = np.tanh((-1.04212680836281 + -0.484939056845666 * P + 0.0116518536560464 * T))

    u += -1446.64509367172 * H1_1
    u += 1174.42287038803 * H1_10
    u += 26.0663709626282 * H1_11
    u += 478.510727358678 * H1_12
    u += 213.76235040309 * H1_13
    u += 47.1707429768787 * H1_14
    u += -14.8049823561287 * H1_15
    u += 399.518468943879 * H1_2
    u += -333.71693491459 * H1_3
    u += -228.44619912214 * H1_4
    u += 30.515033222315 * H1_5
    u += 74.6722187603707 * H1_6
    u += -58.2024392498572 * H1_7
    u += -1594.44683837146 * H1_8
    u += 69.542870910495 * H1_9

    # Convert from kJ/kg to J/kg
    u *= 1e3

    return u


def gh2_h(P, T):
    """
    Compute enthalpy of hydrogen gas.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    It is based on data collected at pressures between 100-750 kPa and 20-150 K.
    See page 100 of the thesis for more details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa)
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Enthalpy of gaseous hydrogen (J/kg)
    """
    # Convert pressure to MPa
    P *= 1e-6

    h = 1281.75444728572

    H1_1 = np.tanh((1.73579390297524 + -0.511704824190866 * P + -0.00328063298246929 * T))
    H1_2 = np.tanh((-0.17506458742044 + -0.0787067450602815 * P + 0.0043943295092826 * T))
    H1_3 = np.tanh((-1.49793344088779 + -2.75869130445277 * P + 0.110698824574587 * T))
    H1_4 = np.tanh((-1.37669485784553 + 0.133730746273731 * P + 0.00649135107315252 * T))
    H1_5 = np.tanh((-1.1405914153055 + -1.88672855275111 * P + 0.0981301522587858 * T))
    H1_6 = np.tanh((0.446124093572671 + 0.429585607469454 * P + -0.0285787427235227 * T))
    H1_7 = np.tanh((-2.49683893292792 + 0.398927054757526 * P + 0.0128111795517157 * T))
    H1_8 = np.tanh((-1.34037612046496 + -1.82172820931808 * P + 0.0975510622080853 * T))
    H1_9 = np.tanh((-0.00454329936396595 + -0.0840846337708169 * P + 0.00192700682148055 * T))
    H1_10 = np.tanh((3.25989647210865 + -0.817575328835318 * P + -0.00790145526484188 * T))
    H1_11 = np.tanh((1.2138688950769 + -5.5187467484401 * P + 0.115937604475061 * T))
    H1_12 = np.tanh((-4.2841473959043 + -3.44784008416792 * P + 0.225314720754115 * T))
    H1_13 = np.tanh((-0.749714454876193 + 0.59904718534549 * P + 0.015852118328226 * T))
    H1_14 = np.tanh((-3.56659915773571 + -3.26552973737454 * P + 0.201884080139387 * T))
    H1_15 = np.tanh((-0.368559368031245 + -4.14838409559996 * P + 0.1417710518596 * T))

    h += -245.023690422356 * H1_1
    h += 143.039857861931 * H1_10
    h += 57.936905487342 * H1_11
    h += 141.765833322109 * H1_12
    h += 91.6603988756128 * H1_13
    h += -354.856864121115 * H1_14
    h += -407.462486717321 * H1_15
    h += -762.587308746834 * H1_2
    h += -117.086388339974 * H1_3
    h += 2084.0234411658 * H1_4
    h += 3494.85395981366 * H1_5
    h += -79.8462693619194 * H1_6
    h += -257.346057454727 * H1_7
    h += -2180.32225879755 * H1_8
    h += 3232.88193434466 * H1_9

    # Convert from kJ/kg to J/kg
    h *= 1e3

    return h


def lh2_P(T):
    """
    Pressure of saturated liquid hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Pressure of saturated liquid hydrogen (Pa)
    """
    return 0.0138 * T**5.2644


def lh2_h(T):
    """
    Enthalpy of saturated liquid hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Enthalpy of saturated liquid hydrogen (J/kg)
    """
    return (
        -371985.2
        + 16864.749 * T
        + 893.59208 * (T - 27.6691) ** 2
        + 103.63758 * (T - 27.6691) ** 3
        + 7.756004 * (T - 27.6691) ** 4
    )


def lh2_u(T):
    """
    Internal energy of saturated liquid hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Internal energy of saturated liquid hydrogen (J/kg)
    """
    return (
        -334268
        + 15183.043 * T
        + 614.10133 * (T - 27.6691) ** 2
        + 40.845478 * (T - 27.6691) ** 3
        + 9.1394916 * (T - 27.6691) ** 4
        + 1.8297788 * (T - 27.6691) ** 5
        + 0.1246228 * (T - 27.6691) ** 6
    )


def lh2_cp(T):
    """
    Specific heat at constant pressure of saturated liquid hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Specific heat at constant pressure of saturated liquid hydrogen (J/kg)
    """
    return 1 / (0.0002684 - 7.6143e-6 * T - 2.5759e-7 * (T - 27.6691) ** 2)


def lh2_rho(T):
    """
    Density of saturated liquid hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Density of saturated liquid hydrogen (kg/m^3)
    """
    return (
        115.53291
        - 2.0067591 * T
        - 0.1067411 * (T - 27.6691) ** 2
        - 0.0085915 * (T - 27.6691) ** 3
        - 0.0019879 * (T - 27.6691) ** 4
        - 0.0003988 * (T - 27.6691) ** 5
        - 2.7179e-5 * (T - 27.6691) ** 6
    )


def sat_gh2_rho(T):
    """
    Density of saturated gaseous hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Density of saturated gaseous hydrogen (kg/m^3)
    """
    return (
        -28.97599
        + 1.2864736 * T
        + 0.1140157 * (T - 27.6691) ** 2
        + 0.0086723 * (T - 27.6691) ** 3
        + 0.0019006 * (T - 27.6691) ** 4
        + 0.0003805 * (T - 27.6691) ** 5
        + 2.5918e-5 * (T - 27.6691) ** 6
    )


def sat_gh2_h(T):
    """
    Enthalpy of saturated gaseous hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Enthalpy of saturated gaseous hydrogen (J/kg)
    """
    return (
        577302.07
        - 4284.432 * T
        - 1084.1238 * (T - 27.6691) ** 2
        - 73.011186 * (T - 27.6691) ** 3
        - 15.407809 * (T - 27.6691) ** 4
        - 2.9987887 * (T - 27.6691) ** 5
        - 0.2022147 * (T - 27.6691) ** 6
    )


def sat_gh2_cp(T):
    """
    Enthalpy of saturated gaseous hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Enthalpy of saturated gaseous hydrogen (J/kg)
    """
    return np.exp(
        6.445199
        + 0.1249361 * T
        + 0.0125811 * (T - 27.6691) ** 2
        + 0.0027137 * (T - 27.6691) ** 3
        + 0.0006249 * (T - 27.6691) ** 4
        + 4.8352e-5 * (T - 27.6691) ** 5
    )


def sat_gh2_k(T):
    """
    Thermal conductivity of saturated gaseous hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Thermal conductivity of saturated gaseous hydrogen (W/(m-K))
    """
    return 1 / (110.21937 - 2.6596443 * T - 0.0153377 * (T - 27.6691) ** 2 - 0.0088632 * (T - 27.6691) ** 3)


def sat_gh2_viscosity(T):
    """
    Viscosity of saturated gaseous hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Viscosity of saturated gaseous hydrogen (Pa-s)
    """
    return 1 / (
        1582670.2
        - 34545.242 * T
        - 211.73722 * (T - 27.6691) ** 2
        - 283.70972 * (T - 27.6691) ** 3
        - 18.848797 * (T - 27.6691) ** 4
    )


def sat_gh2_beta(T):
    """
    Coefficient of thermal expansion of saturated gaseous hydrogen.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    T : float or numpy array
        Hydrogen temperature (K)

    Returns
    -------
    float or numpy array
       Coefficient of thermal expansion of saturated gaseous hydrogen (Pa-s)
    """
    return 1 / T


def sat_gh2_T(P):
    """
    Temperature of saturated gaseous hydrogen at the specified pressure.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    The equations are based on data obtained from the NIST chemistry webbook for
    saturated thermophysical properties of hydrogen. Fits were done in JMP software
    at temperatures ranging from 20.369-32 K. See page 100 of the thesis for more
    details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa)

    Returns
    -------
    float or numpy array
       Temperature of saturated gaseous hydrogen (K)
    """
    return (
        22.509518
        + 9.5791e-6 * P
        - 5.85e-12 * (P - 598825) ** 2
        + 3.292e-18 * (P - 598825) ** 3
        - 1.246e-24 * (P - 598825) ** 4
        + 2.053e-29 * (P - 598825) ** 5
        - 3.463e-35 * (P - 598825) ** 6
    )
