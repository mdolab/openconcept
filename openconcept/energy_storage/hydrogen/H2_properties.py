import numpy as np


def gh2_cv(P, T, deriv=False):
    """
    Compute specific heat of hydrogen gas at constant volume.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    It is based on data collected at pressures between 100-750 kPa and 20-150 K.
    See page 100 of the thesis for more details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
    T : float or numpy array
        Hydrogen temperature (K); P and T must be the same shape if they're both arrays
    deriv : bool, optional
        Compute the derivative of the output with respect to P and T instead
        of the output itself, by default False. If this is set to True, there
        will be two return values that are a numpy array if either P or T is also
        an array and a float otherwise.

    Returns
    -------
    float or numpy array
        Specific heat at constant volume of gaseous hydrogen (J/(kg-K)) or
        the derivative with respect to P if deriv is set to True
    float or numpy array
        If deriv is set to True, the derivative of specific heat with respect
        to temperature
    """
    # Check inputs
    if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
        raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

    if deriv:
        # Convert pressure to MPa
        P_MPa = P * 1e-6
        PMPa_P = 1e-6

        H1_PMPa = 0.275976950953691 * (1 - np.tanh((0.3990654435825 + 0.275976950953691 * P_MPa + -0.0364541001467881 * T))**2)
        H1_T = -0.0364541001467881 * (1 - np.tanh((0.3990654435825 + 0.275976950953691 * P_MPa + -0.0364541001467881 * T))**2)
        H2_PMPa = -2.62940079620686 * (1 - np.tanh((-4.86826687998567 + -2.62940079620686 * P_MPa + 0.216404289884235 * T))**2)
        H2_T = 0.216404289884235 * (1 - np.tanh((-4.86826687998567 + -2.62940079620686 * P_MPa + 0.216404289884235 * T))**2)
        H3_PMPa = 0.269571399186571 * (1 - np.tanh((1.12548975067467 + 0.269571399186571 * P_MPa + -0.0349250996335972 * T))**2)
        H3_T = -0.0349250996335972 * (1 - np.tanh((1.12548975067467 + 0.269571399186571 * P_MPa + -0.0349250996335972 * T))**2)
        H4_PMPa = -3.1347851189393 * (1 - np.tanh((-5.57528770123298 + -3.1347851189393 * P_MPa + 0.276378205247149 * T))**2)
        H4_T = 0.276378205247149 * (1 - np.tanh((-5.57528770123298 + -3.1347851189393 * P_MPa + 0.276378205247149 * T))**2)
        H5_PMPa = -0.415187001771689 * (1 - np.tanh((-0.768555052808218 + -0.415187001771689 * P_MPa + 0.0303660811195935 * T))**2)
        H5_T = 0.0303660811195935 * (1 - np.tanh((-0.768555052808218 + -0.415187001771689 * P_MPa + 0.0303660811195935 * T))**2)
        H6_PMPa = -1.67288064390926 * (1 - np.tanh((0.271020942398391 + -1.67288064390926 * P_MPa + 0.0673489649704582 * T))**2)
        H6_T = 0.0673489649704582 * (1 - np.tanh((0.271020942398391 + -1.67288064390926 * P_MPa + 0.0673489649704582 * T))**2)
        H7_PMPa = 0.0083174017874869 * (1 - np.tanh((-1.10142968268782 + 0.0083174017874869 * P_MPa + 0.0108662672630239 * T))**2)
        H7_T = 0.0108662672630239 * (1 - np.tanh((-1.10142968268782 + 0.0083174017874869 * P_MPa + 0.0108662672630239 * T))**2)
        H8_PMPa = 1.17456681024276 * (1 - np.tanh((1.90064900956529 + 1.17456681024276 * P_MPa + -0.0902336799991085 * T))**2)
        H8_T = -0.0902336799991085 * (1 - np.tanh((1.90064900956529 + 1.17456681024276 * P_MPa + -0.0902336799991085 * T))**2)
        H9_PMPa = -0.943525497821755 * (1 - np.tanh((-0.673064219823259 + -0.943525497821755 * P_MPa + 0.0734160926798799 * T))**2)
        H9_T = 0.0734160926798799 * (1 - np.tanh((-0.673064219823259 + -0.943525497821755 * P_MPa + 0.0734160926798799 * T))**2)
        H10_PMPa = -1.99739736255923 * (1 - np.tanh((-1.6061497898911 + -1.99739736255923 * P_MPa + 0.108199114343877 * T))**2)
        H10_T = 0.108199114343877 * (1 - np.tanh((-1.6061497898911 + -1.99739736255923 * P_MPa + 0.108199114343877 * T))**2)
        H11_PMPa = 0.388202467775416 * (1 - np.tanh((1.73534543350612 + 0.388202467775416 * P_MPa + -0.0616088045146524 * T))**2)
        H11_T = -0.0616088045146524 * (1 - np.tanh((1.73534543350612 + 0.388202467775416 * P_MPa + -0.0616088045146524 * T))**2)
        H12_PMPa = 1.3326759135354 * (1 - np.tanh((2.70166390032434 + 1.3326759135354 * P_MPa + -0.118211564739528 * T))**2)
        H12_T = -0.118211564739528 * (1 - np.tanh((2.70166390032434 + 1.3326759135354 * P_MPa + -0.118211564739528 * T))**2)
        H13_PMPa = 9.42488755971789 * (1 - np.tanh((-9.32984593347698 + 9.42488755971789 * P_MPa + 0.0784916020511085 * T))**2)
        H13_T = 0.0784916020511085 * (1 - np.tanh((-9.32984593347698 + 9.42488755971789 * P_MPa + 0.0784916020511085 * T))**2)
        H14_PMPa = 1.13189377766378 * (1 - np.tanh((1.2207418073194 + 1.13189377766378 * P_MPa + -0.0841043542652104 * T))**2)
        H14_T = -0.0841043542652104 * (1 - np.tanh((1.2207418073194 + 1.13189377766378 * P_MPa + -0.0841043542652104 * T))**2)
        H15_PMPa = 1.84291010959674 * (1 - np.tanh((2.58665026763982 + 1.84291010959674 * P_MPa + -0.13970364770731 * T))**2)
        H15_T = -0.13970364770731 * (1 - np.tanh((2.58665026763982 + 1.84291010959674 * P_MPa + -0.13970364770731 * T))**2)

        cv_P = cv_T = 0.0
        cv_P += -70.3037835837473 * H1_PMPa
        cv_T += -70.3037835837473 * H1_T
        cv_P += -3.65980723192549 * H2_PMPa
        cv_T += -3.65980723192549 * H2_T
        cv_P += 16.4087880702894 * H3_PMPa
        cv_T += 16.4087880702894 * H3_T
        cv_P += -3.5451526649205 * H4_PMPa
        cv_T += -3.5451526649205 * H4_T
        cv_P += 1.59237668513045 * H5_PMPa
        cv_T += 1.59237668513045 * H5_T
        cv_P += 16.3356631788945 * H6_PMPa
        cv_T += 16.3356631788945 * H6_T
        cv_P += 2.87332235554376 * H7_PMPa
        cv_T += 2.87332235554376 * H7_T
        cv_P += -32.0695317395356 * H8_PMPa
        cv_T += -32.0695317395356 * H8_T
        cv_P += -162.92972771138 * H9_PMPa
        cv_T += -162.92972771138 * H9_T
        cv_P += -12.804152874995 * H10_PMPa
        cv_T += -12.804152874995 * H10_T
        cv_P += 8.56885046215908 * H11_PMPa
        cv_T += 8.56885046215908 * H11_T
        cv_P += 16.8828056394326 * H12_PMPa
        cv_T += 16.8828056394326 * H12_T
        cv_P += 0.000773883721821057 * H13_PMPa
        cv_T += 0.000773883721821057 * H13_T
        cv_P += -11.6665802642623 * H14_PMPa
        cv_T += -11.6665802642623 * H14_T
        cv_P += -43.8044401310621 * H15_PMPa
        cv_T += -43.8044401310621 * H15_T

        # Convert from J/(g-K) to J/(kg-K)
        cv_P *= 1e3
        cv_T *= 1e3

        # Pressure to MPa factor
        cv_P *= PMPa_P

        return cv_P, cv_T

    # Convert pressure to MPa
    P_MPa = P * 1e-6

    cv = 56.0992565764207

    H1 = np.tanh((0.3990654435825 + 0.275976950953691 * P_MPa + -0.0364541001467881 * T))
    H2 = np.tanh((-4.86826687998567 + -2.62940079620686 * P_MPa + 0.216404289884235 * T))
    H3 = np.tanh((1.12548975067467 + 0.269571399186571 * P_MPa + -0.0349250996335972 * T))
    H4 = np.tanh((-5.57528770123298 + -3.1347851189393 * P_MPa + 0.276378205247149 * T))
    H5 = np.tanh((-0.768555052808218 + -0.415187001771689 * P_MPa + 0.0303660811195935 * T))
    H6 = np.tanh((0.271020942398391 + -1.67288064390926 * P_MPa + 0.0673489649704582 * T))
    H7 = np.tanh((-1.10142968268782 + 0.0083174017874869 * P_MPa + 0.0108662672630239 * T))
    H8 = np.tanh((1.90064900956529 + 1.17456681024276 * P_MPa + -0.0902336799991085 * T))
    H9 = np.tanh((-0.673064219823259 + -0.943525497821755 * P_MPa + 0.0734160926798799 * T))
    H10 = np.tanh((-1.6061497898911 + -1.99739736255923 * P_MPa + 0.108199114343877 * T))
    H11 = np.tanh((1.73534543350612 + 0.388202467775416 * P_MPa + -0.0616088045146524 * T))
    H12 = np.tanh((2.70166390032434 + 1.3326759135354 * P_MPa + -0.118211564739528 * T))
    H13 = np.tanh((-9.32984593347698 + 9.42488755971789 * P_MPa + 0.0784916020511085 * T))
    H14 = np.tanh((1.2207418073194 + 1.13189377766378 * P_MPa + -0.0841043542652104 * T))
    H15 = np.tanh((2.58665026763982 + 1.84291010959674 * P_MPa + -0.13970364770731 * T))

    cv += -70.3037835837473 * H1
    cv += -3.65980723192549 * H2
    cv += 16.4087880702894 * H3
    cv += -3.5451526649205 * H4
    cv += 1.59237668513045 * H5
    cv += 16.3356631788945 * H6
    cv += 2.87332235554376 * H7
    cv += -32.0695317395356 * H8
    cv += -162.92972771138 * H9
    cv += -12.804152874995 * H10
    cv += 8.56885046215908 * H11
    cv += 16.8828056394326 * H12
    cv += 0.000773883721821057 * H13
    cv += -11.6665802642623 * H14
    cv += -43.8044401310621 * H15

    # Convert from J/(g-K) to J/(kg-K)
    cv *= 1e3

    return cv


def gh2_cp(P, T, deriv=False):
    """
    Compute specific heat of hydrogen gas at constant pressure.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    It is based on data collected at pressures between 100-750 kPa and 20-150 K.
    See page 100 of the thesis for more details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
    T : float or numpy array
        Hydrogen temperature (K); P and T must be the same shape if they're both arrays
    deriv : bool, optional
        Compute the derivative of the output with respect to P and T instead
        of the output itself, by default False. If this is set to True, there
        will be two return values that are a numpy array if either P or T is also
        an array and a float otherwise.

    Returns
    -------
    float or numpy array
        Specific heat at constant pressure of gaseous hydrogen (J/(kg-K)) or
        the derivative with respect to P if deriv is set to True
    float or numpy array
        If deriv is set to True, the derivative of specific heat with respect
        to temperature
    """
    # Check inputs
    if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
        raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

    if deriv:
        # Convert pressure to MPa
        P_MPa = P * 1e-6
        PMPa_P = 1e-6

        H1_PMPa = 1.13159708689763 * (1 - np.tanh((0.614735260887242 + 1.13159708689763 * P_MPa + -0.0551837204578247 * T))**2)
        H1_T = -0.0551837204578247 * (1 - np.tanh((0.614735260887242 + 1.13159708689763 * P_MPa + -0.0551837204578247 * T))**2)
        H2_PMPa = -0.0140493655427582 * (1 - np.tanh((-0.400739830511314 + -0.0140493655427582 * P_MPa + -0.000581742517018706 * T))**2)
        H2_T = -0.000581742517018706 * (1 - np.tanh((-0.400739830511314 + -0.0140493655427582 * P_MPa + -0.000581742517018706 * T))**2)
        H3_PMPa = -0.584213145968296 * (1 - np.tanh((-1.25439265281633 + -0.584213145968296 * P_MPa + 0.100382842687856 * T))**2)
        H3_T = 0.100382842687856 * (1 - np.tanh((-1.25439265281633 + -0.584213145968296 * P_MPa + 0.100382842687856 * T))**2)
        H4_PMPa = 1.1389617257106 * (1 - np.tanh((0.613718515601412 + 1.1389617257106 * P_MPa + -0.0766902288518594 * T))**2)
        H4_T = -0.0766902288518594 * (1 - np.tanh((0.613718515601412 + 1.1389617257106 * P_MPa + -0.0766902288518594 * T))**2)
        H5_PMPa = 1.12786935775064 * (1 - np.tanh((0.783863063691068 + 1.12786935775064 * P_MPa + -0.0580710892849312 * T))**2)
        H5_T = -0.0580710892849312 * (1 - np.tanh((0.783863063691068 + 1.12786935775064 * P_MPa + -0.0580710892849312 * T))**2)
        H6_PMPa = 6.11836677910538 * (1 - np.tanh((3.0862845706326 + 6.11836677910538 * P_MPa + -0.28171194186548 * T))**2)
        H6_T = -0.28171194186548 * (1 - np.tanh((3.0862845706326 + 6.11836677910538 * P_MPa + -0.28171194186548 * T))**2)
        H7_PMPa = -1.74531315193444 * (1 - np.tanh((-4.01267608558022 + -1.74531315193444 * P_MPa + 0.13245397217098 * T))**2)
        H7_T = 0.13245397217098 * (1 - np.tanh((-4.01267608558022 + -1.74531315193444 * P_MPa + 0.13245397217098 * T))**2)
        H8_PMPa = -2.69493649415449 * (1 - np.tanh((-4.73597371609159 + -2.69493649415449 * P_MPa + 0.241924487447467 * T))**2)
        H8_T = 0.241924487447467 * (1 - np.tanh((-4.73597371609159 + -2.69493649415449 * P_MPa + 0.241924487447467 * T))**2)
        H9_PMPa = -2.55750416599446 * (1 - np.tanh((-3.13544345816356 + -2.55750416599446 * P_MPa + 0.149566251986045 * T))**2)
        H9_T = 0.149566251986045 * (1 - np.tanh((-3.13544345816356 + -2.55750416599446 * P_MPa + 0.149566251986045 * T))**2)
        H10_PMPa = -4.95468689159338 * (1 - np.tanh((-3.57473472514032 + -4.95468689159338 * P_MPa + 0.281591654527403 * T))**2)
        H10_T = 0.281591654527403 * (1 - np.tanh((-3.57473472514032 + -4.95468689159338 * P_MPa + 0.281591654527403 * T))**2)
        H11_PMPa = 3.38001091079686 * (1 - np.tanh((4.41722437760112 + 3.38001091079686 * P_MPa + -0.242556425504075 * T))**2)
        H11_T = -0.242556425504075 * (1 - np.tanh((4.41722437760112 + 3.38001091079686 * P_MPa + -0.242556425504075 * T))**2)
        H12_PMPa = 0.0072937040171696 * (1 - np.tanh((-0.398001325855848 + 0.0072937040171696 * P_MPa + 0.000277104672759379 * T))**2)
        H12_T = 0.000277104672759379 * (1 - np.tanh((-0.398001325855848 + 0.0072937040171696 * P_MPa + 0.000277104672759379 * T))**2)
        H13_PMPa = 0.576128323850331 * (1 - np.tanh((-0.31379942783082 + 0.576128323850331 * P_MPa + 0.0347928080405698 * T))**2)
        H13_T = 0.0347928080405698 * (1 - np.tanh((-0.31379942783082 + 0.576128323850331 * P_MPa + 0.0347928080405698 * T))**2)
        H14_PMPa = 1.16723768888899 * (1 - np.tanh((7.34692537702712 + 1.16723768888899 * P_MPa + -0.317917699002537 * T))**2)
        H14_T = -0.317917699002537 * (1 - np.tanh((7.34692537702712 + 1.16723768888899 * P_MPa + -0.317917699002537 * T))**2)
        H15_PMPa = 0.784988883943678 * (1 - np.tanh((-0.816554474050802 + 0.784988883943678 * P_MPa + 0.0377876586456731 * T))**2)
        H15_T = 0.0377876586456731 * (1 - np.tanh((-0.816554474050802 + 0.784988883943678 * P_MPa + 0.0377876586456731 * T))**2)

        cp_P = cp_T = 0.0
        cp_P += 346.455492384544 * H1_PMPa
        cp_T += 346.455492384544 * H1_T
        cp_P += -521.915978427339 * H2_PMPa
        cp_T += -521.915978427339 * H2_T
        cp_P += 137.870380603119 * H3_PMPa
        cp_T += 137.870380603119 * H3_T
        cp_P += 452.163820434345 * H4_PMPa
        cp_T += 452.163820434345 * H4_T
        cp_P += -374.519768532003 * H5_PMPa
        cp_T += -374.519768532003 * H5_T
        cp_P += -41.7239727718235 * H6_PMPa
        cp_T += -41.7239727718235 * H6_T
        cp_P += 2.51241979089427 * H7_PMPa
        cp_T += 2.51241979089427 * H7_T
        cp_P += -55.4472053157707 * H8_PMPa
        cp_T += -55.4472053157707 * H8_T
        cp_P += 23.7176680036494 * H9_PMPa
        cp_T += 23.7176680036494 * H9_T
        cp_P += -131.305889878852 * H10_PMPa
        cp_T += -131.305889878852 * H10_T
        cp_P += -87.5906286526561 * H11_PMPa
        cp_T += -87.5906286526561 * H11_T
        cp_P += -881.40560167543 * H12_PMPa
        cp_T += -881.40560167543 * H12_T
        cp_P += -87.1001293092837 * H13_PMPa
        cp_T += -87.1001293092837 * H13_T
        cp_P += -2.24612605221145 * H14_PMPa
        cp_T += -2.24612605221145 * H14_T
        cp_P += 40.3225563600741 * H15_PMPa
        cp_T += 40.3225563600741 * H15_T

        # Convert from J/(g-K) to J/(kg-K)
        cp_P *= 1e3
        cp_T *= 1e3

        # Pressure to MPa factor
        cp_P *= PMPa_P

        return cp_P, cp_T

    # Convert pressure to MPa
    P_MPa = P * 1e-6

    cp = -163.178217071559

    H1 = np.tanh((0.614735260887242 + 1.13159708689763 * P_MPa + -0.0551837204578247 * T))
    H2 = np.tanh((-0.400739830511314 + -0.0140493655427582 * P_MPa + -0.000581742517018706 * T))
    H3 = np.tanh((-1.25439265281633 + -0.584213145968296 * P_MPa + 0.100382842687856 * T))
    H4 = np.tanh((0.613718515601412 + 1.1389617257106 * P_MPa + -0.0766902288518594 * T))
    H5 = np.tanh((0.783863063691068 + 1.12786935775064 * P_MPa + -0.0580710892849312 * T))
    H6 = np.tanh((3.0862845706326 + 6.11836677910538 * P_MPa + -0.28171194186548 * T))
    H7 = np.tanh((-4.01267608558022 + -1.74531315193444 * P_MPa + 0.13245397217098 * T))
    H8 = np.tanh((-4.73597371609159 + -2.69493649415449 * P_MPa + 0.241924487447467 * T))
    H9 = np.tanh((-3.13544345816356 + -2.55750416599446 * P_MPa + 0.149566251986045 * T))
    H10 = np.tanh((-3.57473472514032 + -4.95468689159338 * P_MPa + 0.281591654527403 * T))
    H11 = np.tanh((4.41722437760112 + 3.38001091079686 * P_MPa + -0.242556425504075 * T))
    H12 = np.tanh((-0.398001325855848 + 0.0072937040171696 * P_MPa + 0.000277104672759379 * T))
    H13 = np.tanh((-0.31379942783082 + 0.576128323850331 * P_MPa + 0.0347928080405698 * T))
    H14 = np.tanh((7.34692537702712 + 1.16723768888899 * P_MPa + -0.317917699002537 * T))
    H15 = np.tanh((-0.816554474050802 + 0.784988883943678 * P_MPa + 0.0377876586456731 * T))

    cp += 346.455492384544 * H1
    cp += -521.915978427339 * H2
    cp += 137.870380603119 * H3
    cp += 452.163820434345 * H4
    cp += -374.519768532003 * H5
    cp += -41.7239727718235 * H6
    cp += 2.51241979089427 * H7
    cp += -55.4472053157707 * H8
    cp += 23.7176680036494 * H9
    cp += -131.305889878852 * H10
    cp += -87.5906286526561 * H11
    cp += -881.40560167543 * H12
    cp += -87.1001293092837 * H13
    cp += -2.24612605221145 * H14
    cp += 40.3225563600741 * H15

    # Convert from J/(g-K) to J/(kg-K)
    cp *= 1e3

    return cp


def gh2_u(P, T, deriv=False):
    """
    Compute internal energy of hydrogen gas.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    It is based on data collected at pressures between 100-750 kPa and 20-150 K.
    See page 100 of the thesis for more details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
    T : float or numpy array
        Hydrogen temperature (K); P and T must be the same shape if they're both arrays
    deriv : bool, optional
        Compute the derivative of the output with respect to P and T instead
        of the output itself, by default False. If this is set to True, there
        will be two return values that are a numpy array if either P or T is also
        an array and a float otherwise.

    Returns
    -------
    float or numpy array
        Internal energy of gaseous hydrogen (J/kg) or
        the derivative with respect to P if deriv is set to True
    float or numpy array
        If deriv is set to True, the derivative of specific heat with respect
        to temperature
    """
    # Check inputs
    if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
        raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")
    
    if deriv:
        # Convert pressure to MPa
        P_MPa = P * 1e-6
        PMPa_P = 1e-6

        H1_PMPa = 0.948951297504785 * (1 - np.tanh((0.233539747755315 + 0.948951297504785 * P_MPa + -0.0499745330828293 * T))**2)
        H1_T = -0.0499745330828293 * (1 - np.tanh((0.233539747755315 + 0.948951297504785 * P_MPa + -0.0499745330828293 * T))**2)
        H2_PMPa = 2.13093505892378 * (1 - np.tanh((2.05471470504367 + 2.13093505892378 * P_MPa + -0.119572061260962 * T))**2)
        H2_T = -0.119572061260962 * (1 - np.tanh((2.05471470504367 + 2.13093505892378 * P_MPa + -0.119572061260962 * T))**2)
        H3_PMPa = 0.587703772663913 * (1 - np.tanh((0.398892133197 + 0.587703772663913 * P_MPa + -0.00336363426965824 * T))**2)
        H3_T = -0.00336363426965824 * (1 - np.tanh((0.398892133197 + 0.587703772663913 * P_MPa + -0.00336363426965824 * T))**2)
        H4_PMPa = 0.992347038797521 * (1 - np.tanh((0.143237719165029 + 0.992347038797521 * P_MPa + -0.0143964824374372 * T))**2)
        H4_T = -0.0143964824374372 * (1 - np.tanh((0.143237719165029 + 0.992347038797521 * P_MPa + -0.0143964824374372 * T))**2)
        H5_PMPa = -2.74657042229242 * (1 - np.tanh((-3.96616131435837 + -2.74657042229242 * P_MPa + 0.186453818992668 * T))**2)
        H5_T = 0.186453818992668 * (1 - np.tanh((-3.96616131435837 + -2.74657042229242 * P_MPa + 0.186453818992668 * T))**2)
        H6_PMPa = -0.645489048931109 * (1 - np.tanh((1.44405264544866 + -0.645489048931109 * P_MPa + -0.0104476083443945 * T))**2)
        H6_T = -0.0104476083443945 * (1 - np.tanh((1.44405264544866 + -0.645489048931109 * P_MPa + -0.0104476083443945 * T))**2)
        H7_PMPa = 3.10126879273836 * (1 - np.tanh((2.02006732389655 + 3.10126879273836 * P_MPa + -0.132381179154892 * T))**2)
        H7_T = -0.132381179154892 * (1 - np.tanh((2.02006732389655 + 3.10126879273836 * P_MPa + -0.132381179154892 * T))**2)
        H8_PMPa = -0.295895155010423 * (1 - np.tanh((0.455915191247191 + -0.295895155010423 * P_MPa + -0.00270733720731719 * T))**2)
        H8_T = -0.00270733720731719 * (1 - np.tanh((0.455915191247191 + -0.295895155010423 * P_MPa + -0.00270733720731719 * T))**2)
        H9_PMPa = 0.135982980631872 * (1 - np.tanh((-1.42121486333193 + 0.135982980631872 * P_MPa + 0.0473542542463267 * T))**2)
        H9_T = 0.0473542542463267 * (1 - np.tanh((-1.42121486333193 + 0.135982980631872 * P_MPa + 0.0473542542463267 * T))**2)
        H10_PMPa = -0.206113999100671 * (1 - np.tanh((-0.662630708916202 + -0.206113999100671 * P_MPa + 0.00414455453741901 * T))**2)
        H10_T = 0.00414455453741901 * (1 - np.tanh((-0.662630708916202 + -0.206113999100671 * P_MPa + 0.00414455453741901 * T))**2)
        H11_PMPa = 1.67371525829477 * (1 - np.tanh((0.56740916395517 + 1.67371525829477 * P_MPa + -0.0218895513528701 * T))**2)
        H11_T = -0.0218895513528701 * (1 - np.tanh((0.56740916395517 + 1.67371525829477 * P_MPa + -0.0218895513528701 * T))**2)
        H12_PMPa = 0.1589500143037 * (1 - np.tanh((0.237517777100042 + 0.1589500143037 * P_MPa + -0.0106519272801902 * T))**2)
        H12_T = -0.0106519272801902 * (1 - np.tanh((0.237517777100042 + 0.1589500143037 * P_MPa + -0.0106519272801902 * T))**2)
        H13_PMPa = 0.754447306182754 * (1 - np.tanh((1.54182270996619 + 0.754447306182754 * P_MPa + -0.0541064232047049 * T))**2)
        H13_T = -0.0541064232047049 * (1 - np.tanh((1.54182270996619 + 0.754447306182754 * P_MPa + -0.0541064232047049 * T))**2)
        H14_PMPa = 1.72079624611917 * (1 - np.tanh((2.62921984539398 + 1.72079624611917 * P_MPa + -0.0971068814297633 * T))**2)
        H14_T = -0.0971068814297633 * (1 - np.tanh((2.62921984539398 + 1.72079624611917 * P_MPa + -0.0971068814297633 * T))**2)
        H15_PMPa = -0.484939056845666 * (1 - np.tanh((-1.04212680836281 + -0.484939056845666 * P_MPa + 0.0116518536560464 * T))**2)
        H15_T = 0.0116518536560464 * (1 - np.tanh((-1.04212680836281 + -0.484939056845666 * P_MPa + 0.0116518536560464 * T))**2)

        u_P = u_T = 0.0
        u_P += -1446.64509367172 * H1_PMPa
        u_T += -1446.64509367172 * H1_T
        u_P += 399.518468943879 * H2_PMPa
        u_T += 399.518468943879 * H2_T
        u_P += -333.71693491459 * H3_PMPa
        u_T += -333.71693491459 * H3_T
        u_P += -228.44619912214 * H4_PMPa
        u_T += -228.44619912214 * H4_T
        u_P += 30.515033222315 * H5_PMPa
        u_T += 30.515033222315 * H5_T
        u_P += 74.6722187603707 * H6_PMPa
        u_T += 74.6722187603707 * H6_T
        u_P += -58.2024392498572 * H7_PMPa
        u_T += -58.2024392498572 * H7_T
        u_P += -1594.44683837146 * H8_PMPa
        u_T += -1594.44683837146 * H8_T
        u_P += 69.542870910495 * H9_PMPa
        u_T += 69.542870910495 * H9_T
        u_P += 1174.42287038803 * H10_PMPa
        u_T += 1174.42287038803 * H10_T
        u_P += 26.0663709626282 * H11_PMPa
        u_T += 26.0663709626282 * H11_T
        u_P += 478.510727358678 * H12_PMPa
        u_T += 478.510727358678 * H12_T
        u_P += 213.76235040309 * H13_PMPa
        u_T += 213.76235040309 * H13_T
        u_P += 47.1707429768787 * H14_PMPa
        u_T += 47.1707429768787 * H14_T
        u_P += -14.8049823561287 * H15_PMPa
        u_T += -14.8049823561287 * H15_T

        # Convert from J/(g-K) to J/(kg-K)
        u_P *= 1e3
        u_T *= 1e3

        # Pressure to MPa factor
        u_P *= PMPa_P

        return u_P, u_T

    # Convert pressure to MPa
    P_MPa = P * 1e-6

    u = 673.611983193655

    H1 = np.tanh((0.233539747755315 + 0.948951297504785 * P_MPa + -0.0499745330828293 * T))
    H2 = np.tanh((2.05471470504367 + 2.13093505892378 * P_MPa + -0.119572061260962 * T))
    H3 = np.tanh((0.398892133197 + 0.587703772663913 * P_MPa + -0.00336363426965824 * T))
    H4 = np.tanh((0.143237719165029 + 0.992347038797521 * P_MPa + -0.0143964824374372 * T))
    H5 = np.tanh((-3.96616131435837 + -2.74657042229242 * P_MPa + 0.186453818992668 * T))
    H6 = np.tanh((1.44405264544866 + -0.645489048931109 * P_MPa + -0.0104476083443945 * T))
    H7 = np.tanh((2.02006732389655 + 3.10126879273836 * P_MPa + -0.132381179154892 * T))
    H8 = np.tanh((0.455915191247191 + -0.295895155010423 * P_MPa + -0.00270733720731719 * T))
    H9 = np.tanh((-1.42121486333193 + 0.135982980631872 * P_MPa + 0.0473542542463267 * T))
    H10 = np.tanh((-0.662630708916202 + -0.206113999100671 * P_MPa + 0.00414455453741901 * T))
    H11 = np.tanh((0.56740916395517 + 1.67371525829477 * P_MPa + -0.0218895513528701 * T))
    H12 = np.tanh((0.237517777100042 + 0.1589500143037 * P_MPa + -0.0106519272801902 * T))
    H13 = np.tanh((1.54182270996619 + 0.754447306182754 * P_MPa + -0.0541064232047049 * T))
    H14 = np.tanh((2.62921984539398 + 1.72079624611917 * P_MPa + -0.0971068814297633 * T))
    H15 = np.tanh((-1.04212680836281 + -0.484939056845666 * P_MPa + 0.0116518536560464 * T))

    u += -1446.64509367172 * H1
    u += 399.518468943879 * H2
    u += -333.71693491459 * H3
    u += -228.44619912214 * H4
    u += 30.515033222315 * H5
    u += 74.6722187603707 * H6
    u += -58.2024392498572 * H7
    u += -1594.44683837146 * H8
    u += 69.542870910495 * H9
    u += 1174.42287038803 * H10
    u += 26.0663709626282 * H11
    u += 478.510727358678 * H12
    u += 213.76235040309 * H13
    u += 47.1707429768787 * H14
    u += -14.8049823561287 * H15

    # Convert from kJ/kg to J/kg
    u *= 1e3

    return u


def gh2_h(P, T, deriv=False):
    """
    Compute enthalpy of hydrogen gas.
    This fit is from Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).
    It is based on data collected at pressures between 100-750 kPa and 20-150 K.
    See page 100 of the thesis for more details on the data source and fitting process.

    Parameters
    ----------
    P : float or numpy array
        Hydrogen pressure (Pa); P and T must be the same shape if they're both arrays
    T : float or numpy array
        Hydrogen temperature (K); P and T must be the same shape if they're both arrays
    deriv : bool, optional
        Compute the derivative of the output with respect to P and T instead
        of the output itself, by default False. If this is set to True, there
        will be two return values that are a numpy array if either P or T is also
        an array and a float otherwise.

    Returns
    -------
    float or numpy array
       Enthalpy of gaseous hydrogen (J/kg) or
        the derivative with respect to P if deriv is set to True
    float or numpy array
        If deriv is set to True, the derivative of specific heat with respect
        to temperature
    """
    # Check inputs
    if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
        raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")

    if deriv:
        # Convert pressure to MPa
        P_MPa = P * 1e-6
        PMPa_P = 1e-6

        H1_PMPa = -0.511704824190866 * (1 - np.tanh((1.73579390297524 + -0.511704824190866 * P_MPa + -0.00328063298246929 * T))**2)
        H1_T = -0.00328063298246929 * (1 - np.tanh((1.73579390297524 + -0.511704824190866 * P_MPa + -0.00328063298246929 * T))**2)
        H2_PMPa = -0.0787067450602815 * (1 - np.tanh((-0.17506458742044 + -0.0787067450602815 * P_MPa + 0.0043943295092826 * T))**2)
        H2_T = 0.0043943295092826 * (1 - np.tanh((-0.17506458742044 + -0.0787067450602815 * P_MPa + 0.0043943295092826 * T))**2)
        H3_PMPa = -2.75869130445277 * (1 - np.tanh((-1.49793344088779 + -2.75869130445277 * P_MPa + 0.110698824574587 * T))**2)
        H3_T = 0.110698824574587 * (1 - np.tanh((-1.49793344088779 + -2.75869130445277 * P_MPa + 0.110698824574587 * T))**2)
        H4_PMPa = 0.133730746273731 * (1 - np.tanh((-1.37669485784553 + 0.133730746273731 * P_MPa + 0.00649135107315252 * T))**2)
        H4_T = 0.00649135107315252 * (1 - np.tanh((-1.37669485784553 + 0.133730746273731 * P_MPa + 0.00649135107315252 * T))**2)
        H5_PMPa = -1.88672855275111 * (1 - np.tanh((-1.1405914153055 + -1.88672855275111 * P_MPa + 0.0981301522587858 * T))**2)
        H5_T = 0.0981301522587858 * (1 - np.tanh((-1.1405914153055 + -1.88672855275111 * P_MPa + 0.0981301522587858 * T))**2)
        H6_PMPa = 0.429585607469454 * (1 - np.tanh((0.446124093572671 + 0.429585607469454 * P_MPa + -0.0285787427235227 * T))**2)
        H6_T = -0.0285787427235227 * (1 - np.tanh((0.446124093572671 + 0.429585607469454 * P_MPa + -0.0285787427235227 * T))**2)
        H7_PMPa = 0.398927054757526 * (1 - np.tanh((-2.49683893292792 + 0.398927054757526 * P_MPa + 0.0128111795517157 * T))**2)
        H7_T = 0.0128111795517157 * (1 - np.tanh((-2.49683893292792 + 0.398927054757526 * P_MPa + 0.0128111795517157 * T))**2)
        H8_PMPa = -1.82172820931808 * (1 - np.tanh((-1.34037612046496 + -1.82172820931808 * P_MPa + 0.0975510622080853 * T))**2)
        H8_T = 0.0975510622080853 * (1 - np.tanh((-1.34037612046496 + -1.82172820931808 * P_MPa + 0.0975510622080853 * T))**2)
        H9_PMPa = -0.0840846337708169 * (1 - np.tanh((-0.00454329936396595 + -0.0840846337708169 * P_MPa + 0.00192700682148055 * T))**2)
        H9_T = 0.00192700682148055 * (1 - np.tanh((-0.00454329936396595 + -0.0840846337708169 * P_MPa + 0.00192700682148055 * T))**2)
        H10_PMPa = -0.817575328835318 * (1 - np.tanh((3.25989647210865 + -0.817575328835318 * P_MPa + -0.00790145526484188 * T))**2)
        H10_T = -0.00790145526484188 * (1 - np.tanh((3.25989647210865 + -0.817575328835318 * P_MPa + -0.00790145526484188 * T))**2)
        H11_PMPa = -5.5187467484401 * (1 - np.tanh((1.2138688950769 + -5.5187467484401 * P_MPa + 0.115937604475061 * T))**2)
        H11_T = 0.115937604475061 * (1 - np.tanh((1.2138688950769 + -5.5187467484401 * P_MPa + 0.115937604475061 * T))**2)
        H12_PMPa = -3.44784008416792 * (1 - np.tanh((-4.2841473959043 + -3.44784008416792 * P_MPa + 0.225314720754115 * T))**2)
        H12_T = 0.225314720754115 * (1 - np.tanh((-4.2841473959043 + -3.44784008416792 * P_MPa + 0.225314720754115 * T))**2)
        H13_PMPa = 0.59904718534549 * (1 - np.tanh((-0.749714454876193 + 0.59904718534549 * P_MPa + 0.015852118328226 * T))**2)
        H13_T = 0.015852118328226 * (1 - np.tanh((-0.749714454876193 + 0.59904718534549 * P_MPa + 0.015852118328226 * T))**2)
        H14_PMPa = -3.26552973737454 * (1 - np.tanh((-3.56659915773571 + -3.26552973737454 * P_MPa + 0.201884080139387 * T))**2)
        H14_T = 0.201884080139387 * (1 - np.tanh((-3.56659915773571 + -3.26552973737454 * P_MPa + 0.201884080139387 * T))**2)
        H15_PMPa = -4.14838409559996 * (1 - np.tanh((-0.368559368031245 + -4.14838409559996 * P_MPa + 0.1417710518596 * T))**2)
        H15_T = 0.1417710518596 * (1 - np.tanh((-0.368559368031245 + -4.14838409559996 * P_MPa + 0.1417710518596 * T))**2)

        h_P = h_T = 0.0
        h_P += -245.023690422356 * H1_PMPa
        h_T += -245.023690422356 * H1_T
        h_P += -762.587308746834 * H2_PMPa
        h_T += -762.587308746834 * H2_T
        h_P += -117.086388339974 * H3_PMPa
        h_T += -117.086388339974 * H3_T
        h_P += 2084.0234411658 * H4_PMPa
        h_T += 2084.0234411658 * H4_T
        h_P += 3494.85395981366 * H5_PMPa
        h_T += 3494.85395981366 * H5_T
        h_P += -79.8462693619194 * H6_PMPa
        h_T += -79.8462693619194 * H6_T
        h_P += -257.346057454727 * H7_PMPa
        h_T += -257.346057454727 * H7_T
        h_P += -2180.32225879755 * H8_PMPa
        h_T += -2180.32225879755 * H8_T
        h_P += 3232.88193434466 * H9_PMPa
        h_T += 3232.88193434466 * H9_T
        h_P += 143.039857861931 * H10_PMPa
        h_T += 143.039857861931 * H10_T
        h_P += 57.936905487342 * H11_PMPa
        h_T += 57.936905487342 * H11_T
        h_P += 141.765833322109 * H12_PMPa
        h_T += 141.765833322109 * H12_T
        h_P += 91.6603988756128 * H13_PMPa
        h_T += 91.6603988756128 * H13_T
        h_P += -354.856864121115 * H14_PMPa
        h_T += -354.856864121115 * H14_T
        h_P += -407.462486717321 * H15_PMPa
        h_T += -407.462486717321 * H15_T

        # Convert from J/(g-K) to J/(kg-K)
        h_P *= 1e3
        h_T *= 1e3

        # Pressure to MPa factor
        h_P *= PMPa_P

        return h_P, h_T

    # Convert pressure to MPa
    P_MPa = P * 1e-6

    h = 1281.75444728572

    H1 = np.tanh((1.73579390297524 + -0.511704824190866 * P_MPa + -0.00328063298246929 * T))
    H2 = np.tanh((-0.17506458742044 + -0.0787067450602815 * P_MPa + 0.0043943295092826 * T))
    H3 = np.tanh((-1.49793344088779 + -2.75869130445277 * P_MPa + 0.110698824574587 * T))
    H4 = np.tanh((-1.37669485784553 + 0.133730746273731 * P_MPa + 0.00649135107315252 * T))
    H5 = np.tanh((-1.1405914153055 + -1.88672855275111 * P_MPa + 0.0981301522587858 * T))
    H6 = np.tanh((0.446124093572671 + 0.429585607469454 * P_MPa + -0.0285787427235227 * T))
    H7 = np.tanh((-2.49683893292792 + 0.398927054757526 * P_MPa + 0.0128111795517157 * T))
    H8 = np.tanh((-1.34037612046496 + -1.82172820931808 * P_MPa + 0.0975510622080853 * T))
    H9 = np.tanh((-0.00454329936396595 + -0.0840846337708169 * P_MPa + 0.00192700682148055 * T))
    H10 = np.tanh((3.25989647210865 + -0.817575328835318 * P_MPa + -0.00790145526484188 * T))
    H11 = np.tanh((1.2138688950769 + -5.5187467484401 * P_MPa + 0.115937604475061 * T))
    H12 = np.tanh((-4.2841473959043 + -3.44784008416792 * P_MPa + 0.225314720754115 * T))
    H13 = np.tanh((-0.749714454876193 + 0.59904718534549 * P_MPa + 0.015852118328226 * T))
    H14 = np.tanh((-3.56659915773571 + -3.26552973737454 * P_MPa + 0.201884080139387 * T))
    H15 = np.tanh((-0.368559368031245 + -4.14838409559996 * P_MPa + 0.1417710518596 * T))

    h += -245.023690422356 * H1
    h += -762.587308746834 * H2
    h += -117.086388339974 * H3
    h += 2084.0234411658 * H4
    h += 3494.85395981366 * H5
    h += -79.8462693619194 * H6
    h += -257.346057454727 * H7
    h += -2180.32225879755 * H8
    h += 3232.88193434466 * H9
    h += 143.039857861931 * H10
    h += 57.936905487342 * H11
    h += 141.765833322109 * H12
    h += 91.6603988756128 * H13
    h += -354.856864121115 * H14
    h += -407.462486717321 * H15

    # Convert from kJ/kg to J/kg
    h *= 1e3

    return h


def lh2_P(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Pressure of saturated liquid hydrogen (Pa) or the derivative with
       respect to T if deriv is set to True
    """
    if deriv:
        return 5.2644 * 0.0138 * T**4.2644
    return 0.0138 * T**5.2644


def lh2_h(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Enthalpy of saturated liquid hydrogen (J/kg) or the derivative with respect
       to T if deriv is set to True
    """
    if deriv:
        return (
            + 16864.749
            + 2 * 893.59208 * (T - 27.6691)
            + 3 * 103.63758 * (T - 27.6691) ** 2
            + 4 * 7.756004 * (T - 27.6691) ** 3
        )
    return (
        -371985.2
        + 16864.749 * T
        + 893.59208 * (T - 27.6691) ** 2
        + 103.63758 * (T - 27.6691) ** 3
        + 7.756004 * (T - 27.6691) ** 4
    )


def lh2_u(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Internal energy of saturated liquid hydrogen (J/kg) or the derivative with
       respect to T if deriv is set to True
    """
    if deriv:
        return (
            + 15183.043
            + 2 * 614.10133 * (T - 27.6691)
            + 3 * 40.845478 * (T - 27.6691) ** 2
            + 4 * 9.1394916 * (T - 27.6691) ** 3
            + 5 * 1.8297788 * (T - 27.6691) ** 4
            + 6 * 0.1246228 * (T - 27.6691) ** 5
        )
    return (
        -334268
        + 15183.043 * T
        + 614.10133 * (T - 27.6691) ** 2
        + 40.845478 * (T - 27.6691) ** 3
        + 9.1394916 * (T - 27.6691) ** 4
        + 1.8297788 * (T - 27.6691) ** 5
        + 0.1246228 * (T - 27.6691) ** 6
    )


def lh2_cp(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Specific heat at constant pressure of saturated liquid hydrogen (J/(kg-K)) or the
       derivative with respect to T if deriv is set to True
    """
    if deriv:
        return -(-7.6143e-6 - 2 * 2.5759e-7 * (T - 27.6691)) / (0.0002684 - 7.6143e-6 * T - 2.5759e-7 * (T - 27.6691) ** 2)**2
    return 1 / (0.0002684 - 7.6143e-6 * T - 2.5759e-7 * (T - 27.6691) ** 2)


def lh2_rho(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Density of saturated liquid hydrogen (kg/m^3) or the derivative with respect
       to T if deriv is set to True
    """
    if deriv:
        return (
            - 2.0067591
            - 2 * 0.1067411 * (T - 27.6691)
            - 3 * 0.0085915 * (T - 27.6691) ** 2
            - 4 * 0.0019879 * (T - 27.6691) ** 3
            - 5 * 0.0003988 * (T - 27.6691) ** 4
            - 6 * 2.7179e-5 * (T - 27.6691) ** 5
        )
    return (
        115.53291
        - 2.0067591 * T
        - 0.1067411 * (T - 27.6691) ** 2
        - 0.0085915 * (T - 27.6691) ** 3
        - 0.0019879 * (T - 27.6691) ** 4
        - 0.0003988 * (T - 27.6691) ** 5
        - 2.7179e-5 * (T - 27.6691) ** 6
    )


def sat_gh2_rho(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Density of saturated gaseous hydrogen (kg/m^3) or the derivative with respect
       to T if deriv is set to True
    """
    if deriv:
        return (
            + 1.2864736
            + 2 * 0.1140157 * (T - 27.6691)
            + 3 * 0.0086723 * (T - 27.6691) ** 2
            + 4 * 0.0019006 * (T - 27.6691) ** 3
            + 5 * 0.0003805 * (T - 27.6691) ** 4
            + 6 * 2.5918e-5 * (T - 27.6691) ** 5
        )
    return (
        -28.97599
        + 1.2864736 * T
        + 0.1140157 * (T - 27.6691) ** 2
        + 0.0086723 * (T - 27.6691) ** 3
        + 0.0019006 * (T - 27.6691) ** 4
        + 0.0003805 * (T - 27.6691) ** 5
        + 2.5918e-5 * (T - 27.6691) ** 6
    )


def sat_gh2_h(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Enthalpy of saturated gaseous hydrogen (J/kg) or the derivative with respect
       to T if deriv is set to True
    """
    if deriv:
        return (
            - 4284.432
            - 2 * 1084.1238 * (T - 27.6691)
            - 3 * 73.011186 * (T - 27.6691) ** 2
            - 4 * 15.407809 * (T - 27.6691) ** 3
            - 5 * 2.9987887 * (T - 27.6691) ** 4
            - 6 * 0.2022147 * (T - 27.6691) ** 5
        )
    return (
        577302.07
        - 4284.432 * T
        - 1084.1238 * (T - 27.6691) ** 2
        - 73.011186 * (T - 27.6691) ** 3
        - 15.407809 * (T - 27.6691) ** 4
        - 2.9987887 * (T - 27.6691) ** 5
        - 0.2022147 * (T - 27.6691) ** 6
    )


def sat_gh2_cp(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Enthalpy of saturated gaseous hydrogen (J/kg) or the derivative with respect
       to T if deriv is set to True
    """
    if deriv:
        return np.exp(
            6.445199
            + 0.1249361 * T
            + 0.0125811 * (T - 27.6691) ** 2
            + 0.0027137 * (T - 27.6691) ** 3
            + 0.0006249 * (T - 27.6691) ** 4
            + 4.8352e-5 * (T - 27.6691) ** 5
        ) * (
            + 0.1249361
            + 2 * 0.0125811 * (T - 27.6691)
            + 3 * 0.0027137 * (T - 27.6691) ** 2
            + 4 * 0.0006249 * (T - 27.6691) ** 3
            + 5 * 4.8352e-5 * (T - 27.6691) ** 4
        )
    return np.exp(
        6.445199
        + 0.1249361 * T
        + 0.0125811 * (T - 27.6691) ** 2
        + 0.0027137 * (T - 27.6691) ** 3
        + 0.0006249 * (T - 27.6691) ** 4
        + 4.8352e-5 * (T - 27.6691) ** 5
    )


def sat_gh2_k(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Thermal conductivity of saturated gaseous hydrogen (W/(m-K)) or the derivative
       with respect to T if deriv is set to True
    """
    if deriv:
        return -(-2.6596443 - 2 * 0.0153377 * (T - 27.6691) - 3 * 0.0088632 * (T - 27.6691) ** 2) / (110.21937 - 2.6596443 * T - 0.0153377 * (T - 27.6691) ** 2 - 0.0088632 * (T - 27.6691) ** 3)**2
    return 1 / (110.21937 - 2.6596443 * T - 0.0153377 * (T - 27.6691) ** 2 - 0.0088632 * (T - 27.6691) ** 3)


def sat_gh2_viscosity(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Viscosity of saturated gaseous hydrogen (Pa-s) or the derivative with respect
       to T if deriv is set to True
    """
    if deriv:
        return -(
            - 34545.242
            - 2 * 211.73722 * (T - 27.6691)
            - 3 * 283.70972 * (T - 27.6691) ** 2
            - 4 * 18.848797 * (T - 27.6691) ** 3
        ) / (
            1582670.2
            - 34545.242 * T
            - 211.73722 * (T - 27.6691) ** 2
            - 283.70972 * (T - 27.6691) ** 3
            - 18.848797 * (T - 27.6691) ** 4
        )**2
    return 1 / (
        1582670.2
        - 34545.242 * T
        - 211.73722 * (T - 27.6691) ** 2
        - 283.70972 * (T - 27.6691) ** 3
        - 18.848797 * (T - 27.6691) ** 4
    )


def sat_gh2_beta(T, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to T instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Coefficient of thermal expansion of saturated gaseous hydrogen (1 / K) or the
       derivative with respect to T if deriv is set to True
    """
    if deriv:
        return -1 / T**2
    return 1 / T


def sat_gh2_T(P, deriv=False):
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
    deriv : bool, optional
        Compute the derivative of the output with respect to P instead
        of the output itself, by default False

    Returns
    -------
    float or numpy array
       Temperature of saturated gaseous hydrogen (K) or the derivative with respect
       to P if deriv is set to True
    """
    is_float = not isinstance(P, np.ndarray)
    if is_float:
        P = np.array([P])

    if deriv:
        val = (
            + 9.5791e-6
            - 2 * 5.85e-12 * (P - 598825)
            + 3 * 3.292e-18 * (P - 598825) ** 2
            - 4 * 1.246e-24 * (P - 598825) ** 3
            + 5 * 2.053e-29 * (P - 598825) ** 4
            - 6 * 3.463e-35 * (P - 598825) ** 5
        )
        val[P > 1235172] = 0.0
        return val.item() if is_float else val
    val = (
        22.509518
        + 9.5791e-6 * P
        - 5.85e-12 * (P - 598825) ** 2
        + 3.292e-18 * (P - 598825) ** 3
        - 1.246e-24 * (P - 598825) ** 4
        + 2.053e-29 * (P - 598825) ** 5
        - 3.463e-35 * (P - 598825) ** 6
    )

    # The curve fit isn't meant to be used at very high pressures, so just use the
    # max value if a high pressure is provided. The solution shouldn't use this value,
    # but it may be requested in the middle of a nonlinear solution.
    val[P > 1235172] = 32.459
    return val.item() if is_float else val
