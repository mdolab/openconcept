import numpy as np
import scipy.sparse as sp
from openmdao.api import ExplicitComponent
import warnings


def bdf3_cache_matrix(n, all_bdf=False):
    """
    This implements the base block Jacobian of the BDF3 method.
    BDF3 is third order accurate and suitable for stiff systems.

    The first couple of points are handled by 3rd-order offset finite difference stencils.
    """
    """
    Any multistep method can be posed as the following:

    [A] y = h[B] y'

    Where A and B are both N-1 rows by N columns (since y(0) aka y1 is already determined as the initial condition).
    h is a time step.
    Remove the first COLUMN of both matrices to obtain N-1 by N-1 matrices [a] and [b].
    The first columns are [av] and [bv] which are both N-1 by 1.
    The system can then be expressed as: [a] {y2-yN} + [av] y1 = [b] {y'2-y'N} + [bv] y'1
    We can then obtain a closed-form expression for {y2-yN} (the unknown states) as follows:
    {y2-yN} = h inv([a]) [b] {y'2-y'N} + h inv([a]) [bv] y'1 - inv([a]) [av] y1

    The last quantity inv([a]) [av] always turns out as just ones
    (since all states are equally linearly dependent on the initial condition).

    We can then solve for the entire state vector {y1-yN} by constructing an N x N block matrix with:
    All zeros in the first row (as y1 cannot depend on anything else)
    inv([a]) [bv] in the first column (to capture the y'1 dependency, if any)
    inv([a]) [b] in the lower right Nx1 by Nx1 squares

    The final form is:
    y = h [M] y' + [ones] y(0)
    where
            _____1_____________N-1__________
    [M] = 1 |___0____________|____0...._____|
            |  inv([a])[bv] |    inv([a])[b]|
        N-1 |..             |               |
            |.._____________|_______________|

    In this case, bv is all zeros because BDF has no dependence on y1'
    In the event that the method is being applied across multiple subintervals, a generally lower-triangular matrix will need to be constructed.
    The [M] matrix for each subinterval * h will go on the block diagonals.
    Any block diagonals below will need to be filled in with dense matrices consisting of the LAST row ([M] * h) repeated over and over again.
    It will look like this:

    [Big Matrix] =  ______ N1_______|________N2______|_______N3_____|
                 N1 |____[M] * h1___|____zeros_______|_____zeros____|
                 N2 |__last row of_1|___[M] * h2_____|_____zeros____|
                 N3 |__last row of_1|__last_row_of_2_|___[M] * h3___|

    Since the first row of [M] is completely blank, this basically means that the FIRST point of each subinterval is equal to the LAST point of the prior one.

    """
    # construct [a] and [b] matrices for a BDF3 scheme with 3rd order finite difference for the first two derivatives
    # the FULL [A] matrix looks like:
    # -1/3 | -1/2    1     -1/6  0 ......
    #  1/6 |  -1    1/2    1/3  0 ......
    # -2/11| 9/11 -18/11    1   0 ......
    #  0   | -2/11  9/11 -18/11  1 0......
    #  0   |   0     -2/11 9/11  -18/11 .... and so on

    # the full [B] matrix looks like:
    #  0  |  1  0   0 ...
    #  0  |  0  1   0 ....
    #  0  |  0  0  6/11 ....
    #  0  |  0  0   0   6/11 0 ..... and so on

    # the all_bdf stencil bootstrps the first two points with BDF1 (backward euler) and BDF2 respectively.
    if all_bdf:
        a_diag_1 = np.zeros((n - 1,))
        # a_diag_1[0] = 1/2
        a_diag_2 = np.ones((n - 1,))
        # a_diag_2[0] = 0
        a_diag_2[0] = 1
        a_diag_3 = np.ones((n - 1,)) * -18 / 11
        a_diag_3[0] = -4 / 3
        a_diag_4 = np.ones((n - 1,)) * 9 / 11
        a_diag_5 = np.ones((n - 1,)) * -2 / 11
        A = sp.diags(
            [a_diag_1, a_diag_2, a_diag_3, a_diag_4, a_diag_5], [1, 0, -1, -2, -3], shape=(n - 1, n - 1)
        ).asformat("csc")
        b_diag = np.ones((n - 1,)) * 6 / 11
        b_diag[0] = 1
        b_diag[1] = 2 / 3
    else:
        # otherwise use a full third order stencil as described in the ASCII art above
        a_diag_0 = np.zeros((n - 1,))
        a_diag_0[0] = -1 / 6
        a_diag_1 = np.zeros((n - 1,))
        a_diag_1[0] = 1
        a_diag_1[1] = 1 / 3
        a_diag_2 = np.ones((n - 1,))
        a_diag_2[0] = -1 / 2
        a_diag_2[1] = 1 / 2
        a_diag_3 = np.ones((n - 1,)) * -18 / 11
        a_diag_3[0] = -1
        a_diag_4 = np.ones((n - 1,)) * 9 / 11
        a_diag_5 = np.ones((n - 1,)) * -2 / 11
        A = sp.diags(
            [a_diag_0, a_diag_1, a_diag_2, a_diag_3, a_diag_4, a_diag_5], [2, 1, 0, -1, -2, -3], shape=(n - 1, n - 1)
        ).asformat("csc")

        b_diag = np.ones((n - 1,)) * 6 / 11
        b_diag[0] = 1
        b_diag[1] = 1
    B = sp.diags([b_diag], [0])
    # C is the base Jacobian matrix
    C = sp.linalg.inv(A).dot(B)
    # we need to offset the entire thing by one row (because the first quantity Q1 is given as an initial condition)
    # and one column (because we do not make use of the initial derivative dQdt1, as this is a stiff method)
    # this is the same as saying that Bv = 0
    C = C.asformat("csr")
    indices = C.nonzero()
    # the main lower triangular-ish matrix:
    tri_mat = sp.csc_matrix((C.data, (indices[0] + 1, indices[1] + 1)))
    # we need to create a dense matrix of the last row repeated n times for multi-subinterval problems
    last_row = tri_mat.getrow(-1).toarray()
    # but we need it in sparse format for openMDAO
    repeat_mat = sp.csc_matrix(np.tile(last_row, n).reshape(n, n))
    return tri_mat, repeat_mat


def simpson_cache_matrix(n):
    # Simpsons rule defines the "deltas" between each segment as [B] dqdt as follows
    # B is n-1 rows by n columns
    # the structure of this is (1/12) * the following:
    # 5 8 -1
    # -1 8 5
    #      5 8 -1
    #      -1 8 5
    #           5 8 -1
    #           -1 8 5    and so on
    # the row indices are basically 0 0 0 1 1 1 2 2 2 ....
    jacmat_rowidx = np.repeat(np.arange((n - 1)), 3)
    # the column indices are 0 1 2 0 1 2 2 3 4 2 3 4 4 5 6 and so on
    # so superimpose a 0 1 2 repeating pattern on a 0 0 0 0 0 0 2 2 2 2 2 2 2 repeating pattern
    jacmat_colidx = np.repeat(np.arange(0, (n - 1), 2), 6) + np.tile(np.arange(3), (n - 1))
    jacmat_data = np.tile(np.array([5, 8, -1, -1, 8, 5]) / 12, (n - 1) // 2)
    jacmat_base = sp.csr_matrix((jacmat_data, (jacmat_rowidx, jacmat_colidx)))
    b = jacmat_base[:, 1:]
    bv = jacmat_base[:, 0]

    a = sp.diags([-1, 1], [-1, 0], shape=(n - 1, n - 1)).asformat("csc")

    ia = sp.linalg.inv(a)
    c = ia.dot(b)
    cv = ia.dot(bv)
    first_row_zeros = sp.csr_matrix(np.zeros((1, n - 1)))
    tri_mat = sp.bmat([[None, first_row_zeros], [cv, c]])

    # we need to create a dense matrix of the last row repeated n times for multi-subinterval problems
    last_row = tri_mat.getrow(-1).toarray()
    # but we need it in sparse format for openMDAO
    repeat_mat = sp.csc_matrix(np.tile(last_row, n).reshape(n, n))
    return tri_mat, repeat_mat


def multistep_integrator(q0, dqdt, dts, tri_mat, repeat_mat, segment_names=None, segments_to_count=None, partials=True):
    """
    This implements the base block Jacobian of the BDF3 method.
    BDF3 is third order accurate and suitable for stiff systems.
    A central-difference approximation and BDF2 are used for the first couple of points,
    so strictly speaking this method is only second order accurate.
    """
    n = int(len(dqdt) / len(dts))

    n_segments = len(dts)
    row_list = []
    for i in range(n_segments):
        col_list = []
        for j in range(n_segments):
            dt = dts[j]
            count_col = True
            if segment_names is not None and segments_to_count is not None:
                if segment_names[j] not in segments_to_count:
                    # skip col IFF not counting this segment
                    count_col = False
            if i > j and count_col:
                # repeat mat
                col_list.append(repeat_mat * dt)
            elif i == j and count_col:
                # diagonal
                col_list.append(tri_mat * dt)
            else:
                col_list.append(sp.csr_matrix(([], ([], [])), shape=(n, n)))
        row_list.append(col_list)
    dQdqdt = sp.bmat(row_list).asformat("csr")
    if not partials:
        Q = dQdqdt.dot(dqdt) + q0
        return Q

    # compute dQ / d dt for each segment
    dt_partials_list = []
    for j in range(n_segments):
        count_col = True
        if segment_names is not None and segments_to_count is not None:
            if segment_names[j] not in segments_to_count:
                # skip col IFF not counting this segment
                count_col = False
        # jth segment
        row_list = []
        for i in range(n_segments):
            # ith row
            if i > j and count_col:
                row_list.append([repeat_mat])
            elif i == j and count_col:
                row_list.append([tri_mat])
            else:
                row_list.append([sp.csr_matrix(([], ([], [])), shape=(n, n))])
        dQddt = sp.bmat(row_list).dot(dqdt[j * n : (j + 1) * n])
        dt_partials_list.append(sp.csr_matrix(dQddt).transpose())

    return dQdqdt, dt_partials_list


# def three_point_lagrange_integration(dqdt, dts, num_segments=1, num_intervals=2,):
#     """This method integrates a rate over time using a 3 point Lagrange interpolant
#     Similar to Simpson's rule except extended to provide increments at every subinterval

#     The number of points per segment nn_seg = (2 * num_intervals + 1)
#     The total number of points is nn_tot = nn_seg * num_segments

#     Inputs
#     ------
#     dqdt : float
#         The rate dqdt to integrate into quantity q (vector, length nn_tot)
#     dts : list
#         A list of timesteps dt corresponding to each interval (length num_intervals)
#     num_segments : int
#         The number of segments to integrate with different dts
#     num_intervals : int
#         The number of Simpson / 3 point quadrature intervals per segment

#     Returns
#     -------
#     delta_q : float
#         Amount of q accumulated during each interval (vector, length num_segments * (nn_seg - 1)
#     partials_wrt_dqdt : float
#         The Jacobian of delta_q with respect to the input rate dqdt
#         The result is a sparse matrix with num_segments * (nn_seg - 1) rows and nn_tot columns
#     partials_wrt_dts : list
#         A list of the Jacobians of delta_q with respect to the time steps
#         There will be one sparse vector with num_segments * (nn_seg - 1) rows per segment
#         But only (nn_seg - 1) rows will actually be populated
#     """
#     nn_seg = (2 * num_intervals + 1)
#     ndelta_seg = 2 * num_intervals
#     nn_tot = nn_seg * num_segments
#     ndelta_tot = ndelta_seg * num_segments

#     if len(dqdt) != nn_tot:
#         raise ValueError('dqdt must be of the correct length. dqdt is of length ' + str(len(dqdt)) +
#                          ' the number of nodes should be' + str(nn_tot))

#     if len(dts) != num_segments:
#         raise ValueError('must provide same number of dts as segments')

#     # first let us construct the basic three point quadrature jacobian which will be
#     # multiplied by the timesteps to obtain the block matrices for the overall jacobian

#     # the structure of this is (1/12) * the following:
#     # 5 8 -1
#     # -1 8 5
#     #      5 8 -1
#     #      -1 8 5
#     #           5 8 -1
#     #           -1 8 5    and so on

#     # the row indices are basically 0 0 0 1 1 1 2 2 2 ....
#     jacmat_rowidx = np.repeat(np.arange(ndelta_seg), 3)
#     # the column indices are 0 1 2 0 1 2 2 3 4 2 3 4 4 5 6 and so on
#     # so superimpose a 0 1 2 repeating pattern on a 0 0 0 0 0 0 2 2 2 2 2 2 2 repeating pattern
#     jacmat_colidx = np.repeat(np.arange(0, ndelta_seg, 2), 6) + np.tile(np.arange(3), ndelta_seg)
#     jacmat_data = np.tile(np.array([5, 8, -1, -1, 8, 5]) / 12, ndelta_seg // 2)
#     jacmat_base = sp.csr_matrix((jacmat_data, (jacmat_rowidx, jacmat_colidx)))

#     jacmats_list = []
#     partials_wrt_dts = []
#     for i_seg in range(num_segments):
#         jacmats_list.append(jacmat_base * dts[i_seg])
#         # get the vector of partials of q with respect to this time step
#         dt_partials = jacmat_base.dot(dqdt[i_seg * nn_seg: (i_seg + 1) * nn_seg])
#         # offset the sparse partials if not the first segment to make it work in OpenMDAO terms
#         dt_partials_rowidxs = np.arange(i_seg * ndelta_seg, (i_seg + 1) * ndelta_seg)
#         dt_partials_colidxs = np.zeros((ndelta_seg,), dtype=np.int32)
#         partials_wrt_dts.append(sp.csr_matrix((dt_partials,
#                                               (dt_partials_rowidxs, dt_partials_colidxs)),
#                                                shape=(ndelta_tot, nn_tot)))
#     # now assemble the overall sparse block diagonal matrix to obtain the final result
#     partials_wrt_dqdt = sp.block_diag(jacmats_list)
#     delta_q = partials_wrt_dqdt.dot(dqdt)

#     return delta_q, partials_wrt_dqdt, partials_wrt_dts

# def trapezoid_integration(dqdt, dts, num_segments=1, num_intervals=2,):
#     """This method integrates a rate over time using a 2 point Trapezoid rule
#     For now this component is written to be interoperable with Simpson's rule,
#     but the concept of subintervals is not strictly necessary.

#     The number of points per segment nn_seg = (2 * num_intervals + 1)
#     The total number of points is nn_tot = nn_seg * num_segments

#     Inputs
#     ------
#     dqdt : float
#         The rate dqdt to integrate into quantity q (vector, length nn_tot)
#     dts : list
#         A list of timesteps dt corresponding to each interval (length num_intervals)
#     num_segments : int
#         The number of segments to integrate with different dts
#     num_intervals : int
#         The number of Simpson / 3 point quadrature intervals per segment

#     Returns
#     -------
#     delta_q : float
#         Amount of q accumulated during each interval (vector, length num_segments * (nn_seg - 1)
#     partials_wrt_dqdt : float
#         The Jacobian of delta_q with respect to the input rate dqdt
#         The result is a sparse matrix with num_segments * (nn_seg - 1) rows and nn_tot columns
#     partials_wrt_dts : list
#         A list of the Jacobians of delta_q with respect to the time steps
#         There will be one sparse vector with num_segments * (nn_seg - 1) rows per segment
#         But only (nn_seg - 1) rows will actually be populated
#     """
#     nn_seg = (2 * num_intervals + 1)
#     ndelta_seg = 2 * num_intervals
#     nn_tot = nn_seg * num_segments
#     ndelta_tot = ndelta_seg * num_segments

#     if len(dqdt) != nn_tot:
#         raise ValueError('dqdt must be of the correct length. dqdt is of length ' + str(len(dqdt)) +
#                          ' the number of nodes should be' + str(nn_tot))

#     if len(dts) != num_segments:
#         raise ValueError('must provide same number of dts as segments')

#     # the structure of this is (1/2) * the following:
#     # 1 1
#     #   1 1
#     #      1 1 and so on

#     # the row indices are basically 0 0 1 1 2 2  ....
#     jacmat_rowidx = np.repeat(np.arange(ndelta_seg), 2)
#     # the column indices are 0 1 1 2 2 3 3 4....
#     # so superimpose a 0 1 repeating pattern on a 0 0 1 1 2 2 repeating pattern
#     jacmat_colidx = np.tile(np.arange(2), ndelta_seg) + np.repeat(np.arange(0, ndelta_seg, 1), 2)
#     jacmat_data = np.tile(np.array([1, 1]) / 2, ndelta_seg)
#     jacmat_base = sp.csr_matrix((jacmat_data, (jacmat_rowidx, jacmat_colidx)))

#     jacmats_list = []
#     partials_wrt_dts = []
#     for i_seg in range(num_segments):
#         jacmats_list.append(jacmat_base * dts[i_seg])
#         # get the vector of partials of q with respect to this time step
#         dt_partials = jacmat_base.dot(dqdt[i_seg * nn_seg: (i_seg + 1) * nn_seg])
#         # offset the sparse partials if not the first segment to make it work in OpenMDAO terms
#         dt_partials_rowidxs = np.arange(i_seg * ndelta_seg, (i_seg + 1) * ndelta_seg)
#         dt_partials_colidxs = np.zeros((ndelta_seg,), dtype=np.int32)
#         partials_wrt_dts.append(sp.csr_matrix((dt_partials,
#                                               (dt_partials_rowidxs, dt_partials_colidxs)),
#                                                shape=(ndelta_tot, nn_tot)))
#     # now assemble the overall sparse block diagonal matrix to obtain the final result
#     partials_wrt_dqdt = sp.block_diag(jacmats_list)
#     delta_q = partials_wrt_dqdt.dot(dqdt)

#     return delta_q, partials_wrt_dqdt, partials_wrt_dts

# def backward_euler(dqdt, dts, num_segments=1, num_intervals=2,):
#     """This method integrates a rate over time using a backward Euler method
#     For now this component is written to be interoperable with Simpson's rule,
#     but the concept of subintervals is not strictly necessary.

#     The number of points per segment nn_seg = (2 * num_intervals + 1)
#     The total number of points is nn_tot = nn_seg * num_segments

#     Inputs
#     ------
#     dqdt : float
#         The rate dqdt to integrate into quantity q (vector, length nn_tot)
#     dts : list
#         A list of timesteps dt corresponding to each interval (length num_intervals)
#     num_segments : int
#         The number of segments to integrate with different dts
#     num_intervals : int
#         The number of Simpson / 3 point quadrature intervals per segment

#     Returns
#     -------
#     delta_q : float
#         Amount of q accumulated during each interval (vector, length num_segments * (nn_seg - 1)
#     partials_wrt_dqdt : float
#         The Jacobian of delta_q with respect to the input rate dqdt
#         The result is a sparse matrix with num_segments * (nn_seg - 1) rows and nn_tot columns
#     partials_wrt_dts : list
#         A list of the Jacobians of delta_q with respect to the time steps
#         There will be one sparse vector with num_segments * (nn_seg - 1) rows per segment
#         But only (nn_seg - 1) rows will actually be populated
#     """
#     nn_seg = (2 * num_intervals + 1)
#     ndelta_seg = 2 * num_intervals
#     nn_tot = nn_seg * num_segments
#     ndelta_tot = ndelta_seg * num_segments

#     if len(dqdt) != nn_tot:
#         raise ValueError('dqdt must be of the correct length. dqdt is of length ' + str(len(dqdt)) +
#                          ' the number of nodes should be' + str(nn_tot))

#     if len(dts) != num_segments:
#         raise ValueError('must provide same number of dts as segments')

#     # the structure of this is the following:
#     # 0 1
#     # 0 0 1
#     # 0 0 0 1 1 and so on

#     # the row indices are 0, 1, 2 ... n_delta seg
#     jacmat_rowidx = np.arange(ndelta_seg)
#     # the column indices are 1, 2, 3, ....
#     jacmat_colidx = np.arange(1, ndelta_seg + 1, 1)
#     jacmat_data = np.tile(np.array([1]), ndelta_seg)
#     jacmat_base = sp.csr_matrix((jacmat_data, (jacmat_rowidx, jacmat_colidx)))

#     jacmats_list = []
#     partials_wrt_dts = []
#     for i_seg in range(num_segments):
#         jacmats_list.append(jacmat_base * dts[i_seg])
#         # get the vector of partials of q with respect to this time step
#         dt_partials = jacmat_base.dot(dqdt[i_seg * nn_seg: (i_seg + 1) * nn_seg])
#         # offset the sparse partials if not the first segment to make it work in OpenMDAO terms
#         dt_partials_rowidxs = np.arange(i_seg * ndelta_seg, (i_seg + 1) * ndelta_seg)
#         dt_partials_colidxs = np.zeros((ndelta_seg,), dtype=np.int32)
#         partials_wrt_dts.append(sp.csr_matrix((dt_partials,
#                                               (dt_partials_rowidxs, dt_partials_colidxs)),
#                                                shape=(ndelta_tot, nn_tot)))
#     # now assemble the overall sparse block diagonal matrix to obtain the final result
#     partials_wrt_dqdt = sp.block_diag(jacmats_list)
#     delta_q = partials_wrt_dqdt.dot(dqdt)

#     return delta_q, partials_wrt_dqdt, partials_wrt_dts

# def integrator_partials_wrt_deltas(num_segments, num_intervals):
# """
# This function computes partials of an integrated quantity with respect to the "delta quantity per interval"
# in the context of openConcept's Simpson's rule approximated integration technique.

# Inputs
# ------
# num_segments : float
#     Number of mission segments to integrate (scalar)
# num_intervals : float
#     Number of Simpson intervals per segment (scalar)

# Outputs
# -------
# partial_q_wrt_deltas : float
#     A sparse (CSR) matrix representation of the partial derivatives of q
#     with respect to the delta quantity per half-interval
#     Dimension is nn * num_segments (rows) by (nn -1) * num_segments (cols)
#     where nn = (2 * num_intervals + 1)

# """
# nn = num_intervals * 2 + 1
# # the basic structure of the jacobian is lower triangular (all late values depend on all early ones)
# jacmat = np.tril(np.ones((num_segments*(nn-1),num_segments*(nn-1))))
# # the first entry of q has no dependence on the deltas so insert a row of zeros
# jacmat = np.insert(jacmat,0,np.zeros(num_segments*(nn-1)),axis=0)
# for i in range(1,num_segments):
#     # since the end of each segment is equal to the beginning of the next
#     # duplicate the jacobian row once at the end of each segment
#     duplicate_row = jacmat[nn*i-1,:]
#     jacmat = np.insert(jacmat,nn*i,duplicate_row,axis=0)
# partials_q_wrt_deltas = sp.csr_matrix(jacmat)
# return partials_q_wrt_deltas


class Integrator(ExplicitComponent):
    """
    Integrates rate variables implicitly.
    Add new integrated quantities by using the add_integrand method.
    "q" inputs here are illustrative only.

    Inputs
    ------
    duration : float
        The duration of the integration interval (can also use dt) (scalar)
    dq_dt : float
        Rate to integrate (vector)
    q_initial : float
        Starting value of quantity (scalar)

    Outputs
    -------
    q : float
        The vector quantity corresponding integral of dqdt over time
        Will have units  'rate_units' / 'diff_units'
    q_final : float
        The final value of the vector (scalar)
        Useful for connecting the end of one integrator to beginning of another

    Options
    -------
    num_nodes : int
        num_nodes = 2N + 1 where N = num_intervals
        The total length of the vector q is 2N + 1
    diff_units : str
        The units of the integrand (none by default)
    method : str
        Numerical method (default 'bdf3'; alternatively, 'simpson')
    time_setup : str
        Time configuration (default 'dt')
        'dt' creates input 'dt'
        'duration' creates input 'duration'
        'bounds' creates inputs 't_initial', 't_final'
    """

    def __init__(self, **kwargs):
        super(Integrator, self).__init__(**kwargs)
        self._state_vars = {}
        num_nodes = self.options["num_nodes"]
        method = self.options["method"]

        # check to make sure num nodes is OK
        if (num_nodes - 1) % 2 > 0:
            raise ValueError("num_nodes is " + str(num_nodes) + " and must be odd")

        if num_nodes > 1:
            if method == "bdf3":
                self.tri_mat, self.repeat_mat = bdf3_cache_matrix(num_nodes)
            elif method == "simpson":
                self.tri_mat, self.repeat_mat = simpson_cache_matrix(num_nodes)

    def initialize(self):
        self.options.declare("diff_units", default=None, desc="Units of the differential")
        self.options.declare("num_nodes", default=11, desc="Analysis points per segment")
        self.options.declare("method", default="bdf3", desc="Numerical method to use.")
        self.options.declare("time_setup", default="dt")

    def add_integrand(
        self,
        name,
        rate_name=None,
        start_name=None,
        end_name=None,
        val=0.0,
        start_val=0.0,
        units=None,
        rate_units=None,
        zero_start=False,
        final_only=False,
        lower=-1e30,
        upper=1e30,
    ):
        """
        Add a new integrated variable q = integrate(dqdt) + q0
        This will add an output with the integrated quantity, an output with the final value,
        an input with the rate source, and an input for the initial quantity.

        Parameters
        ----------
        name : str
            The name of the integrated variable to be created.
        rate_name : str
            The name of the input rate (default name"_rate")
        start_name  : str
            The name of the initial value input (default value name"_initial")
        end_name : str
            The name of the end value output (default value name"_final")
        units : str or None
            Units for the integrated quantity (or inferred automatically from rate_units)
        rate_units : str or None
            Units of the rate (can be inferred automatically from units)
        zero_start : bool
            If true, eliminates start value input and always begins from zero (default False)
        final_only : bool
            If true, only integrates final quantity, not all the intermediate points (default False)
        val : float
            Default value for the integrated output (default 0.0)
            Can be scalar or shape num_nodes
        start_val : float
            Default value for the initial value input (default 0.0)
        upper : float
            Upper bound on integrated quantity
        lower : float
            Lower bound on integrated quantity
        """

        num_nodes = self.options["num_nodes"]
        diff_units = self.options["diff_units"]
        time_setup = self.options["time_setup"]

        if units and rate_units:
            raise ValueError("Specify either quantity units or rate units, but not both")
        if units:
            # infer rate units from diff units and quantity units
            if not diff_units:
                rate_units = units
                warnings.warn(
                    "You have specified a integral with respect to a unitless integrand. Be aware of this.",
                    stacklevel=2,
                )
            else:
                rate_units = "(" + units + ") / (" + diff_units + ")"
        elif rate_units:
            # infer quantity units from rate units and diff units
            if not diff_units:
                units = rate_units
                warnings.warn(
                    "You have specified a integral with respect to a unitless integrand. Be aware of this.",
                    stacklevel=2,
                )
            else:
                units = "(" + rate_units + ") * (" + diff_units + ")"
        elif diff_units:
            # neither quantity nor rate units specified
            rate_units = "(" + diff_units + ")** -1"

        if not rate_name:
            rate_name = name + "_rate"
        if not start_name:
            start_name = name + "_initial"
        if not end_name:
            end_name = name + "_final"

        options = {
            "name": name,
            "rate_name": rate_name,
            "start_name": start_name,
            "start_val": start_val,
            "end_name": end_name,
            "units": units,
            "rate_units": rate_units,
            "zero_start": zero_start,
            "final_only": final_only,
            "upper": upper,
            "lower": lower,
        }

        # TODO maybe later can pass kwargs
        self._state_vars[name] = options
        if not hasattr(val, "__len__"):
            # scalar
            default_final_val = val
        else:
            # vector
            default_final_val = val[-1]

        self.add_input(rate_name, val=0.0, shape=(num_nodes), units=rate_units)
        self.add_output(end_name, units=units, val=default_final_val, upper=options["upper"], lower=options["lower"])
        if not final_only:
            self.add_output(
                name, shape=(num_nodes), val=val, units=units, upper=options["upper"], lower=options["lower"]
            )
        if not zero_start:
            self.add_input(start_name, val=start_val, units=units)
            if not final_only:
                self.declare_partials(
                    [name],
                    [start_name],
                    rows=np.arange(num_nodes),
                    cols=np.zeros((num_nodes,)),
                    val=np.ones((num_nodes,)),
                )
            self.declare_partials([end_name], [start_name], val=1)

        # set up sparse partial structure
        if num_nodes > 1:
            # single point analysis has no dqdt dependency since the outputs are equal to the inputs
            dQdrate, dQddtlist = multistep_integrator(
                0,
                np.ones((num_nodes,)),
                np.ones((1,)),
                self.tri_mat,
                self.repeat_mat,
                segment_names=None,
                segments_to_count=None,
                partials=True,
            )
            dQdrate_indices = dQdrate.nonzero()
            dQfdrate_indices = dQdrate.getrow(-1).nonzero()
            if not final_only:
                self.declare_partials([name], [rate_name], rows=dQdrate_indices[0], cols=dQdrate_indices[1])
            self.declare_partials(
                [end_name], [rate_name], rows=dQfdrate_indices[0], cols=dQfdrate_indices[1]
            )  # rows are zeros

            dQddt_seg = dQddtlist[0]
            dQddt_indices = dQddt_seg.nonzero()
            dQfddt_indices = dQddt_seg.getrow(-1).nonzero()

            if time_setup == "dt":
                if not final_only:
                    self.declare_partials([name], ["dt"], rows=dQddt_indices[0], cols=dQddt_indices[1])
                self.declare_partials([end_name], ["dt"], rows=dQfddt_indices[0], cols=dQfddt_indices[1])
            elif time_setup == "duration":
                if not final_only:
                    self.declare_partials([name], ["duration"], rows=dQddt_indices[0], cols=dQddt_indices[1])
                self.declare_partials([end_name], ["duration"], rows=dQfddt_indices[0], cols=dQfddt_indices[1])
            elif time_setup == "bounds":
                if not final_only:
                    self.declare_partials(
                        [name], ["t_initial", "t_final"], rows=dQddt_indices[0], cols=dQddt_indices[1]
                    )
                self.declare_partials(
                    [end_name], ["t_initial", "t_final"], rows=dQfddt_indices[0], cols=dQfddt_indices[1]
                )
            else:
                raise ValueError("Only dt, duration, and bounds are allowable values of time_setup")

    def setup(self):
        diff_units = self.options["diff_units"]
        num_nodes = self.options["num_nodes"]
        method = self.options["method"]
        time_setup = self.options["time_setup"]

        # branch logic here for the corner case of 0 segments
        # so point analysis can be run without breaking everything
        if num_nodes == 1:
            single_point = True
        else:
            single_point = False
        if not single_point:
            if method == "bdf3":
                self.tri_mat, self.repeat_mat = bdf3_cache_matrix(num_nodes)
            elif method == "simpson":
                self.tri_mat, self.repeat_mat = simpson_cache_matrix(num_nodes)

        if time_setup == "dt":
            self.add_input("dt", units=diff_units, desc="Time step")
        elif time_setup == "duration":
            self.add_input("duration", units=diff_units, desc="Time duration")
        elif time_setup == "bounds":
            self.add_input("t_initial", units=diff_units, desc="Initial time")
            self.add_input("t_final", units=diff_units, desc="Initial time")
        else:
            raise ValueError("Only dt, duration, and bounds are allowable values of time_setup")

    def compute(self, inputs, outputs):
        num_nodes = self.options["num_nodes"]
        time_setup = self.options["time_setup"]

        if num_nodes == 1:
            single_point = True
        else:
            single_point = False

        if time_setup == "dt":
            dts = [inputs["dt"][0]]
        elif time_setup == "duration":
            if num_nodes == 1:
                dts = [inputs["duration"][0]]
            else:
                dts = [inputs["duration"][0] / (num_nodes - 1)]
        elif time_setup == "bounds":
            delta_t = inputs["t_final"] - inputs["t_initial"]
            dts = [delta_t[0] / (num_nodes - 1)]

        for _, options in self._state_vars.items():
            if options["zero_start"]:
                q0 = np.array([0.0])
            else:
                q0 = inputs[options["start_name"]]
            if not single_point:
                Q = multistep_integrator(
                    q0,
                    inputs[options["rate_name"]],
                    dts,
                    self.tri_mat,
                    self.repeat_mat,
                    segment_names=None,
                    segments_to_count=None,
                    partials=False,
                )
            else:
                # single point case, no change, no dependence on time
                Q = q0

            if not options["final_only"]:
                outputs[options["name"]] = Q
            outputs[options["end_name"]] = Q[-1]

    def compute_partials(self, inputs, J):
        num_nodes = self.options["num_nodes"]
        time_setup = self.options["time_setup"]

        if num_nodes == 1:
            single_point = True
        else:
            single_point = False
        if not single_point:
            if time_setup == "dt":
                dts = [inputs["dt"][0]]
            elif time_setup == "duration":
                dts = [inputs["duration"][0] / (num_nodes - 1)]
            elif time_setup == "bounds":
                delta_t = inputs["t_final"] - inputs["t_initial"]
                dts = [delta_t[0] / (num_nodes - 1)]

            for _, options in self._state_vars.items():
                start_name = options["start_name"]
                end_name = options["end_name"]
                qty_name = options["name"]
                rate_name = options["rate_name"]
                final_only = options["final_only"]
                if options["zero_start"]:
                    q0 = 0
                else:
                    q0 = inputs[start_name]
                dQdrate, dQddtlist = multistep_integrator(
                    q0,
                    inputs[rate_name],
                    dts,
                    self.tri_mat,
                    self.repeat_mat,
                    segment_names=None,
                    segments_to_count=None,
                    partials=True,
                )

                if not final_only:
                    J[qty_name, rate_name] = dQdrate.data
                J[end_name, rate_name] = dQdrate.getrow(-1).data

                if time_setup == "dt":
                    if not final_only:
                        J[qty_name, "dt"] = np.squeeze(dQddtlist[0].toarray()[1:])
                    J[end_name, "dt"] = np.squeeze(dQddtlist[0].getrow(-1).toarray())

                elif time_setup == "duration":
                    if not final_only:
                        J[qty_name, "duration"] = np.squeeze(dQddtlist[0].toarray()[1:] / (num_nodes - 1))
                    J[end_name, "duration"] = np.squeeze(dQddtlist[0].getrow(-1).toarray() / (num_nodes - 1))

                elif time_setup == "bounds":
                    if not final_only:
                        if len(dQddtlist[0].data) == 0:
                            J[qty_name, "t_initial"] = np.zeros(J[qty_name, "t_initial"].shape)
                            J[qty_name, "t_final"] = np.zeros(J[qty_name, "t_final"].shape)
                        else:
                            J[qty_name, "t_initial"] = -dQddtlist[0].data / (num_nodes - 1)
                            J[qty_name, "t_final"] = dQddtlist[0].data / (num_nodes - 1)
                    if len(dQddtlist[0].getrow(-1).data) == 0:
                        J[end_name, "t_initial"] = 0
                        J[end_name, "t_final"] = 0
                    else:
                        J[end_name, "t_initial"] = -dQddtlist[0].getrow(-1).data / (num_nodes - 1)
                        J[end_name, "t_final"] = dQddtlist[0].getrow(-1).data / (num_nodes - 1)


class OldIntegrator(ExplicitComponent):
    """
    This component integrates a vector using a BDF3 formulation
    with 2nd order startup.

    Inputs
    ------
    dqdt : float
        The vector quantity to integrate.
        Length of the vector = (2 * num_intervals + 1) * num_segments
    segment|dt : float
        The timestep of "segment" (scalar)
        1 per segment
    q_initial : float
        Starting value of the quantity (scalar)

    Outputs
    -------
    q : float
        The vector quantity corresponding integral of dqdt over time
        Will have units  'rate_units' / 'diff_units'
    q_final : float
        The final value of the vector (scalar)
        Useful for connecting the end of one integrator to beginning of another

    Options
    -------
    segment_names : list
        A list of str with the names of the individual segments
        By default, if no segment_names are provided, one segment will be assumed and segment|dt will just be named "dt"
    segments_to_count : list
        A list of str with the names of segments to be included in the integration.
        By default, ALL segments will be included.
    num_nodes : int
        num_nodes = 2N + 1 where N = num_intervals
        The total length of the vector q is n_segments x (2N + 1)
    quantity_units : str
        The units of quantity being integrated (not the rate)
    diff_units : str
        The units of the integrand (none by default)
    rate_units : str
        The units of the rate being integrated
    method : str
        Numerical method (default 'bdf3'; alternatively, 'simpson)
    zero_start : bool
        If True, disables q_initial input (default False)
    final_only : bool
        If True, disables q output (q_final only) (default False)
    time_setup : str
        Time configuration (default 'dt')
        'dt' creates input 'dt'
        'duration' creates input 'duration'
        'bounds' creates inputs 't_initial', 't_final'
    """

    def initialize(self):
        self.options.declare("segment_names", default=None, desc="Names of differentiation segments")
        self.options.declare("segments_to_count", default=None, desc="Names of differentiation segments")
        self.options.declare("quantity_units", default=None, desc="Units of the quantity being differentiated")
        self.options.declare("diff_units", default=None, desc="Units of the differential")
        self.options.declare("rate_units", default=None, desc="Units of the rate being integrated")
        self.options.declare("num_nodes", default=11, desc="Analysis points per segment")
        self.options.declare("method", default="bdf3", desc="Numerical method to use.")
        self.options.declare("zero_start", default=False)
        self.options.declare("final_only", default=False)
        self.options.declare("lower", default=-1e30)
        self.options.declare("upper", default=1e30)
        self.options.declare("time_setup", default="dt")

    def setup(self):
        segment_names = self.options["segment_names"]
        segments_to_count = self.options["segments_to_count"]
        quantity_units = self.options["quantity_units"]
        diff_units = self.options["diff_units"]
        num_nodes = self.options["num_nodes"]
        method = self.options["method"]
        zero_start = self.options["zero_start"]
        final_only = self.options["final_only"]
        time_setup = self.options["time_setup"]

        # check to make sure num nodes is OK
        if (num_nodes - 1) % 2 > 0:
            raise ValueError("num_nodes must be odd")

        # branch logic here for the corner case of 0 segments
        # so point analysis can be run without breaking everything
        if num_nodes == 1:
            single_point = True
        else:
            single_point = False
        if not single_point:
            if method == "bdf3":
                self.tri_mat, self.repeat_mat = bdf3_cache_matrix(num_nodes)
            elif method == "simpson":
                self.tri_mat, self.repeat_mat = simpson_cache_matrix(num_nodes)

        if segment_names is None:
            n_segments = 1
        else:
            n_segments = len(segment_names)
        nn_tot = num_nodes * n_segments

        # TODO enable specifying rate units

        if quantity_units is None and diff_units is None:
            rate_units = None
        elif quantity_units is None:
            rate_units = "(" + diff_units + ")** -1"
        elif diff_units is None:
            rate_units = quantity_units
            warnings.warn(
                "You have specified a integral with respect to a unitless integrand. Be aware of this.", stacklevel=2
            )
        else:
            rate_units = "(" + quantity_units + ") / (" + diff_units + ")"
        # the output of this function is of length nn - 1. NO partial for first row (initial value)
        # get the partials of the delta quantities WRT the rates dDelta / drate

        self.add_input("dqdt", val=0, units=rate_units, desc="Quantity to integrate", shape=(nn_tot,))
        self.add_output(
            "q_final",
            units=quantity_units,
            desc="Final value of q",
            upper=self.options["upper"],
            lower=self.options["lower"],
        )

        if not final_only:
            self.add_output(
                "q",
                units=quantity_units,
                desc="Integral of dqdt",
                shape=(nn_tot,),
                upper=self.options["upper"],
                lower=self.options["lower"],
            )

        if not zero_start:
            self.add_input("q_initial", val=0, units=quantity_units, desc="Initial value")
            if not final_only:
                self.declare_partials(
                    ["q"], ["q_initial"], rows=np.arange(nn_tot), cols=np.zeros((nn_tot,)), val=np.ones((nn_tot,))
                )
            self.declare_partials(["q_final"], ["q_initial"], val=1)

        if not single_point:
            # single point analysis has no dqdt dependency since the outputs are equal to the inputs
            dQdrate, dQddtlist = multistep_integrator(
                0,
                np.ones((nn_tot,)),
                np.ones((n_segments,)),
                self.tri_mat,
                self.repeat_mat,
                segment_names=segment_names,
                segments_to_count=segments_to_count,
                partials=True,
            )
            dQdrate_indices = dQdrate.nonzero()
            dQfdrate_indices = dQdrate.getrow(-1).nonzero()
            if not final_only:
                self.declare_partials(["q"], ["dqdt"], rows=dQdrate_indices[0], cols=dQdrate_indices[1])
            self.declare_partials(
                ["q_final"], ["dqdt"], rows=dQfdrate_indices[0], cols=dQfdrate_indices[1]
            )  # rows are zeros

            if segment_names is None:
                dQddt_seg = dQddtlist[0]
                dQddt_indices = dQddt_seg.nonzero()
                dQfddt_indices = dQddt_seg.getrow(-1).nonzero()
                if time_setup == "dt":
                    self.add_input("dt", units=diff_units, desc="Time step")
                    if not final_only:
                        self.declare_partials(["q"], ["dt"], rows=dQddt_indices[0], cols=dQddt_indices[1])
                    self.declare_partials(["q_final"], ["dt"], rows=dQfddt_indices[0], cols=dQfddt_indices[1])
                elif time_setup == "duration":
                    self.add_input("duration", units=diff_units, desc="Time duration")
                    if not final_only:
                        self.declare_partials(["q"], ["duration"], rows=dQddt_indices[0], cols=dQddt_indices[1])
                    self.declare_partials(["q_final"], ["duration"], rows=dQfddt_indices[0], cols=dQfddt_indices[1])
                elif time_setup == "bounds":
                    self.add_input("t_initial", units=diff_units, desc="Initial time")
                    self.add_input("t_final", units=diff_units, desc="Initial time")
                    if not final_only:
                        self.declare_partials(
                            ["q"], ["t_initial", "t_final"], rows=dQddt_indices[0], cols=dQddt_indices[1]
                        )
                    self.declare_partials(
                        ["q_final"], ["t_initial", "t_final"], rows=dQfddt_indices[0], cols=dQfddt_indices[1]
                    )
                else:
                    raise ValueError("Only dt, duration, and bounds are allowable values of time_setup")

            else:
                if time_setup != "dt":
                    raise ValueError("dt is the only time_setup supported for multisegment integrations")
                for i_seg, segment_name in enumerate(segment_names):
                    self.add_input(segment_name + "|dt", units=diff_units, desc="Time step")
                    dQddt_seg = dQddtlist[i_seg]
                    dQddt_indices = dQddt_seg.nonzero()
                    dQfddt_indices = dQddt_seg.getrow(-1).nonzero()
                    if not final_only:
                        self.declare_partials(
                            ["q"], [segment_name + "|dt"], rows=dQddt_indices[0], cols=dQddt_indices[1]
                        )
                    self.declare_partials(
                        ["q_final"], [segment_name + "|dt"], rows=dQfddt_indices[0], cols=dQfddt_indices[1]
                    )

        else:
            if time_setup == "dt":
                self.add_input("dt", units=diff_units, desc="Time step")
            elif time_setup == "duration":
                self.add_input("duration", units=diff_units, desc="Time duration")
            elif time_setup == "bounds":
                self.add_input("t_initial", units=diff_units, desc="Initial time")
                self.add_input("t_final", units=diff_units, desc="Initial time")
            else:
                raise ValueError("Only dt, duration, and bounds are allowable values of time_setup")

    def compute(self, inputs, outputs):
        segment_names = self.options["segment_names"]
        num_nodes = self.options["num_nodes"]
        segments_to_count = self.options["segments_to_count"]
        zero_start = self.options["zero_start"]
        final_only = self.options["final_only"]
        time_setup = self.options["time_setup"]

        if num_nodes == 1:
            single_point = True
        else:
            single_point = False

        if segment_names is None:
            if time_setup == "dt":
                dts = [inputs["dt"][0]]
            elif time_setup == "duration":
                if num_nodes == 1:
                    dts = [inputs["duration"][0]]
                else:
                    dts = [inputs["duration"][0] / (num_nodes - 1)]
            elif time_setup == "bounds":
                delta_t = inputs["t_final"] - inputs["t_initial"]
                dts = [delta_t[0] / (num_nodes - 1)]
        else:
            dts = []
            for segment_name in segment_names:
                input_name = segment_name + "|dt"
                dts.append(inputs[input_name][0])
        if zero_start:
            q0 = 0
        else:
            q0 = inputs["q_initial"]
        if not single_point:
            Q = multistep_integrator(
                q0,
                inputs["dqdt"],
                dts,
                self.tri_mat,
                self.repeat_mat,
                segment_names=segment_names,
                segments_to_count=segments_to_count,
                partials=False,
            )
        else:
            # single point case, no change, no dependence on time
            Q = q0

        if not final_only:
            outputs["q"] = Q
        outputs["q_final"] = Q[-1]

    def compute_partials(self, inputs, J):
        segment_names = self.options["segment_names"]
        num_nodes = self.options["num_nodes"]
        segments_to_count = self.options["segments_to_count"]
        zero_start = self.options["zero_start"]
        final_only = self.options["final_only"]
        time_setup = self.options["time_setup"]

        if num_nodes == 1:
            single_point = True
        else:
            single_point = False
        if not single_point:
            if segment_names is None:
                if time_setup == "dt":
                    dts = [inputs["dt"][0]]
                elif time_setup == "duration":
                    dts = [inputs["duration"][0] / (num_nodes - 1)]
                elif time_setup == "bounds":
                    delta_t = inputs["t_final"] - inputs["t_initial"]
                    dts = [delta_t[0] / (num_nodes - 1)]
            else:
                dts = []
                for segment_name in segment_names:
                    input_name = segment_name + "|dt"
                    dts.append(inputs[input_name][0])

            if zero_start:
                q0 = 0
            else:
                q0 = inputs["q_initial"]
            dQdrate, dQddtlist = multistep_integrator(
                q0,
                inputs["dqdt"],
                dts,
                self.tri_mat,
                self.repeat_mat,
                segment_names=segment_names,
                segments_to_count=segments_to_count,
                partials=True,
            )

            if not final_only:
                J["q", "dqdt"] = dQdrate.data
            J["q_final", "dqdt"] = dQdrate.getrow(-1).data

            if segment_names is None:
                if time_setup == "dt":
                    if not final_only:
                        # if len(dQddtlist[0].data) == 0:
                        #     J['q','dt'] = np.zeros(J['q','dt'].shape)
                        # else:
                        #     J['q','dt'] = dQddtlist[0].data
                        J["q", "dt"] = np.squeeze(dQddtlist[0].toarray()[1:])
                    # if len(dQddtlist[0].getrow(-1).data) == 0:
                    #     J['q_final','dt'] = 0
                    # else:
                    #     J['q_final','dt'] = dQddtlist[0].getrow(-1).data
                    J["q_final", "dt"] = np.squeeze(dQddtlist[0].getrow(-1).toarray())

                elif time_setup == "duration":
                    if not final_only:
                        # if len(dQddtlist[0].data) == 0:
                        #     J['q','duration'] = np.zeros(J['q','duration'].shape)
                        # else:
                        #     J['q','duration'] = dQddtlist[0].data / (num_nodes - 1)
                        J["q", "duration"] = np.squeeze(dQddtlist[0].toarray()[1:] / (num_nodes - 1))
                    # if len(dQddtlist[0].getrow(-1).data) == 0:
                    #     J['q_final','duration'] = 0
                    # else:
                    #     J['q_final','duration'] = dQddtlist[0].getrow(-1).data / (num_nodes - 1)
                    J["q_final", "duration"] = np.squeeze(dQddtlist[0].getrow(-1).toarray() / (num_nodes - 1))

                elif time_setup == "bounds":
                    if not final_only:
                        if len(dQddtlist[0].data) == 0:
                            J["q", "t_initial"] = np.zeros(J["q", "t_initial"].shape)
                            J["q", "t_final"] = np.zeros(J["q", "t_final"].shape)
                        else:
                            J["q", "t_initial"] = -dQddtlist[0].data / (num_nodes - 1)
                            J["q", "t_final"] = dQddtlist[0].data / (num_nodes - 1)
                    if len(dQddtlist[0].getrow(-1).data) == 0:
                        J["q_final", "t_initial"] = 0
                        J["q_final", "t_final"] = 0
                    else:
                        J["q_final", "t_initial"] = -dQddtlist[0].getrow(-1).data / (num_nodes - 1)
                        J["q_final", "t_final"] = dQddtlist[0].getrow(-1).data / (num_nodes - 1)
            else:
                for i_seg, segment_name in enumerate(segment_names):
                    if not final_only:
                        J["q", segment_name + "|dt"] = dQddtlist[i_seg].data
                    J["q_final", segment_name + "|dt"] = dQddtlist[i_seg].getrow(-1).data
