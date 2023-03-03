import numpy as np
import scipy.sparse as sp
from openmdao.api import ExplicitComponent
import warnings


def first_deriv_second_order_accurate_stencil(nn_seg):
    # thought: eventually, create this matrix upon initializing the component to save time

    # assemble a sparse matrix which includes the finite difference stencil for ONE segment
    # CSR format: mat[rowidx[i], colidx[i]] = matvec[i]
    # TODO: middle CD coefficient is 0, could remove from sparsity pattern for big improvement
    rowidx = np.repeat(np.arange(1, nn_seg - 1), 2)  # [1, 1, 2, 2, .... ]
    rowidx = np.concatenate([np.tile(0, 3), rowidx, np.tile(nn_seg - 1, 3)])  # adds one-sided [0,0,0,1,1,2,2...]
    # the columns have an interesting sparsity pattern due to the one sided stencils at the edges:
    # [0, 1, 2,    0, 2,      1, 3,     2, 4, ...]
    offset = np.repeat(np.arange(0, nn_seg - 2), 2)
    colidx = np.tile(np.array([0, 2]), nn_seg - 2) + offset
    colidx = np.concatenate([np.arange(0, 3), colidx, np.arange(nn_seg - 3, nn_seg)])

    # the central difference stencil is:
    cd_stencil = np.array([-1 / 2, 1 / 2])
    # the biased stencils for the first, second, second-to-last, and last entries are:
    fwd_0_stencil = np.array([-3 / 2, 2, -1 / 2])
    bwd_0_stencil = np.array([1 / 2, -2, 3 / 2])
    stencil_vec = np.tile(cd_stencil, nn_seg - 2)
    stencil_vec = np.concatenate([fwd_0_stencil, stencil_vec, bwd_0_stencil])
    return stencil_vec, rowidx, colidx


def first_deriv_fourth_order_accurate_stencil(nn_seg):
    # thought: eventually, create this matrix upon initializing the component to save time

    # assemble a sparse matrix which includes the finite difference stencil for ONE segment
    # CSR format: mat[rowidx[i], colidx[i]] = matvec[i]
    rowidx = np.repeat(np.arange(0, nn_seg), 5)  # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1 .... ]
    # the columns have an interesting sparsity pattern due to the one sided stencils at the edges:
    # [0, 1, 2, 3, 4,    0, 1, 2, 3, 4,    0, 1, 2, 3, 4,   1, 2, 3, 4, 5,      2, 3, 4, 5, 6, ...]
    offset = np.repeat(np.arange(-2, nn_seg - 2), 5)
    colidx = np.tile(np.arange(0, 5), nn_seg) + offset
    colidx[:10] = np.tile(np.arange(0, 5), 2)
    colidx[-10:] = np.tile(np.arange(nn_seg - 5, nn_seg), 2)

    # the central difference stencil is:
    cd_stencil = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
    # the biased stencils for the first, second, second-to-last, and last entries are:
    fwd_0_stencil = np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4])
    fwd_1_stencil = np.array([-1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12])
    bwd_1_stencil = np.array([-1 / 12, 1 / 2, -3 / 2, 5 / 6, 1 / 4])
    bwd_0_stencil = np.array([1 / 4, -4 / 3, 3, -4, 25 / 12])
    stencil_vec = np.tile(cd_stencil, nn_seg)
    stencil_vec[:5] = fwd_0_stencil
    stencil_vec[5:10] = fwd_1_stencil
    stencil_vec[-10:-5] = bwd_1_stencil
    stencil_vec[-5:] = bwd_0_stencil

    return stencil_vec, rowidx, colidx


def first_deriv(dts, q, n_segments=1, n_simpson_intervals_per_segment=2, order=4):
    """
    This method differentiates a quantity over time using fourth order finite differencing

    A "segment" is defined as a portion of the quantity vector q with a
    constant delta t (or delta x, etc).
    This routine is designed to be used in the context of Simpson's rule integration
    where segment endpoints are coincident in time and n_points_per_seg = 2*n + 1

    Inputs
    ------
    dts : float
        "Time" step between points for each segment. Length of dts must equal n_segments (vector)
        Note that this is half the corresponding Simpson integration timestep
    q : float
        Quantity to be differentiated (vector)
        Total length of q = n_segments * n_points_per_seg
    n_segments : int
        Number of segments to differentiate. Each segment has constant dt (scalar)
    n_simpson_intervals_per_segment : int
        Number of Simpson's rule intervals per segment. Minimum is 2. (scalar)
        The number of points per segment = 2*n + 1
    order : int
        Order of accuracy (choose 2 or 4)

    Returns
    -------
    dqdt : float
        Derivative of q with respect to time (vector)
    """

    n_int_seg = n_simpson_intervals_per_segment
    nn_seg = n_simpson_intervals_per_segment * 2 + 1
    nn_tot = n_segments * nn_seg
    if order == 4 and n_int_seg < 2:
        raise ValueError(
            "Must use a minimum of 2 Simpson intervals or 5 points per segment due to fourth-order FD stencil"
        )
    if len(q) != nn_tot:
        raise ValueError("q must be of the correct length")
    if len(dts) != n_segments:
        raise ValueError("must provide same number of dts as segments")

    dqdt = np.zeros(q.shape)
    if order == 4:
        stencil_vec, rowidx, colidx = first_deriv_fourth_order_accurate_stencil(nn_seg)
    elif order == 2:
        stencil_vec, rowidx, colidx = first_deriv_second_order_accurate_stencil(nn_seg)
    else:
        raise ValueError("Must choose second or fourth order accuracy")

    stencil_mat = sp.csr_matrix((stencil_vec, (rowidx, colidx)))
    # now we have a generic stencil for each segment
    # assemble a block diagonal overall stencil for the entire vector q

    block_mat_list = []
    for i in range(n_segments):
        dt_seg = dts[i]
        block_mat_list.append(stencil_mat / dt_seg)
    overall_stencil = sp.block_diag(block_mat_list).toarray()
    dqdt = overall_stencil.dot(q)
    return dqdt


def first_deriv_partials(dts, q, n_segments=1, n_simpson_intervals_per_segment=2, order=4):
    """
    This method provides the Jacobian of a temporal first derivative

    A "segment" is defined as a portion of the quantity vector q with a
    constant delta t (or delta x, etc).
    This routine is designed to be used in the context of Simpson's rule integration
    where segment endpoints are coincident in time and n_points_per_seg = 2*n + 1

    Inputs
    ------
    dts : float
        "Time" step between points for each segment. Length of dts must equal n_segments (vector)
        Note that this is half the corresponding Simpson integration timestep
    q : float
        Quantity to be differentiated (vector)
        Total length of q = n_segments * n_points_per_seg
    n_segments : int
        Number of segments to differentiate. Each segment has constant dt (scalar)
    n_simpson_intervals_per_segment : int
        Number of Simpson's rule intervals per segment. Minimum is 2. (scalar)
        The number of points per segment = 2*n + 1

    Returns
    -------
    d_dqdt_dq : list of floats
        Jacobian of the temporal derivative vector dqdt with respect its quantity vector q (NxN sparse array)
        returned in CSR format as rowidx, colidx, data
    d_dqdt_ddt : list of list of floats
        Jacobians of the temporal derivative vector dqdt with respect to segment timestep dt[i] (Nx1 sparse arrays)
        returned in CSR format as rowidxs[i], colidxs[i], data[i]
    """
    n_int_seg = n_simpson_intervals_per_segment
    nn_seg = n_simpson_intervals_per_segment * 2 + 1
    nn_tot = n_segments * nn_seg
    if order == 4 and n_int_seg < 2:
        raise ValueError(
            "Must use a minimum of 2 Simpson intervals or 5 points per segment due to fourth-order FD stencil"
        )
    if len(q) != nn_tot:
        raise ValueError("q must be of the correct length")
    if len(dts) != n_segments:
        raise ValueError("must provide same number of dts as segments")

    if order == 4:
        stencil_vec, rowidx, colidx = first_deriv_fourth_order_accurate_stencil(nn_seg)
    elif order == 2:
        stencil_vec, rowidx, colidx = first_deriv_second_order_accurate_stencil(nn_seg)
    else:
        raise ValueError("Must choose second or fourth order accuracy")

    # now we have a generic stencil for each segment

    rowidx_wrt_q = np.array([])
    colidx_wrt_q = np.array([])
    partials_wrt_q = np.array([])
    rowidxs_wrt_dt = []
    colidxs_wrt_dt = []
    partials_wrt_dt = []
    stencil_mat = sp.csr_matrix((stencil_vec, (rowidx, colidx))).toarray()

    for i in range(n_segments):
        # first compute the indices and values of of dq' / dq in CSR format
        # the dimension of the matrix for all segments is nn_tot x nn_tot
        dt_seg = dts[i]
        local_rowidx = rowidx + i * nn_seg
        local_colidx = colidx + i * nn_seg
        rowidx_wrt_q = np.concatenate([rowidx_wrt_q, local_rowidx])
        colidx_wrt_q = np.concatenate([colidx_wrt_q, local_colidx])
        partials_wrt_q = np.concatenate([partials_wrt_q, stencil_vec / dt_seg])

        # next compute the indices and values of dq' / d(dt[i]) in CSR format
        # the dimension of the matrix for all segments is nn_tot x 1
        rowidxs_wrt_dt.append(np.arange(0, nn_seg) + i * nn_seg)
        colidxs_wrt_dt.append(np.zeros((nn_seg,), dtype=np.int32))
        local_partials = -np.dot(stencil_mat, q[i * nn_seg : (i + 1) * nn_seg]) * dt_seg**-2
        partials_wrt_dt.append(local_partials)

    wrt_q = [rowidx_wrt_q.astype(np.int32), colidx_wrt_q.astype(np.int32), partials_wrt_q]
    wrt_dt = [rowidxs_wrt_dt, colidxs_wrt_dt, partials_wrt_dt]
    return wrt_q, wrt_dt


class FirstDerivative(ExplicitComponent):
    """
    This component differentiates a vector using a second or fourth order finite difference approximation

    Inputs
    ------
    q : float
        The vector quantity to differentiate. q is defined consistent with the Simpson's Rule formulation used in the rest of this package
        q is comprised of multiple "segments" or phases with equal temporal spacing.
        The endpoints of each segment correspond to exacty the same timepoint as the endpoint of the next section
        q is of length nn_tot where nn_tot = n_segments  x(2 x num_intervals + 1)
        Each segment of q is of length nn_seg = (2 x num_intervals + 1)
    """

    """
        For example, a two-interval vector q has indices |0 1 2 3 4 | 5 6 7 8 9 |
        Elements 4 and 5 correspond to exactly the same time point
        All elements 0 to 4 have time interval dt[0] and 5 to 9 have time interval dt[1]
    segment|dt : float
        Time step for a given segment where the segment's name is "segment"
        There will be one timestep dt for each segment in the segment_names list
        By default, if no segment_names are provided, this input will be named simply "dt"

    Outputs
    -------
    dqdt : float
        The vector quantity corresponding to the first derivative of q (vector)
        Will have units  'quantity units' / 'diff_ units'

    Options
    -------
    segment_names : list
        A list of str with the names of the individual segments
        By default, if no segment_names are provided, one segment will be assumed and segment|dt will just be named "dt"
    num_intervals : int
        The number of Simpson intervals per segment
        The total number of points per segment is 2N + 1 where N = num_intervals
        The total length of the vector q is n_segments x (2N + 1)
    quantity_units : str
        The units of the quantity q being differentiated
    diff_units : str
        The units of the derivative being taken (none by default)
    order : int
        Order of accuracy (default 4). May also choose 2 for 2nd order accuracy and sparser structure
    """

    def initialize(self):
        self.options.declare("segment_names", default=None, desc="Names of differentiation segments")
        self.options.declare("quantity_units", default=None, desc="Units of the quantity being differentiated")
        self.options.declare("diff_units", default=None, desc="Units of the differential")
        self.options.declare("num_intervals", default=5, desc="Number of Simpsons rule intervals per segment")
        self.options.declare("order", default=4, desc="Order of accuracy")

    def setup(self):
        segment_names = self.options["segment_names"]
        quantity_units = self.options["quantity_units"]
        diff_units = self.options["diff_units"]
        order = self.options["order"]

        n_int_per_seg = self.options["num_intervals"]
        nn_seg = n_int_per_seg * 2 + 1
        if segment_names is None:
            n_segments = 1
        else:
            n_segments = len(segment_names)
        nn_tot = nn_seg * n_segments

        if quantity_units is None and diff_units is None:
            deriv_units = None
        elif quantity_units is None:
            deriv_units = "(" + diff_units + ")** -1"
        elif diff_units is None:
            deriv_units = quantity_units
            warnings.warn(
                "You have specified a derivative with respect to a unitless differential. Be aware of this.",
                stacklevel=2,
            )
        else:
            deriv_units = "(" + quantity_units + ") / (" + diff_units + ")"
        wrt_q, wrt_dt = first_deriv_partials(
            np.ones((n_segments,)),
            np.ones((nn_tot,)),
            n_segments=n_segments,
            n_simpson_intervals_per_segment=n_int_per_seg,
            order=order,
        )

        self.add_input("q", val=0, units=quantity_units, desc="Quantity to differentiate", shape=(nn_tot,))
        self.add_output("dqdt", units=deriv_units, desc="First derivative of q", shape=(nn_tot,))
        self.declare_partials(["dqdt"], ["q"], rows=wrt_q[0], cols=wrt_q[1])

        if segment_names is None:
            self.add_input("dt", units=diff_units, desc="Time step")
            self.declare_partials(["dqdt"], ["dt"], rows=wrt_dt[0][0], cols=wrt_dt[1][0])
        else:
            for i_seg, segment_name in enumerate(segment_names):
                self.add_input(segment_name + "|dt", units=diff_units, desc="Time step")
                self.declare_partials(["dqdt"], [segment_name + "|dt"], rows=wrt_dt[0][i_seg], cols=wrt_dt[1][i_seg])

    def compute(self, inputs, outputs):
        segment_names = self.options["segment_names"]
        order = self.options["order"]
        n_int_per_seg = self.options["num_intervals"]
        if segment_names is None:
            n_segments = 1
            dts = [inputs["dt"][0]]
        else:
            n_segments = len(segment_names)
            dts = []
            for segment_name in segment_names:
                input_name = segment_name + "|dt"
                dts.append(inputs[input_name][0])
        dqdt = first_deriv(
            dts, inputs["q"], n_segments=n_segments, n_simpson_intervals_per_segment=n_int_per_seg, order=order
        )
        outputs["dqdt"] = dqdt

    def compute_partials(self, inputs, J):
        segment_names = self.options["segment_names"]
        order = self.options["order"]

        n_int_per_seg = self.options["num_intervals"]
        if segment_names is None:
            n_segments = 1
            dts = [inputs["dt"][0]]
        else:
            n_segments = len(segment_names)
            dts = []
            for segment_name in segment_names:
                input_name = segment_name + "|dt"
                dts.append(inputs[input_name][0])
        wrt_q, wrt_dt = first_deriv_partials(
            dts, inputs["q"], n_segments=n_segments, n_simpson_intervals_per_segment=n_int_per_seg, order=order
        )

        J["dqdt", "q"] = wrt_q[2]
        if segment_names is None:
            J["dqdt", "dt"] = wrt_dt[2][0]
        else:
            for i_seg, segment_name in enumerate(segment_names):
                J["dqdt", segment_name + "|dt"] = wrt_dt[2][i_seg]
