from __future__ import division
import numpy as np
import scipy.sparse as sp
from openmdao.api import ExplicitComponent

def three_point_lagrange_integration(dqdt, dts, num_segments=1, num_intervals=2,):
    """This method integrates a rate over time using a 3 point Lagrange interpolant
    Similar to Simpson's rule except extended to provide increments at every subinterval

    The number of points per segment nn_seg = (2 * num_intervals + 1)
    The total number of points is nn_tot = nn_seg * num_segments

    Inputs
    ------
    dqdt : float
        The rate dqdt to integrate into quantity q (vector, length nn_tot)
    dts : list
        A list of timesteps dt corresponding to each interval (length num_intervals)
    num_segments : int
        The number of segments to integrate with different dts
    num_intervals : int
        The number of Simpson / 3 point quadrature intervals per segment

    Returns
    -------
    delta_q : float
        Amount of q accumulated during each interval (vector, length num_segments * (nn_seg - 1)
    partials_wrt_dqdt : float
        The Jacobian of delta_q with respect to the input rate dqdt
        The result is a sparse matrix with num_segments * (nn_seg - 1) rows and nn_tot columns
    partials_wrt_dts : list
        A list of the Jacobians of delta_q with respect to the time steps
        There will be one sparse vector with num_segments * (nn_seg - 1) rows per segment
        But only (nn_seg - 1) rows will actually be populated
    """
    nn_seg = (2 * num_intervals + 1)
    ndelta_seg = 2 * num_intervals
    nn_tot = nn_seg * num_segments
    ndelta_tot = ndelta_seg * num_segments

    if len(dqdt) != nn_tot:
        raise ValueError('dqdt must be of the correct length. dqdt is of length ' + str(len(dqdt)) +
                         ' the number of nodes should be' + str(nn_tot))

    if len(dts) != num_segments:
        raise ValueError('must provide same number of dts as segments')

    # first let us construct the basic three point quadrature jacobian which will be
    # multiplied by the timesteps to obtain the block matrices for the overall jacobian

    # the structure of this is (1/12) * the following:
    # 5 8 -1
    # -1 8 5
    #      5 8 -1
    #      -1 8 5
    #           5 8 -1
    #           -1 8 5    and so on

    # the row indices are basically 0 0 0 1 1 1 2 2 2 ....
    jacmat_rowidx = np.repeat(np.arange(ndelta_seg), 3)
    # the column indices are 0 1 2 0 1 2 2 3 4 2 3 4 4 5 6 and so on
    # so superimpose a 0 1 2 repeating pattern on a 0 0 0 0 0 0 2 2 2 2 2 2 2 repeating pattern
    jacmat_colidx = np.repeat(np.arange(0, ndelta_seg, 2), 6) + np.tile(np.arange(3), ndelta_seg)
    jacmat_data = np.tile(np.array([5, 8, -1, -1, 8, 5]) / 12, ndelta_seg // 2)
    jacmat_base = sp.csr_matrix((jacmat_data, (jacmat_rowidx, jacmat_colidx)))

    jacmats_list = []
    partials_wrt_dts = []
    for i_seg in range(num_segments):
        jacmats_list.append(jacmat_base * dts[i_seg])
        # get the vector of partials of q with respect to this time step
        dt_partials = jacmat_base.dot(dqdt[i_seg * nn_seg: (i_seg + 1) * nn_seg])
        # offset the sparse partials if not the first segment to make it work in OpenMDAO terms
        dt_partials_rowidxs = np.arange(i_seg * ndelta_seg, (i_seg + 1) * ndelta_seg)
        dt_partials_colidxs = np.zeros((ndelta_seg,), dtype=np.int32)
        raise ValueError(str(dt_partials.data.shape)+' '+str(dt_partials_colidxs.shape)+' '+str(dt_partials_rowidxs.shape))
        partials_wrt_dts.append(sp.csr_matrix((dt_partials.data,
                                              (dt_partials_rowidxs, dt_partials_colidxs)),
                                               shape=(ndelta_tot, nn_tot)))

    # now assemble the overall sparse block diagonal matrix to obtain the final result
    partials_wrt_dqdt = sp.block_diag(jacmats_list)
    delta_q = partials_wrt_dqdt.dot(dqdt)

    return delta_q, partials_wrt_dqdt, partials_wrt_dts

def integrator_partials_wrt_deltas(num_segments, num_intervals):
    """
    This function computes partials of an integrated quantity with respect to the "delta quantity per interval"
    in the context of openConcept's Simpson's rule approximated integration technique.

    Inputs
    ------
    num_segments : float
        Number of mission segments to integrate (scalar)
    num_intervals : float
        Number of Simpson intervals per segment (scalar)

    Outputs
    -------
    partial_q_wrt_deltas : float
        A sparse (CSR) matrix representation of the partial derivatives of q
        with respect to the delta quantity per half-interval
        Dimension is nn * num_segments (rows) by (nn -1) * num_segments (cols)
        where nn = (2 * num_intervals + 1)

    """
    nn = num_intervals * 2 + 1
    # the basic structure of the jacobian is lower triangular (all late values depend on all early ones)
    jacmat = np.tril(np.ones((num_segments*(nn-1),num_segments*(nn-1))))
    # the first entry of q has no dependence on the deltas so insert a row of zeros
    jacmat = np.insert(jacmat,0,np.zeros(num_segments*(nn-1)),axis=0)
    for i in range(1,num_segments):
        # since the end of each segment is equal to the beginning of the next
        # duplicate the jacobian row once at the end of each segment
        duplicate_row = jacmat[nn*i-1,:]
        jacmat = np.insert(jacmat,nn*i,duplicate_row,axis=0)
    partials_q_wrt_deltas = sp.csr_matrix(jacmat)
    return partials_q_wrt_deltas

class IntegrateQuantityEveryNode(ExplicitComponent):
    """
    This component integrates a vector using a 3rd order Lagrange polynomial
    equivalent to Simpson's rule, but with quadrature between every subinterval

    Inputs
    ------
    dqdt : float
        The vector quantity to integrate.
        Length of the vector = (2 * num_intervals + 1) * num_segments
    segment|dt : float
        The timestep of "segment" (scalar)
        1 per segment
    start_value : float
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
    num_intervals : int
        The number of Simpson intervals per segment
        The total number of points per segment is 2N + 1 where N = num_intervals
        The total length of the vector q is n_segments x (2N + 1)
    quantity_units : str
        The units of quantity being integrated (not the rate)
    diff_units : str
        The units of the integrand (none by default)
    """

    def initialize(self):
        self.options.declare('segment_names', default=None, desc="Names of differentiation segments")
        self.options.declare('quantity_units',default=None, desc="Units of the quantity being differentiated")
        self.options.declare('diff_units',default=None, desc="Units of the differential")
        self.options.declare('num_intervals',default=5, desc="Number of Simpsons rule intervals per segment")

    def setup(self):
        segment_names = self.options['segment_names']
        quantity_units = self.options['quantity_units']
        diff_units = self.options['diff_units']
        n_int_per_seg = self.options['num_intervals']

        nn_seg = (n_int_per_seg*2 + 1)
        if segment_names is None:
            n_segments = 1
        else:
            n_segments = len(segment_names)
        nn_tot = nn_seg * n_segments

        if quantity_units is None and diff_units is None:
            rate_units = None
        elif quantity_units is None:
            rate_units = '(' + diff_units +')** -1'
        elif diff_units is None:
            rate_units = quantity_units
            warnings.warn('You have specified a integral with respect to a unitless integrand. Be aware of this.')
        else:
            rate_units = '('+quantity_units+') / (' + diff_units +')'
        # the output of this function is of length nn - 1. NO partial for first row (initial value)
        # get the partials of the delta quantities WRT the rates dDelta / drate
        delta_q, dDelta_ddqdt, dDelta_dts = three_point_lagrange_integration(np.ones((nn_tot,)), np.ones((n_segments, )),
                                                                             num_segments=n_segments, num_intervals=n_int_per_seg)
        dq_dDelta = integrator_partials_wrt_deltas(n_segments, n_int_per_seg)
        # we need the partial of q with respect to the rates
        # which is dq / dDelta * dDelta / d{parameter}
        dq_ddqdt = dq_dDelta.dot(dDelta_ddqdt)

        self.add_input('dqdt', val=0, units=rate_units, desc='Quantity to integrate',shape=(nn_tot,))
        self.add_input('q_initial', val=0, units=quantity_units, desc='Initial value')
        self.add_output('q', units=quantity_units, desc='Integral of dqdt', shape=(nn_tot,))
        self.add_output('q_final', units=quantity_units, desc='Final value of q')

        self.declare_partials(['q'], ['q_initial'], rows=np.arange(nn_tot), cols=np.zeros((nn_tot,)), val=np.ones((nn_tot,)))
        self.declare_partials(['q_final'], ['q_initial'], val=1)

        dq_ddqdt_indices = dq_ddqdt.nonzero()
        dqfinal_ddqdt_indices = dq_ddqdt.getrow(-1).nonzero()
        self.declare_partials(['q'], ['dqdt'], rows=dq_ddqdt_indices[0], cols=dq_ddqdt_indices[1])
        self.declare_partials(['q_final'], ['dqdt'], rows=np.zeros((nn_tot,)), cols=dqfinal_ddqdt_indices[1])

        if segment_names is None:
            self.add_input('dt', units=diff_units, desc='Time step')
            dq_ddt = dq_dDelta.dot(dDelta_dts[0])
            dq_ddt_indices = dq_ddt.nonzero()
            self.declare_partials(['q'], ['dt'], rows=dq_ddt_indices[0], cols=dq_ddt_indices[1])
            dqfinal_ddt_indices = dq_ddt.getrow(-1).nonzero()
            self.declare_partials(['q_final'], ['dt'], rows=dqfinal_ddt_indices[0], cols=dqfinal_ddt_indices[1])
        else:
            for i_seg, segment_name in enumerate(segment_names):
                self.add_input(segment_name +'|dt', units=diff_units, desc='Time step')
                dq_ddt = dq_dDelta.dot(dDelta_dts[i_seg])
                dq_ddt_indices = dq_ddt.nonzero()
                self.declare_partials(['q'], [segment_name +'|dt'], rows=dq_ddt_indices[0], cols=dq_ddt_indices[1])
                dqfinal_ddt_indices = dq_ddt.getrow(-1).nonzero()
                self.declare_partials(['q_final'], [segment_name +'|dt'], rows=dqfinal_ddt_indices[0], cols=dqfinal_ddt_indices[1])

    def compute(self, inputs, outputs):
        segment_names = self.options['segment_names']
        n_int_per_seg = self.options['num_intervals']
        nn_seg = (n_int_per_seg*2 + 1)
        if segment_names is None:
            n_segments = 1
            dts = [inputs['dt'][0]]
        else:
            n_segments = len(segment_names)
            dts = []
            for i_seg, segment_name in enumerate(segment_names):
                input_name = segment_name+'|dt'
                dts.append(inputs[input_name][0])
        delta_q, dDelta_ddqdt, dDelta_dts = three_point_lagrange_integration(inputs['dqdt'], dts,
                                                                             num_segments=n_segments, num_intervals=n_int_per_seg)
        cumsum = np.cumsum(delta_q)
        cumsum = np.insert(cumsum, 0, 0)
        cumsum = cumsum + inputs['q_initial']
        if n_segments > 1:
            for i in range(1,n_segments):
                duplicate_row = cumsum[i*nn_seg-1]
                cumsum = np.insert(cumsum,i*nn_seg,duplicate_row)
        outputs['q'] = cumsum
        outputs['q_final'] = cumsum[-1]

    def compute_partials(self, inputs, J):
        segment_names = self.options['segment_names']
        quantity_units = self.options['quantity_units']
        diff_units = self.options['diff_units']
        n_int_per_seg = self.options['num_intervals']

        nn_seg = (n_int_per_seg*2 + 1)
        if segment_names is None:
            n_segments = 1
        else:
            n_segments = len(segment_names)
        nn_tot = nn_seg * n_segments

        if segment_names is None:
            n_segments = 1
            dts = [inputs['dt'][0]]
        else:
            n_segments = len(segment_names)
            dts = []
            for i_seg, segment_name in enumerate(segment_names):
                input_name = segment_name+'|dt'
                dts.append(inputs[input_name][0])

        delta_q, dDelta_ddqdt, dDelta_dts = three_point_lagrange_integration(inputs['dqdt'], dts,
                                                                     num_segments=n_segments, num_intervals=n_int_per_seg)
        dq_dDelta = integrator_partials_wrt_deltas(n_segments, n_int_per_seg)

        dq_ddqdt = dq_dDelta.dot(dDelta_ddqdt)

        J['q','dqdt'] = dq_ddqdt.data
        J['q_final', 'dqdt'] = dq_ddqdt.getrow(-1).data

        if segment_names is None:
            dq_ddt = dq_dDelta.dot(dDelta_dts[0])
            J['q','dt'] = dq_ddt.data
            J['q_final','dt'] = dq_ddt.getrow(-1).data
        else:
            for i_seg, segment_name in enumerate(segment_names):
                dq_ddt = dq_dDelta.dot(dDelta_dts[i_seg])
                J['q',segment_name+'|dt'] = dq_ddt.data
                J['q_final',segment_name+'|dt'] = dq_ddt.getrow(-1).data