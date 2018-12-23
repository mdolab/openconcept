from __future__ import division
import numpy as np
import scipy.sparse as sp
from openmdao.api import ExplicitComponent

def simpson_integral(dts, q, n_segments=1, n_simpson_intervals_per_segment=2):
    """
    This method integrates a rate over time using Simpson's rule

    A "segment" is defined as a portion of the quantity vector q with a
    constant delta t (or delta x, etc)
    dts = list of doubles representing the time steps for each segment.
    This is the data timestep - the interval timestep is 2x this
    q = the data to be integrated
    n_segments = how many segments
    n_simpson_intervals_per_segment = how many simpson intervals to use per segment.
    Each one requires 2*N+1 data points

    Returns
    -------
    delta_q : float
        Amount of q accumulated during each interval (vector)
    int_q : float
        Total amount of q accumulated during all phases (scalar)

    """
    n_int_seg = n_simpson_intervals_per_segment
    n_int_tot = n_segments * n_int_seg
    nn_seg = (n_simpson_intervals_per_segment * 2 + 1)
    nn_tot = n_segments * nn_seg
    if len(q) != nn_tot:
        raise ValueError('q must be of the correct length')

    if len(dts) != n_segments:
        raise ValueError('must provide same number of dts as segments')

    # first let us take partial derivatives of the integral with respect to the states
    # each row represents a value of the integrated vector. It will contain three values,
    # so repeat it 3 times
    delta_q = np.zeros(n_int_tot)
    # create a placeholder for the column indices

    for i in range(n_segments):
        dt_seg = dts[i]
        delta_q[i * n_int_seg:(i + 1) * n_int_seg] = (q[nn_seg * i:nn_seg * (i + 1) - 2:2] +
                                                      4 * q[nn_seg * i + 1:nn_seg * (i + 1) - 1:2] +
                                                      q[nn_seg * i + 2:nn_seg * (i + 1):2]) * dt_seg / 3

    int_q = np.sum(delta_q)
    return int_q, delta_q


def simpson_partials(dts, q, n_segments=1, n_simpson_intervals_per_segment=2,):
    """This method integrates a rate over time using Simpson's rule

    A "segment" is defined as a portion of the quantity vector q
    with a constant delta t (or delta x, etc)
    dts = list of doubles representing the time steps for each segment.
    This is the data timestep - the interval timestep is 2x this
    q = the data to be integrated
    n_segments = how many segments
    n_simpson_intervals_per_segment = how many simpson intervals to use per segment.
    Each one requires 2*N+1 data points

    Returns
    -------
    delta_q : float
        Amount of q accumulated during each interval (vector)
    int_q : float
        Total amount of q accumulated during all phases (scalar)

    """
    n_int_seg = n_simpson_intervals_per_segment
    n_int_tot = n_segments * n_int_seg
    nn_seg = (n_simpson_intervals_per_segment * 2 + 1)
    nn_tot = n_segments * nn_seg

    if len(q) != nn_tot:
        raise ValueError('q must be of the correct length. q is of length ' + str(len(q)) +
                         ' the number of nodes should be' + str(nn_tot))

    if len(dts) != n_segments:
        raise ValueError('must provide same number of dts as segments')

    # first let us take partial derivatives of the integral with respect to the states
    # each row represents a value of the integrated vector.
    # It will contain three values, so repeat it 3 times
    rowidx_wrt_q = np.repeat(range(n_int_tot), 3)
    # create a placeholder for the column indices
    colidx_wrt_q = np.zeros(n_int_tot * 3)
    partials_wrt_q = np.tile([1 / 3, 4 / 3, 1 / 3], n_int_tot)
    rowidxs_wrt_dt = []
    colidxs_wrt_dt = []
    partials_wrt_dt = []

    for i in range(n_segments):
        dt_seg = dts[i]
        start_colidx = nn_seg*i
        end_colidx = nn_seg*(i+1)
        partials_wrt_q[i*n_int_seg*3:(i+1)*n_int_seg*3] = partials_wrt_q[i*n_int_seg*3:(i+1)*n_int_seg*3] * dt_seg
        colidx_wrt_q[i*n_int_seg*3+0:(i+1)*n_int_seg*3-2:3] = np.arange(start_colidx,end_colidx-2,2)
        colidx_wrt_q[i*n_int_seg*3+1:(i+1)*n_int_seg*3-1:3] = np.arange(start_colidx+1,end_colidx-1,2)
        colidx_wrt_q[i*n_int_seg*3+2:(i+1)*n_int_seg*3-0:3] = np.arange(start_colidx+2,end_colidx,2)

        rowidx_wrt_dt = np.arange(i*n_int_seg,(i+1)*n_int_seg)
        colidx_wrt_dt = np.zeros(n_int_seg)
        partial_wrt_dt = (q[nn_seg*i:nn_seg*(i+1)-2:2]+4*q[nn_seg*i+1:nn_seg*(i+1)-1:2]+q[nn_seg*i+2:nn_seg*(i+1):2])/3
        rowidxs_wrt_dt.append(rowidx_wrt_dt.astype(np.int32))
        colidxs_wrt_dt.append(colidx_wrt_dt.astype(np.int32))
        partials_wrt_dt.append(partial_wrt_dt)

    wrt_q = [rowidx_wrt_q.astype(np.int32), colidx_wrt_q.astype(np.int32), partials_wrt_q]
    wrt_dt = [rowidxs_wrt_dt, colidxs_wrt_dt, partials_wrt_dt]
    return wrt_q, wrt_dt


def simpson_integral_every_node(dts,q,n_segments=1,n_simpson_intervals_per_segment=2):
    """This method integrates a rate over time using Simpson's rule and assumes that q linearly changes within the Simpson subintervals.
        Unlike the intervals above, this method returns a vector of length nn-1 instead of nn-1/2
        A "segment" is defined as a portion of the quantity vector q with a constant delta t (or delta x, etc)
        dts = list of doubles representing the time steps for each segment. This is the data timestep - the interval timestep is 2x this
        q = the data to be integrated
        n_segments = how many segments
        n_simpson_intervals_per_segment = how many simpson intervals to use per segment. Each one requires 2*N+1 data points

        returns:
        delta_q = amount of q accumulated during each interval (corresponds to the intervals between q, 2x as often as the simpson subintervals)
        int_q = total amount of q accumulated during all phases

    """
    int_q, delta_q_2x = simpson_integral(dts,q,n_segments=n_segments,n_simpson_intervals_per_segment=n_simpson_intervals_per_segment)
    delta_q = np.repeat(delta_q_2x,2)/2.0
    return int_q, delta_q


def simpson_partials_every_node(dts,q,n_segments=1,n_simpson_intervals_per_segment=2):
    wrt_q_2x, wrt_dt_2x = simpson_partials(dts,q,n_segments=n_segments,n_simpson_intervals_per_segment=n_simpson_intervals_per_segment)
    # since row 0 and 1, row 2 and 3, etc are identical, it's easiest to edit the row index and repeat the
    n_int_seg = n_simpson_intervals_per_segment
    n_int_tot = n_segments * n_int_seg
    nn_seg = (n_simpson_intervals_per_segment * 2 + 1)
    nn_tot = n_segments*nn_seg

    #first let us take partial derivatives of the integral with respect to the states
    #each row represents a value of the integrated vector. It will contain three values, so repeat it 3 times
    #we want every other row, like so: [0,2,4,6...,1,3,5,7...]
    rownos = np.concatenate([np.arange(0,2*n_int_tot-1,2),np.arange(1,2*n_int_tot,2)])
    rowidx_wrt_q = np.repeat(rownos,3)
    #We then append a second copy of the original row indices (since row 0 is identical to row 1 and so on)
    colidx_wrt_q = np.tile(wrt_q_2x[1],2)
    #recall that the partials are all /2 since we divided by two
    partials_wrt_q = np.tile(wrt_q_2x[2],2) / 2

    rowidxs_wrt_dt = []
    colidxs_wrt_dt = []
    partials_wrt_dt = []

    for i in range(n_segments):
        #there are now twice as many rows. we need to go from:
        # rowidx = [0,1,2,3]
        # to
        # rowidx = [0,2,4,6,1,3,5,7]

        rowidx_i = np.concatenate([wrt_dt_2x[0][i]*2, wrt_dt_2x[0][i]*2 + 1])
        colidx_i = np.tile(wrt_dt_2x[1][i],2)
        part_i = np.tile(wrt_dt_2x[2][i],2) / 2

        rowidxs_wrt_dt.append(rowidx_i)
        colidxs_wrt_dt.append(colidx_i)
        partials_wrt_dt.append(part_i)

    wrt_q = [rowidx_wrt_q, colidx_wrt_q, partials_wrt_q]
    wrt_dt = [rowidxs_wrt_dt, colidxs_wrt_dt, partials_wrt_dt]
    return wrt_q, wrt_dt

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
    jacmat = np.tril(np.ones((n_segments*(nn-1),n_segments*(nn-1))))
    # the first entry of q has no dependence on the deltas so insert a row of zeros
    jacmat = np.insert(jacmat,0,np.zeros(n_segments*(nn-1)),axis=0)
    for i in range(1,n_segments):
        # since the end of each segment is equal to the beginning of the next
        # duplicate the jacobian row once at the end of each segment
        duplicate_row = jacmat[nn*i-1,:]
        jacmat = np.insert(jacmat,nn*i,duplicate_row,axis=0)
    partials_q_wrt_deltas = sp.csr_matrix(jacmat)
    return partials_q_wrt_deltas

# class IntegrateQuantityEveryNode(ExplicitComponent):
#     """
#     This component integrates a vector using Simpson's rule with linear subinterval interpolation

#     Inputs
#     ------
#     dqdt : float
#         The vector quantity to integrate.
#         Length of the vector = (2 * num_intervals + 1) * num_segments
#     segment|dt : float
#         The timestep of "segment" (scalar)
#         1 per segment
#     start_value : float
#         Starting value of the quantity (scalar)

#     Outputs
#     -------
#     q : float
#         The vector quantity corresponding integral of dqdt over time
#         Will have units  'rate_units' / 'diff_ units'
#     q_final : float
#         The final value of the vector (scalar)
#         Useful for connecting the end of one integrator to beginning of another

#     Options
#     -------
#     segment_names : list
#         A list of str with the names of the individual segments
#         By default, if no segment_names are provided, one segment will be assumed and segment|dt will just be named "dt"
#     num_intervals : int
#         The number of Simpson intervals per segment
#         The total number of points per segment is 2N + 1 where N = num_intervals
#         The total length of the vector q is n_segments x (2N + 1)
#     quantity_units : str
#         The units of quantity being integrated (not the rate)
#     diff_units : str
#         The units of the integrand (none by default)
#     """

#     def initialize(self):
#         self.options.declare('segment_names', default=None, desc="Names of differentiation segments")
#         self.options.declare('quantity_units',default=None, desc="Units of the quantity being differentiated")
#         self.options.declare('diff_units',default=None, desc="Units of the differential")
#         self.options.declare('num_intervals',default=5, desc="Number of Simpsons rule intervals per segment")

#     def setup(self):
#         segment_names = self.options['segment_names']
#         quantity_units = self.options['quantity_units']
#         diff_units = self.options['diff_units']
#         n_int_per_seg = self.options['num_intervals']

#         nn_seg = (n_int_per_seg*2 + 1)
#         if segment_names is None:
#             n_segments = 1
#         else:
#             n_segments = len(segment_names)
#         nn_tot = nn_seg * n_segments

#         if quantity_units is None and diff_units is None:
#             rate_units = None
#         elif quantity_units is None:
#             rate_units = '(' + diff_units +')** -1'
#         elif diff_units is None:
#             rate_units = quantity_units
#             warnings.warn('You have specified a integral with respect to a unitless integrand. Be aware of this.')
#         else:
#             rate_units = '('+quantity_units+') / (' + diff_units +')'
#         # the output of this function is of length nn - 1. NO partial for first row (initial value)
#         # get the partials of the delta quantities WRT the rates dDelta / drate
#         wrt_rate, wrt_dts = simpson_partials_every_node(np.ones((n_segments, )), np.ones((nn_tot,)),
#                                                        n_segments=n_segments, n_simpson_intervals_per_segment=n_int_per_seg)

#         dDelta_drate = sp.csr_matrix((wrt_rate[2],(wrt_rate[0], wrt_rate[1])))
#         dq_dDelta = integrator_partials_wrt_deltas(num_segments, num_intervals)
#         # we need the partial of q with respect to the rates
#         # which is dq / dDelta * dDelta / d{parameter}
#         dq_drate = dq_dDelta.dot(dDelta_drate)

#         self.add_input('dqdt', val=0, units=rate_units, desc='Quantity to integrate',shape=(nn_tot,))
#         self.add_input('q_initial', val=0, units=quantity_units, desc='Initial value')
#         self.add_output('q', units=quantity_units, desc='Integral of dqdt', shape=(nn_tot,))
#         self.add_output('q_final', units=quantity_units, desc='Final value of q')

#         self.declare_partials(['q'], ['q_initial'], rows=np.arange(nn_tot), cols=np.zeros((nn_tot,)), val=np.ones((nn_tot,)))
#         self.declare_partials(['q_final'], ['q_initial'], val=1)

#         dq_drate_indices = dq_drate.nonzero()
#         dqfinal_drate_indices = dq_drate.getrow(-1).nonzero()
#         self.declare_partials(['q'], ['dqdt'], rows=dq_drate_indices[0], cols=dq_drate_indices[1])
#         self.declare_partials(['q_final'], ['dqdt'], rows=np.zeros((nn_tot,)), cols=dqfinal_drate_indices[1])

#         if segment_names is None:
#             self.add_input('dt', units=diff_units, desc='Time step')
#             dDelta_ddt = sp.csr_matrix((wrt_dts[2][0],(wrt_dts[0][0], wrt_dts[1][0])))
#             dq_ddt = dq_dDelta.dot(dDelta_ddt)
#             dq_ddt_indices = dq_ddt.nonzero()
#             self.declare_partials(['q'], ['dt'], rows=dq_ddt_indices[0], cols=dq_ddt_indices[1])
#             dqfinal_ddt_indices = dq_ddt.getrow(-1).nonzero()
#             self.declare_partials(['q_final'], ['dt'], rows=dqfinal_ddt_indices[0], cols=dqfinal_ddt_indices[1])
#         else:
#             for i_seg, segment_name in enumerate(segment_names):
#                 self.add_input(segment_name +'|dt', units=diff_units, desc='Time step')
#                 dDelta_ddt = sp.csr_matrix((wrt_dts[2][i_seg],(wrt_dts[0][i_seg], wrt_dts[1][i_seg])))
#                 dq_ddt = dq_dDelta.dot(dDelta_ddt)
#                 dq_ddt_indices = dq_ddt.nonzero()
#                 self.declare_partials(['q'], [segment_name +'|dt'], rows=dq_ddt_indices[0], dq_ddt_indices[1])
#                 dqfinal_ddt_indices = dq_ddt.getrow(-1).nonzero()
#                 self.declare_partials(['q_final'], [segment_name +'|dt'], rows=dqfinal_ddt_indices[0], dqfinal_ddt_indices[1])

#     def compute(self, inputs, outputs):
#         segment_names = self.options['segment_names']
#         n_int_per_seg = self.options['num_intervals']
#         nn_seg = (n_int_per_seg*2 + 1)
#         if segment_names is None:
#             n_segments = 1
#             dts = [inputs['dt'][0]]
#         else:
#             n_segments = len(segment_names)
#             dts = []
#             for i_seg, segment_name in enumerate(segment_names):
#                 input_name = segment_name+'|dt'
#                 dts.append(inputs[input_name][0])
#         deltas = simpson_integral_every_node(dts, inputs['dqdt'], n_segments=n_segments, n_simpson_intervals_per_segment=n_int_per_seg)
#         cumsum = np.cumsum(deltas)
#         np.insert(cumsum, 0, 0)
#         cumsum = cumsum + inputs['q_initial']
#         if n_segments > 1:
#             for i in range(1,n_segments):
#                 duplicate_row = cumsum[i*nn_seg-1]
#                 cumsum = np.insert(cumsum,i*nn_seg,duplicate_row)
#         outputs['q'] = cumsum
#         outputs['q_final'] = cumsum[-1]

#     def compute_partials(self, inputs, J):
#         segment_names = self.options['segment_names']
#         quantity_units = self.options['quantity_units']
#         diff_units = self.options['diff_units']
#         n_int_per_seg = self.options['num_intervals']

#         nn_seg = (n_int_per_seg*2 + 1)
#         if segment_names is None:
#             n_segments = 1
#         else:
#             n_segments = len(segment_names)
#         nn_tot = nn_seg * n_segments

#         if segment_names is None:
#             n_segments = 1
#             dts = [inputs['dt'][0]]
#         else:
#             n_segments = len(segment_names)
#             dts = []
#             for i_seg, segment_name in enumerate(segment_names):
#                 input_name = segment_name+'|dt'
#                 dts.append(inputs[input_name][0])

#         wrt_rate, wrt_dts = simpson_partials_every_node(dts, inputs['dqdt'], n_segments=n_segments, n_simpson_intervals_per_segment=n_int_per_seg)

#         dDelta_drate = sp.csr_matrix((wrt_rate[2],(wrt_rate[0], wrt_rate[1])))
#         dq_dDelta = integrator_partials_wrt_deltas(num_segments, num_intervals)
#         # we need the partial of q with respect to the rates
#         # which is dq / dDelta * dDelta / d{parameter}
#         dq_drate = dq_dDelta.dot(dDelta_drate)

#         J['q','dqdt'] = dq_drate.data
#         dqfinal_drate = dq_drate.getrow(-1)
#         J['q_final', 'dqdt'] = dqfinal_drate.data

#         if segment_names is None:
#             dDelta_ddt = sp.csr_matrix((wrt_dts[2][0],(wrt_dts[0][0], wrt_dts[1][0])))
#             dq_ddt = dq_dDelta.dot(dDelta_ddt)
#             J['q','dt'] = dq_ddt.data
#             J['q_final','dt'] = dq_ddt.getrow(-1).data
#         else:
#             for i_seg, segment_name in enumerate(segment_names):
#                 dDelta_ddt = sp.csr_matrix((wrt_dts[2][i_seg],(wrt_dts[0][i_seg], wrt_dts[1][i_seg])))
#                 dq_ddt = dq_dDelta.dot(dDelta_ddt)
#                 J['q',segment_name+'|dt'] = dq_ddt.data
#                 J['q_final',segment_name+'|dt'] = dq_ddt.getrow(-1).data


class IntegrateQuantity(ExplicitComponent):
    """This component integrates a first-order rate quantity vector over a differential with CONSTANT spacing using Simpson's 3rd order method.
    Inputs:
    rate (vector) - rate of change of a quantity q with respect to the differential. E.g. if your differential is time, the rate should be dq/dt. Length must be 2*N+1
    lower_limit (scalar) - the lower limit of the integral. e.g. if your differential var is time, lower_limit is t0
    upper_limit (scalar) - the upper limit of the integral. e.g. if your differential var is time, upper limit is tf
    Outputs:
    delta_quantity: the total change in the quantity over the integral period. E.g. if integrating rate dq/dt wrt t, output is total change in q
    Options:
    num_intervals: Number of Simpson integration intevals to use. Length of the rate input vector will be 2*N+1
    quantity_units: Units of quantity (not including the rate) e.g. kg NOT kg/s
    diff_units: Units of the differential (not incuding the quantity) e.g. s, not kg/s

    Example 1: integrate v (dr/dt) with respect to time over constant time spacing during a fixed-time segment
    Example 2: integrate v / a, a.k.a  [(dr/dt) * (dt/dv)] wrt velocity over constant velocity spacing during a segment with known starting and ending velocities
    """
    def initialize(self):
        self.options.declare('num_intervals',desc="Number of integration intervals to use. Input rate vector must be length 2*N+1")
        self.options.declare('quantity_units',default=None,desc="Units of the quantity (NOT the rate) being integrated.")
        self.options.declare('diff_units',default=None,desc="Units of the differential. Should match the limits of integration")
        self.options.declare('force_signs',default=False,desc='In some cases, spurious integrated results can occur when a system is not physically capable of achieving the integration limits (e.g. integ wrt velocity with negative acceleration). This catches this case and returns an error')

    def setup(self):
        n_int = self.options['num_intervals']
        nn = (n_int*2 + 1)
        qty_unit = self.options['quantity_units']
        diff_unit = self.options['diff_units']
        if qty_unit is None and diff_unit is None:
            rate_unit = None
        elif qty_unit is None or diff_unit is None:
            raise ValueError('Mismatched units in the integrator')
        else:
            rate_unit = '('+qty_unit+') / (' + diff_unit +')'
        self.add_input('lower_limit', val=0, units=diff_unit, desc='Lower limit of integration')
        self.add_input('upper_limit', val=1, units=diff_unit, desc='Upper limit of integration')
        self.add_input('rate', units=rate_unit, desc='Rate to integrate', shape=(nn,))
        self.add_output('delta_quantity', units=qty_unit, desc='Total change in the integrand')

        self.declare_partials(['delta_quantity'], ['rate'], rows=np.zeros(nn), cols=range(nn))
        self.declare_partials(['delta_quantity'], ['upper_limit'])
        self.declare_partials(['delta_quantity'], ['lower_limit'])

    def compute(self, inputs, outputs):
        n_int = self.options['num_intervals']
        force_signs = self.options['force_signs']
        nn = (n_int*2 + 1)
        dts = (inputs['upper_limit'] - inputs['lower_limit']) / (nn - 1)
        if force_signs:
            rate = inputs['rate']
            check = rate * dts
            debug = check < 0
            if np.sum(inputs['rate'] * dts < 0) > 0:
                pass
                #raise ValueError('The numeric integration for this segment is failing because the signs of the rates do not match the sign of the diff interval, e.g. negative acceleration is occuring in an integration wrt velocity')
        int_q, delta_q = simpson_integral(dts,inputs['rate'],n_segments=1,n_simpson_intervals_per_segment=n_int)
        outputs['delta_quantity'] = int_q

    def compute_partials(self, inputs, J):
        n_int = self.options['num_intervals']
        nn = (n_int*2 + 1)
        dts = (inputs['upper_limit'] - inputs['lower_limit']) / (nn - 1)
        wrt_q, wrt_dt = simpson_partials(dts,inputs['rate'],n_segments=1,n_simpson_intervals_per_segment=n_int)
        ddQdq = sp.csr_matrix((wrt_q[2],(wrt_q[0],wrt_q[1])))
        ddQddt = sp.csr_matrix((wrt_dt[2][0],(wrt_dt[0][0],wrt_dt[1][0])))

        J['delta_quantity','rate'] = np.asarray(ddQdq.sum(axis=0)).flatten()
        #partial derivative wrt the time interval
        dQddt = ddQddt.sum()

        J['delta_quantity','lower_limit'] = -dQddt / (nn-1)
        J['delta_quantity','upper_limit'] = dQddt / (nn-1)