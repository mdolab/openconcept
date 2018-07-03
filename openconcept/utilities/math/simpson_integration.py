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

    returns:
    delta_q = amount of q accumulated during each interval
    int_q = total amount of q accumulated during all phases

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

        returns:
        delta_q = amount of q accumulated during each interval
        int_q = total amount of q accumulated during all phases

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






if __name__ == "__main__":
    v1 = 2*np.sin(np.linspace(0,1,11)*np.pi)
    v2 = np.sin(np.linspace(0,1,11)*np.pi)
    q = np.concatenate([v1,v2])
    int_q, delta_q = simpson_integral([np.pi/10,np.pi/10],q,n_segments=2,n_simpson_intervals_per_segment=5)
    print(int_q)
    print(delta_q)
    int_q, delta_q_2x = simpson_integral_every_node([np.pi/10,np.pi/10],q,n_segments=2,n_simpson_intervals_per_segment=5)
    print(delta_q_2x)
    wrt_q, wrt_dt = simpson_partials([np.pi/10,np.pi/10],q,n_segments=2,n_simpson_intervals_per_segment=5)
    wrt_q_2x, wrt_dt_2x = simpson_partials_every_node([np.pi/10,np.pi/10],q,n_segments=2,n_simpson_intervals_per_segment=5)

    print(wrt_q)
    print(wrt_q_2x)
    print(wrt_dt)
    print(wrt_dt_2x)