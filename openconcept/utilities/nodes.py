from __future__ import division


def compute_num_nodes(n_int_per_seg, mission_segments):
    n_segments = len(mission_segments)
    nn = (2*n_int_per_seg+1)
    nn_tot_to = nn*3 +2 #v0v1,v1vr,v1v0, vtr, v2
    nn_tot_m = nn*n_segments
    nn_tot=nn_tot_to+nn_tot_m
    return nn, nn_tot_to, nn_tot_m, nn_tot