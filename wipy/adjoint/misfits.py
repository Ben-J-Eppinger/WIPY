import numpy as np
from obspy import Stream
from copy import deepcopy

def L2_norm(syn: Stream, obs: Stream) -> tuple[Stream, list[float]]:
    """
    Calculates the classical L2 waveform difference and corresponding adjoint source
    inputs: 
        syn: the synthetic data for a shot gather
        obs: the observed data for a shot gather
    outputs:
        adj: the corresponding adjoint source for this synthetic-observed pair
        residuls: the L2 waveform misfits for this synthetic-observed pair
    """
    
    adj = deepcopy(syn)
    residuals = []
    dt = syn.traces[0].stats.delta

    for tr_ind in range(len(syn.traces)):
        adj.traces[tr_ind].data = syn.traces[tr_ind].data - obs.traces[tr_ind].data
        residuals.append(np.sum(adj.traces[tr_ind].data**2)*dt)

    return adj, residuals


def NC_norm(syn: Stream, obs: Stream) -> tuple[Stream, list[float]]:
    """
    Calculates the classical NC_norm (normalized correlation AKA global correlation) misfit and 
    corresponding adjoint source (Choi and Alkhalifah, 2012)
    inputs: 
        syn: the synthetic data for a shot gather
        obs: the observed data for a shot gather
    outputs:
        adj: the corresponding adjoint source for this synthetic-observed pair
        residuals: the NC misfits for this synthetic-observed pair
    Notes:
        we have added a 1/dt term to the misfit so the the misfits are by convention positive
        while this is a change from the original paper, it should not effect the behavior of 
        the inversion (e.g., Eppinger et al., 2024)
    """

    adj = deepcopy(syn)
    residuals = []
    dt = syn.traces[0].stats.delta

    for tr_ind in range(len(syn.traces)):

        obs_tr = obs.traces[tr_ind].data
        syn_tr = syn.traces[tr_ind].data

        if np.sum(np.abs(obs)) == 0:
            adj.traces[tr_ind].data = 0*obs_tr
            residuals.append(0.0)
        
        else:
            norm_syn = np.linalg.norm(syn_tr, ord=2)
            syn_hat = syn_tr/norm_syn

            norm_obs = np.linalg.norm(obs_tr, ord=2)
            obs_hat = obs_tr/norm_obs

            inner = np.dot(syn_hat, obs_hat)

            adj.traces[tr_ind].data = (inner*syn_hat - obs_hat)/norm_syn
            residuals.append(1-inner)

    return adj, residuals


def backproject(syn: Stream, obs: Stream) -> tuple[Stream, list[float]]:
    """
    Compute adjoint sources that will backproject the observed data (veclocity rather than displacement)
    inputs: 
        syn: the synthetic data for a shot gather
        obs: the observed data for a shot gather
    outputs:
        adj: the corresponding adjoint source for this synthetic-observed pair
        residuals: is unimportant
    """

    adj = deepcopy(syn)
    residuals = []
    dt = syn.traces[0].stats.delta

    for tr_ind in range(len(syn.traces)):

        adj.traces[tr_ind].data[:-1] = np.diff(obs.traces[tr_ind].data)     # compute the particle velocity from the displacement
        adj.traces[tr_ind].data[-1] = 0

        residuals.append(0.0)

    return adj, residuals
