import numpy as np

def L2_norm(syn: np.ndarray, obs: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
    """
    Calculates the classical L2 waveform difference and corresponding adjoint source
    inputs: 
        syn: the synthetic data for a trace stored in a 1xNt numpy array
        obs: the observed data for a trace stored in a 1xNT numpy array
    outputs:
        adj: the corresponding adjoint source for this synthetic-observed pair
        misfit: the L2 waveform misfit for this synthetic-observed pair
    """
    
    adj = syn - obs
    misfit = np.sum(adj**2)*dt

    return adj, misfit


def NC_norm(syn: np.ndarray, obs: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
    """
    Calculates the classical NC_norm (normalized correlation AKA global correlation) misfit and 
    corresponding adjoint source (Choi and Alkhalifah, 2012)
    inputs: 
        syn: the synthetic data for a trace stored in a 1xNt numpy array
        obs: the observed data for a trace stored in a 1xNT numpy array
    outputs:
        adj: the corresponding adjoint source for this synthetic-observed pair
        misfit: the NC  misfit for this synthetic-observed pair
    Notes:
        we have added a 1/dt term to the misfit so the the misfits are by convention positive
        while this is a change from the original paper, it should not effect the behavior of 
        the inversion (e.g., Eppinger et al., 2024)
    """

    if np.sum(np.abs(obs)) == 0:
        adj = 0*obs
        misfit = 0.0
    
    else:
        norm_syn = np.linalg.norm(syn, ord=2)
        syn_hat = syn/norm_syn

        norm_obs = np.linalg.norm(obs, ord=2)
        obs_hat = obs/norm_obs

        inner = np.dot(syn_hat, obs_hat)

        adj = (inner*syn_hat - obs_hat)/norm_syn
        misfit = 1-inner

    return adj, misfit

