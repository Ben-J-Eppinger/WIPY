import numpy as np
import scipy
from obspy import Stream
from copy import deepcopy
from wipy.wipy_utils import utils
from wipy.preprocess.mute import calc_offset


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

        if np.sum(np.abs(obs_tr)) == 0:
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


def FATT(syn, obs, taper):

    # get picks from observed data **note the observed data must be properly encoded with the picks
    obs_picks = []
    for trace in obs.traces:
        obs_picks.append(trace.data[0])
    obs_picks = np.array(obs_picks)

    # picks from the synthetic data
    syn_picks = utils.pick_synthetic_data(syn)

    # initialize the time variables
    dt = obs[0].stats.delta
    Nt = obs[0].stats.npts
    t = np.arange(0, Nt*dt, dt)

    # intialize the adjoint source and residuals
    adj = deepcopy(syn)
    residuals = []

    # loop through traces and compute the adjoint sources and residuals
    for tr_ind in range(len(syn.traces)):

        # if the observed pick is zero, then there is no pick here so the adjoint source and residual are zero
        if obs_picks[tr_ind] == 0:
            adj.traces[tr_ind].data = 0.0*syn.traces[tr_ind].data
            residuals.append(0.0)
        
        # otherwise, compute the adjoint source and residual
        else:
            # get the the travel time residual
            delta_T = obs_picks[tr_ind] - syn_picks[tr_ind]

            # create the adjoint source
            adj_tr = 0.0*syn.traces[tr_ind].data
            adj_tr[1:] = np.diff(syn.traces[tr_ind].data)
            adj_tr[0] = 0.0
            E = np.sum(adj_tr**2)*dt
            adj_tr *= delta_T / E
            # compute time mask for adjoint source
            t0: float = syn_picks[tr_ind] + taper
            t1: float = t0 + taper 
            mask: np.ndarray = np.piecewise(
                t, 
                [t <= t0, t > t0, t > t1],
                [1, lambda t: np.cos(np.pi*(t-t0)/(2*taper)), 0],
                )
            adj_tr *= mask
            # assign adjoint source in stream object
            adj.traces[tr_ind].data = adj_tr

            residuals.append(0.5*(delta_T**2))

    return adj, residuals


def dispersion(syn: Stream, obs: Stream, min_rec: int, fmin: float, fmax: float, safe_guard: float) -> tuple[Stream, list[float]]:

    ########################
    ### Define Variables ###
    ########################

    Nt: int = syn[0].stats.npts
    dt: float = syn[0].stats.delta
    df: float = 1.0/(Nt*dt)
    Nx: int = len(syn.traces)

    nfmin: int = int(np.floor(fmin/df))     # index of the minimum frequency
    nfmax: int = int(np.ceil(fmax/df))      # index of the maximum frequency
    nfcount: int = nfmax-nfmin+1            # number of frequencies we will measure the dispersion difference at

    # initialize dispersion difference array
    dispersion_diff: np.ndarray = np.zeros((2,nfcount))       # first row is for LHS dispersion difference while the second row is for the RHS

    # Preform Time domain Fourier Transform on synthetic and observed data
    syn_ft: np.ndarray = np.fft.rfft(syn).T
    obs_ft: np.ndarray = np.fft.rfft(obs).T

    # compute the offsets of the data
    offsets = []
    for trace in syn.traces:
        offsets.append(calc_offset(trace))
    dx = np.mean(np.abs(np.diff(offsets)))

    # compute the index of the source in the data
    src_ind = np.argmin(offsets)

    # get the number of recievers on either side of the source
    num_rec_left = src_ind 
    num_rec_right = Nx - src_ind

    #####################################
    ### Compute Dispersion Difference ###
    #####################################

    # compute the misfit of the LHS DATA
    if num_rec_left > min_rec:
        dk = 1.0/(num_rec_left*dx)

        # loop over frequencies
        for freq_ind, i in zip(range(nfmin, nfmax+1), range(nfcount)):

            # get LHS fk spectrum for the frequency
            syn_fk_lhs = np.fft.fft(syn_ft[freq_ind, src_ind-1::-1])
            obs_fk_lhs = np.fft.fft(obs_ft[freq_ind, src_ind-1::-1])

            # extract half the spectrum and normaliz
            syn_fk_lhs_half = syn_fk_lhs[num_rec_left//2:]
            syn_fk_lhs_half /= np.max(np.abs(syn_fk_lhs_half))
            obs_fk_lhs_half = obs_fk_lhs[num_rec_left//2:]
            obs_fk_lhs_half /= np.max(np.abs(obs_fk_lhs_half))

            # compute the cross correlation function at this frwquency
            xcor = np.real(np.correlate(obs_fk_lhs_half, syn_fk_lhs_half, "full"))
            dd = (np.argmax(xcor)-(num_rec_left//2)+1)*dk
            if np.abs(dd) > safe_guard:
                dispersion_diff[0][i] = 0.0
            else:
                dispersion_diff[0][i] = dd

    # compute the misfit of the LHS DATA
    if num_rec_right > min_rec:
        dk = 1.0/(num_rec_right*dx)

        # loop over frequencies
        for freq_ind, i in zip(range(nfmin, nfmax+1), range(nfcount)):

            # get RHS fk spectrum for the frequency
            syn_fk_rhs = np.fft.fft(syn_ft[freq_ind, src_ind:])
            obs_fk_rhs = np.fft.fft(obs_ft[freq_ind, src_ind:])

            # extract half the spectrum and normaliz
            syn_fk_rhs_half = syn_fk_rhs[num_rec_right//2:]
            syn_fk_rhs_half /= np.max(np.abs(syn_fk_rhs_half))
            obs_fk_rhs_half = obs_fk_rhs[num_rec_right//2:]
            obs_fk_rhs_half /= np.max(np.abs(obs_fk_rhs_half))

            # compute the cross correlation function at this frwquency
            xcor = np.real(np.correlate(obs_fk_rhs_half, syn_fk_rhs_half, "full"))
            dd = (np.argmax(xcor)-(num_rec_right//2)+1)*dk
            if np.abs(dd) > safe_guard:
                dispersion_diff[1][i] = 0.0
            else:
                dispersion_diff[1][i] = dd

    ###############################
    ### Compute Adjoint Sources ###
    ###############################

    # initialize vairaibles use for adjoint source computation
    syn_ft_adj: np.ndarray = 0.0*syn_ft
    Areal: np.ndarray = np.zeros(nfcount)
    adj: Stream = deepcopy(syn)

    # Compute adjoint source for LHS
    if num_rec_left > min_rec:
        # compute amplitude adjustment factors
        for freq_ind, i in zip(range(nfmin, nfmax+1), range(nfcount)): 
            A = 0.0
            for rec_ind in range(0, src_ind):
                A -= 2 * np.pi*offsets[rec_ind]**2 * syn_ft[freq_ind][rec_ind]*np.conj(syn_ft[freq_ind][rec_ind])
            Areal[i] = np.real(A)

        factor=0.1*np.max(np.abs(Areal))
        # compute adjoint source-time functions
        for freq_ind, i in zip(range(nfmin, nfmax+1), range(nfcount)):
            if np.abs(Areal[i]) >= 0.0:
                for rec_ind in range(0, src_ind): 
                    syn_ft_adj[freq_ind][rec_ind] = (1j*dispersion_diff[0][i]*offsets[rec_ind])*syn_ft[freq_ind][rec_ind]/(Areal[i]-factor)

    # Compute adjoint source for RHS
    if num_rec_right > min_rec:
        # compute amplitude adjustment factors
        for freq_ind, i in zip(range(nfmin, nfmax+1), range(nfcount)): 
            A = 0.0
            for rec_ind in range(src_ind, Nx):
                A -= 2 * np.pi*offsets[rec_ind]**2 * syn_ft[freq_ind][rec_ind]*np.conj(syn_ft[freq_ind][rec_ind])
            Areal[i] = np.real(A)

        factor=0.1*np.max(np.abs(Areal))
        # compute adjoint source-time functions
        for freq_ind, i in zip(range(nfmin, nfmax+1), range(nfcount)):
            if np.abs(Areal[i]) >= 0.0:
                for rec_ind in range(src_ind, Nx): 
                    syn_ft_adj[freq_ind][rec_ind] = (1j*dispersion_diff[1][i]*offsets[rec_ind])*syn_ft[freq_ind][rec_ind]/(Areal[i]-factor)

    # Tranform adjoint source back to time domain
    adj_temp = np.fft.irfft(syn_ft_adj, axis=0)
    # Put the adjoint sources into a stream object
    for i, trace in enumerate(adj.traces):
        trace.data = adj_temp[:, i]
    
    residuals = dispersion_diff.flatten()**2
    residuals = residuals.tolist()
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


def Wasserstein(syn, obs, b):

    # initialize adjoint source and residuals
    adj = deepcopy(syn)
    residuals = []

    # initialize time scheme
    dt = syn.traces[0].stats.delta
    Nt = len(syn.traces[0].data)
    t = np.arange(0, Nt)*dt

    # loop through traces
    for tr_ind in range(len(syn.traces)):

        # get observed and synthetic traces
        obs_tr = obs.traces[tr_ind].data
        syn_tr = syn.traces[tr_ind].data

        # normalize the data into the PDFs f and g using softplus scaling
        # f = np.log(1.0 + np.exp(b*syn_tr))
        # g = np.log(1.0 + np.exp(b*obs_tr))
        f = np.exp(b*syn_tr)
        g = np.exp(b*obs_tr)
        f /= np.sum(f)*dt
        g /= np.sum(g)*dt

        # compute the CDFs F and G
        F = np.cumsum(f)*dt
        G = np.cumsum(g)*dt

        # compute t_star
        G_inv_interp = scipy.interpolate.interp1d(
            G, 
            t, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(t[0], t[-1])
        )
        t_star = G_inv_interp(F)

        # compute g_star    
        g_interp = scipy.interpolate.interp1d(
            t, 
            g, 
            kind='linear',
            bounds_error=False, 
            fill_value=(g[0], g[-1])
        )
        g_star = g_interp(t_star)

        # compute the residual
        resid = np.sum((np.abs(t - t_star)**2)*f)*dt
        residuals.append(resid)

        # compute adjoint source
        p1 = (t-t_star)*(f/g_star)
        p = np.zeros(Nt)

        for i in range(len(p1)): 
            p[i] = np.sum(p1[i:])*dt

        adj_tr = (t-t_star)**2  -2*p
        adj[tr_ind].data = adj_tr

    return adj, residuals


def GSOT(syn, obs, freq_lim, eta, p=2):

    # initialize adjoint source and residuals
    adj = deepcopy(syn)
    residuals = []

    # initialize time scheme
    dt = syn.traces[0].stats.delta
    Nt = len(syn.traces[0].data)
    t = np.arange(0, Nt)*dt

    # initialize time scheme
    dt = syn.traces[0].stats.delta
    Nt = len(syn.traces[0].data)
    t = np.arange(0, Nt)*dt

    # come up with downs sample scheme
    dt_dsr = 1/(2*freq_lim)
    dsr = int(dt_dsr/dt)
    t_ds = t[::dsr]

    for tr_ind in range(len(syn.traces)):

        # downsample data
        syn_tr = syn.traces[tr_ind].data[::dsr]
        obs_tr = obs.traces[tr_ind].data[::dsr]

        # calculate cost matrix
        C = np.zeros((len(t_ds), len(t_ds)))
        for i in range(len(t_ds)):
            C[:,i] = eta*np.abs(t_ds - t_ds[i])**p + np.abs(syn_tr - obs_tr[i])**p

        # solve assignment problem
        sigma = scipy.optimize.linear_sum_assignment(C, maximize=False)[1]
        sigma = [int(i) for i in sigma]

        # compute t_sigma and obs_sigma
        t_sigma = t_ds[sigma]
        obs_sigma = obs_tr[sigma]

        # calculate misift 
        resid = np.sum(eta*np.abs(t_ds-t_sigma)**p + np.abs(syn_tr - obs_sigma)**p)*t_ds
        residuals.append(resid)

        # calculate adjoint source
        adj_tr = p*np.abs(syn_tr - obs_sigma)**(p-2) * (syn_tr - obs_sigma)

        # create interpolation object for the adjoint source
        adj_interp = scipy.interpolate.interp1d(
            t_ds,
            adj_tr, 
            kind="cubic", 
            bounds_error=False,
            fill_value=(adj_tr[0], adj_tr[-1])
        )

        # resample adjoint source with the original time scheme
        adj_tr = adj_interp(t)
        adj[tr_ind].data = adj_tr   

    return adj, residuals


##################################################
### Helper Functions for Misfits using the CWT ###
##################################################

ricker = lambda t, f0: (1.0 - 2.0*(np.pi**2)*(f0**2)*((t)**2))*np.exp(-(np.pi**2)*(f0**2)*((t)**2))

def complex_ricker(t, f0):
   w = ricker(t, f0)
   w = np.fft.fft(w)
   w[len(w)//2:] = 0.0 + 0.0j
   w = np.fft.ifft(w)
   return w


def CWT(f, freq_0, S, mother, t):
   """
   Compute the wavelet transform of a signal, f, at scales S and frequency f0
   """
   # set some important variables
   dt = t[1] - t[0]   
   t_half = np.max(t)/2

   # compute the fourier transfor of the input signal, f
   f_hat = np.fft.fft(f)

   # initialize the wavelet transform
   W = np.zeros((len(S), len(f)), dtype=complex)

   # loop through the scales
   for i, s in enumerate(S): 
       # compute the wavelet at the scale s
       w = mother((t-t_half)/s, freq_0)

       # compute the fourier transform of the wavelet
       w_hat = np.fft.fft(w)

       # populate the wavelet transform matrix
       W[i, :] = np.fft.ifft(f_hat * np.conj(w_hat))*(1/np.sqrt(s))

       # shift W by t_half
       W[i, :] = np.roll(W[i, :], int(t_half/dt))

   return W


def WavePhase(syn, obs, max_freq, min_freq, mother = complex_ricker):
    
    # initialize adjoint source and residuals
    adj = deepcopy(syn)
    residuals = []

    # initialize time scheme
    dt = syn.traces[0].stats.delta
    Nt = len(syn.traces[0].data)
    t = np.arange(0, Nt)*dt

    # come up with down sampled scheme
    dt_dsr = (1/(2*max_freq))/4.0
    dsr = int(dt_dsr/dt)
    t_ds = t[::dsr]
    t_cent = t_ds - np.mean(t_ds)

    # compute scale factors 
    S = max_freq/np.linspace(max_freq, min_freq, len(t_ds)//4)

    # set mother wavelet
    mother = mother

    # compute weighting matrices 
    weight = np.zeros((len(S), len(t_ds)), dtype=complex)
    K = np.zeros((len(S), len(t_ds)), dtype=complex)
    for i, s in enumerate(S): 
            weight[i, :] = s**(-1.5)
            K[i, :] = s**(-2.0)

    # set parameters for Gamma function
    eps = 100.0
    eta = 0.05

    for tr_ind in range(len(syn.traces)):

        if np.sum(np.abs(obs.traces[tr_ind].data)) == 0:
            adj[tr_ind].data = 0.0*syn.traces[tr_ind].data
            residuals.append(0.0)

        else:
            # downsample data
            syn_tr = syn.traces[tr_ind].data[::dsr]
            obs_tr = obs.traces[tr_ind].data[::dsr]

            # compute the CFTs of observed and synthetic data
            W_obs = CWT(obs_tr, max_freq, S, mother, t_ds)
            W_syn = CWT(syn_tr, max_freq, S, mother, t_ds)

            # compute the phase difference
            Phase = np.log(W_syn/W_obs).imag
            C = np.abs(np.conj(W_obs)*W_syn)
            C /= np.max(C)
            Gamma = 1 / (1 + np.exp(-eps*(C-eta)))
            Phase *= Gamma
            P = np.conj(W_syn) * W_syn

            # compute adjoint source
            ADJ = Phase * np.imag(W_syn)
            adj_tr = -np.trapz(weight * ADJ, S, axis=0).real

            # create interpolation object for adjoint source
            adj_interp = scipy.interpolate.interp1d(
                t_ds,
                adj_tr, 
                kind="cubic", 
                bounds_error=False,
                fill_value=(adj_tr[0], adj_tr[-1])
            )

            # sample adjoint source at original time scheme
            adj_tr = adj_interp(t)
            adj[tr_ind].data = adj_tr

            # compute residual
            R = K * P * (Phase**2)
            R = np.trapz(R, S, axis=0).real  
            resid = np.sum(R)*dt
            residuals.append(resid)

    return adj, residuals
  
  