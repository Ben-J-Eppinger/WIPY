import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.interpolate import griddata
import obspy
from obspy.core.stream import Stream

def read_fortran_binary(file_path: str) -> np.array:
    """
    Reads the fortran binary files specfems uses.
    Note that the first and last elements of data encode the length of the array.
    We remove these elements here because when dtype "float32" is used the values 
    are meaningless. When read using dtype "int32", dat[0] = dat[-1] = 4*(len(dat)-2).
    inputs: 
        file_path: the absaolute path to the binary file
    outputs: 
        dat: a NumPy array with the values of the binary file
    """

    dat: np.array = np.fromfile(file_path, dtype='float32')
    dat = dat[1:-1]
    return dat 


def write_fortran_binary(file_path: str, dat: np.array) -> None:
    """
    Writes fortran binary files that specfem can use.
    Note how we compute the buffer values (buf) and write 
    them into the binary files at either end of the data array (dat). 
    inputs: 
        file_path: the absolute path of the file to be written
        dat: the data array (usually Nx1) being written as a binary file
    """
    
    buf = np.array([4 * len(dat)], dtype="int32") 
    dat = np.array(dat, dtype="float32") 

    with open(file_path, "wb") as file: 
        buf.tofile(file) 
        dat.tofile(file) 
        buf.tofile(file) 


def load_model(model_path: str, pars: list[str]) -> dict[str: np.array]:
    """
    loads a model from binary files.
    inputs:
        model_path: the absaolute path of the folder with the binary files
        pars: the parameters from the model that will be loaded (e.g., "x", "rho", "vp", etc.)
    outputs: 
        model: a dictionary representation of a model with keys that map parameters
        to NumPy arrays
    """

    model: dict = {}

    for par in pars:
        path: list[str] = "/".join([model_path, 'proc000000_' + par + '.bin']) 
        model[par] = read_fortran_binary(path)

    return model


def write_model(model_path: str, model: dict[str: np.array]) -> None: 
    """
    Writes a dictionary representation of a model to binary files
    inputs: 
        model_path: the absolute path of the directory in which the binary files will be 
        written 
        model: the dictionary representation of a model that will be written as binary files
    """
    
    for key in model.keys():
        path: list[str] = "/".join([model_path, 'proc000000_' + key + '.bin'])
        write_fortran_binary(path, model[key])


def plot_model_fast(model: dict[str: np.ndarray], spac: float, par: str, bounds: list[float] = None, cmap="turbo") -> None:
    """
    Quick plotting funciton to display gridded models
    inputs:
        m: a dictionary representation of a model
        spac: the grid spacing for the plot
        par: the parameter to be plotted
        bounds: the minimum and maximum values displayes on the color scale
        cmap: the color map used
    outputs:
        fig, ax: Pyplot figure and axis handles for the plot
    """

    x_vec = np.arange(
        start=np.min(model['x']),
        stop=np.max(model['x'])+spac,
        step=spac
        )

    z_vec = np.arange(
        start=np.min(model['z']),
        stop=np.max(model['z'])+spac,
        step=spac
        )

    grid_x, grid_z = np.meshgrid(x_vec, z_vec,)

    f = griddata(
        points=(model['x'], model['z']),
        values=model[par],
        xi=(grid_x, grid_z),
        method='linear',
    )

    if bounds is None:
        bounds = [np.min(model[par]), np.max(model[par])]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    pcm = ax.pcolormesh(
        grid_x,
        grid_z,
        f,
        shading='auto',
        cmap=cmap,
        vmin=bounds[0],
        vmax=bounds[1]
    )
    fig.colorbar(pcm, ax=ax, shrink=f.shape[0]/f.shape[1], extend='both')
    ax.set_aspect(1)

    return fig, ax


def plot_model(m: dict[str: np.ndarray], ne: int, par: str, bounds: list[float]=None, cmap="turbo"):
    """
    Precicely plots the SEM mesh elements of a given model
    inputs:
        m: a dictionary representation of a model
        ne: the number of control points per an element 
        par: the parameter to be plotted
        bounds: the minimum and maximum values displayes on the color scale
        cmap: the color map used
    outputs:
        fig, ax: Pyplot figure and axis handles for the plot
    """

    Ne = int(len(m[par])/ne)

    if bounds is None:
        bounds = [np.min(m[par]), np.max(m[par])]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for i in range(Ne):

        inds = np.arange(i*ne, (i+1)*ne)

        pcm = ax.pcolormesh(
            m["x"][inds].reshape((int(np.sqrt(ne)),int(np.sqrt(ne)))), 
            m["z"][inds].reshape((int(np.sqrt(ne)),int(np.sqrt(ne)))), 
            m[par][inds].reshape((int(np.sqrt(ne)),int(np.sqrt(ne)))),
            shading="gouraud",
            cmap=cmap,
            vmin=bounds[0],
            vmax=bounds[1],
            )

    shrink = (np.max(m["z"]) - np.min(m["z"]))/(np.max(m["x"]) - np.min(m["x"]))
    fig.colorbar(pcm, ax=ax, shrink=shrink, extend='both')
    ax.set_aspect(1)

    return fig, ax


def plot_traces(stream: Stream, gain: int = 1, line_spec: str = "k-", deci : int = 1) -> None:
    """
    plot obspy stream objects for shot gathers
    """

    dt = stream.traces[0].stats.delta
    T = stream.traces[0].stats.npts*dt
    t = np.arange(start=0, stop=T, step=dt)

    for idx, trace in enumerate(stream.traces):
        if idx % deci == 0:
            plt.plot(
                trace.data*gain + trace.stats.su.trace_header.group_coordinate_x,
                t, 
                line_spec,
            )


def plot_image(stream: Stream, clip: float, cmap="gray") -> None: 
    """
    plot obspy stream objects for shot gathers as an image
    """

    Nr = len(stream.traces)
    Nt = stream.traces[0].stats.npts

    I = np.zeros((Nt, Nr))

    dt = stream.traces[0].stats.delta
    t_max = stream.traces[0].stats.npts*dt
    t = np.arange(start=0, stop=t_max, step=dt)

    X, T = np.meshgrid(np.arange(0, Nr, 1), t)

    for i in range(Nr):
        I[:, i] = stream.traces[i].data

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    pcm = ax.pcolormesh(X, T, I, shading="auto", cmap=cmap, 
                        vmax=np.max(np.abs(I))*clip,
                        vmin=-np.max(np.abs(I))*clip)
    
    ax.set_ylabel("time [s]")
    ax.set_xlabel("receiver")

    fig.colorbar(pcm, ax=ax, extend='both')

    return fig, ax


def grid_vect(x: np.ndarray, z: np.ndarray, f: np.ndarray):
    """"
    Grids a vector, f, based on the x and z coordinates of the vector.
    inputs:
        x: x coordinates of the vector
        z: z coordinates of the vector
        f: vector to be gridded
    outputs:
        f: gridded vector
        grid_x: x coordinates of the grid
        grid_z: z coordinates of the grid
        spac: spacing between grid points
    """

    # get average spaceing between points
    delta_x = np.diff(x)
    delta_x = delta_x[delta_x != 0]

    delta_z = np.diff(z)
    delta_z = delta_z[delta_z != 0]

    spac = np.mean([np.median(delta_x), np.median(delta_z)])

    # grid the vector
    x_vec = np.arange(
        start=np.min(x),
        stop=np.max(x)+spac,
        step=spac
        )

    z_vec = np.arange(
        start=np.min(z),
        stop=np.max(z)+spac,
        step=spac
        )

    grid_x, grid_z = np.meshgrid(x_vec, z_vec,)

    f = scipy.interpolate.griddata(
        points=(x, z),
        values=f,
        xi=(grid_x, grid_z),
        method='nearest',
    )

    return f, grid_x, grid_z, spac



def smooth_par(m: dict[str: np.array], par: str, sigma_x, sigma_y) -> np.ndarray:
    """
    Smooths a parameter from a dictionary representation of a model.
    inputs:
        m: dictionary representation of a model
        par: parameter to be smoothed
        sigma_x: sigma value for the x direction
        sigma_y: sigma value for the y direction
    outputs:
        g: smoothed field as a 1D array
    """

    # grid the vector
    f, grid_x, grid_z, spac = grid_vect(m["x"], m["z"], m[par])    

    # smooth the gridded field
    f_smooth = scipy.ndimage.gaussian_filter(f, sigma=(sigma_y/spac, sigma_x/spac))

    # interpolate the smoothed gridded field back onto the vector 
    interp = scipy.interpolate.LinearNDInterpolator(list(zip(grid_x.flatten(), grid_z.flatten())), f_smooth.flatten())
    g = interp(m['x'], m['z'])

    return g


def pick_synthetic_data(data: Stream, tol: float = 10**(-3)) -> np.ndarray:
    """
    Naively picks data based on where the amplitude squared is larger than some tollerence level
    For this reason, it is recomended to use this function on noise free synthetic data only
    inputs:
        data: an obspy stream object for a shot gather of data
        tol: amplitude sensitivity tollerence level
    outputs: 
        picks: a numpy array of picks where the pick index corresponds to the trace index
    """
    
    dt = data.traces[0].stats.delta
    picks = []
    for trace in data.traces:
        abs_data = np.abs(trace.data)
        A = np.max(abs_data)
        bool =  abs_data > tol*A
        picks.append(np.argmax(bool)*dt)

    return np.array(picks)
        