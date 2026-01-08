import matplotlib.pyplot as plt
import numpy as np
import mne
from colorsys import hls_to_rgb

def xyz_to_rgb(x, y, z):
    """
    Convert (X, Y, Z) coordinates on the unit sphere into an RGB color.
    - X and Z determine the Hue (angle in X-Z plane).
    - Y determines the Lightness.
    - Saturation is fixed at 1 for full color.

    Returns: (R, G, B) tuple in range [0,1].
    """
    # Compute Hue from X, Y (atan2 maps angle from [-pi, pi])
    hue = -(np.arctan2(z, x) / (2 * np.pi)) % 1 # Normalize to [0, 1]

    # Lightness directly from Z (rescale from [-1, 1] to [0, 1])
    lightness = ((1+y)/2)

    if hasattr(hue, '__iter__') or hasattr(lightness, '__iter__'):
        return [hls_to_rgb(h, l, 1) for h, l in zip(hue, lightness)]
    else:
        return hls_to_rgb(hue, lightness, 1)  # Saturation fixed at 1
    

def complex_topomaps(wdata, x,y):
    """
    Plot topographic maps for complex-valued data across multiple samples and sensors.

    For each sample (row) in `wdata`, this function generates four topomaps:
        - Phase (angle of the complex values)
        - Amplitude (magnitude of the complex values)
        - Real part
        - Imaginary part

    The 2D sensor positions are derived from the provided 3D coordinates using a projection (see code for details).

    Parameters
    ----------
    wdata : np.ndarray, shape (n_samples, n_sensors)
        Complex-valued data for each sample and sensor.
    x : array-like, shape (n_sensors,)
        X-coordinates of each sensor in 3D space.
    y : array-like, shape (n_sensors,)
        Y-coordinates of each sensor in 3D space.

    Returns
    -------
    None
        Displays the topomaps using matplotlib.
    """

    pos = np.c_[y,x]

    # calculate v-values for topomaps
    ampl_max = np.max(np.abs(wdata))
    xy_max = max(np.max(np.abs(np.real(wdata))), np.max(np.abs(np.imag(wdata))))

    n = wdata.shape[0]
    fig, axs = plt.subplots(n, 4, figsize=(10, n*2))
    for i, ax in enumerate(axs.flat):
        # set the current axis
        row = i // 4
        sample = wdata[row,:]
        if i % 4 == 0:
            mne.viz.plot_topomap(np.angle(sample), pos, 
                                cmap='hsv',vlim=[-np.pi, np.pi], image_interp='nearest', axes=ax, show=False, contours=0)
            if i//4 == 0:
                ax.set_title('Phase')
        elif i % 4 == 1:
            mne.viz.plot_topomap(np.abs(sample), pos, cmap='Reds',axes=ax, show=False, vlim=[0,ampl_max])
            if i//4 == 0:
                ax.set_title('Amplitude')
        elif i % 4 == 2:
            mne.viz.plot_topomap(np.real(sample), pos, axes=ax, show=False, vlim=[-xy_max,xy_max],)
            if i//4 == 0:
                ax.set_title('Real')
        elif i % 4 == 3:
            mne.viz.plot_topomap(np.imag(sample), pos, axes=ax, show=False, vlim=[-xy_max,xy_max],)
            if i//4 == 0:
                ax.set_title('Imaginary')
            
    plt.tight_layout()
    plt.show()