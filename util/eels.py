import numpy as np
import matplotlib.pyplot as plt

# EELS fitting methods with help from Jian Huang
def eels_get_nearest_index(arr, val):
    """
    Gives index of array closest to desired value
    """    
    
    id = np.argmin(np.abs(arr - val))
    return id

def eels_normalize(arr):
    """
    Converts an array of arbitrary values to a 0-1 range
    """    
    
    arr = np.asarray(arr)
    M=np.amax(arr);m=np.amin(arr)
    narr = (arr-m)/(M-m)
    return narr

def autoscale_ylim(ax, xdata, ydata, padding=0.01):
    x0, x1 = ax.get_xlim()
    mask = (xdata >= x0) & (xdata <= x1)
    y_visible = ydata[mask]
    if len(y_visible) > 0:
        ymin, ymax = np.min(y_visible), np.max(y_visible)
        pad = padding * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

def power_fit(eels_signal, fit_ranges, get_plot=False, plot_range=None):
    """
    Fits a power law background to EELS

    eels_signal: pure energy loss (HyperSpy spectrum object) with dimensions of: (|n)
        for an image, the imput should be a sum over the spatial dimensions
    fit_ranges: 2k size numpy array of energy loss ranges to fit (eV)
    plot_range: [low_energy, high_energy] range to plot (eV)
    """   

    energies = eels_signal.axes_manager.signal_axes[0].axis
    intensities = eels_signal.data

    energies_fit = []
    intensities_fit= []
    for (low, high) in fit_ranges:
        low_index = eels_get_nearest_index(energies, low)
        high_index = eels_get_nearest_index(energies, high)

        energies_fit.append(energies[low_index:high_index])
        intensities_fit.append(intensities[low_index:high_index])
    energies_fit = np.concatenate(energies_fit)
    intensities_fit = np.concatenate(intensities_fit) 
    
    # Fitting: I = AE^(-r) <=> logI = logA - rlogE
    energies_log = np.array(np.log(energies_fit[np.where(intensities_fit > 0)]))
    intensities_log = np.array(np.log(intensities_fit[np.where(intensities_fit > 0)]))
    minus_r, logA = np.polyfit(energies_log, intensities_log, 1)

    first_low = eels_get_nearest_index(energies, fit_ranges[0][0])
    ret_energies = energies[first_low:]
    ret_background = np.exp(logA) * ret_energies**(minus_r)
    ret_intensities = intensities[first_low:] - ret_background

    if get_plot:
        fig, ax = plt.subplots(3, 1, figsize=(8, 20), dpi=100)
        ax[0].plot(energies, eels_normalize(intensities), 
                   color='r', label='Data')
        for (low, high) in fit_ranges:
            low_index = eels_get_nearest_index(energies, low)
            high_index = eels_get_nearest_index(energies, high)
            ax[0].plot(energies[low_index:high_index],eels_normalize(intensities)[low_index:high_index], marker='o', ms=5, 
                       markerfacecolor='none', markeredgecolor='k', lw=0, alpha=0.5, label='Fit Region')
        ax[0].legend(frameon=False, fontsize=10)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Energy Loss (eV)')
        ax[0].set_ylabel('Normalized Intensity (au)')

        ax[1].plot(energies, eels_normalize(intensities), 
                   color='r',label='Data')
        background_norm = (ret_background - np.amin(intensities))/np.ptp(intensities)
        ax[1].plot(ret_energies, background_norm, color='b', lw=3, label='Power Law Fit')
        ax[1].legend(frameon=False,fontsize=10)
        ax[1].set_yscale('log')   
        ax[1].set_xlabel('Energy Loss (eV)')
        ax[1].set_ylabel('Normalized Intensity (au)')

        ax[2].plot(ret_energies, ret_intensities, color='b',label='Bkg. Subtracted Data')
        ax[2].axhline(0,color='k',lw=0.5,ls='--')
        ax[2].set_xlabel('Energy Loss (eV)')
        ax[2].set_ylabel('Intensity (counts)') 

        if plot_range is not None:
            ax[0].set_xlim(plot_range[0], plot_range[1])
            ax[1].set_xlim(plot_range[0], plot_range[1])   
            ax[2].set_xlim(max(plot_range[0], first_low), plot_range[1])

            autoscale_ylim(ax[0], energies, eels_normalize(intensities), padding = 0.01)
            autoscale_ylim(ax[1], energies, eels_normalize(intensities), padding = 0.01)
            autoscale_ylim(ax[2], ret_energies, ret_intensities, padding = 0.05)
         
    return ret_energies, ret_intensities, ret_background

# EELS mapping methods
