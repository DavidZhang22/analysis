import numpy as np
import matplotlib.pyplot as plt, matplotlib.mlab as mlab
import csaps
import pyspeckit as p

from astropy import log
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, Bounds

from astropy.io import fits
from astropy.modeling.models import Voigt1D
from scipy.signal import medfilt
from scipy import optimize
from scipy import stats
from astropy import modeling


import pandas as pd 

if 1 :
    import warnings
    warnings.filterwarnings("ignore")

File = []

File.append( r"C:\Users\David Zhang\Desktop\STTP\MJD_57734\EXPERT3.2016-12-12T09%3A06%3A46_STAR83image_slice.fits")
File.append(r"C:\Users\David Zhang\Desktop\STTP\MJD_57734\EXPERT3.2016-12-12T09%3A40%3A48_STAR83image_slice.fits")
File.append(r"C:\Users\David Zhang\Desktop\STTP\MJD_57734\EXPERT3.2016-12-12T13%3A03%3A32_STAR83image_slice.fits")
#File.append(r"C:\Users\David Zhang\Desktop\STTP\MJD_57734\EXPERT3.2016-12-12T13%3A03%3A32_STAR83image_slice.fits")



def n2s(num, sd):

    return str(np.around(num, sd))

def rms(arr_flux, wl, lower, upper, line):
    a = 1;
    b = 0;
    if line == 1:
        a = 1.2
    elif line == 2:
        a = 2.2
    elif line == 3:
        b = .5

    c = np.logical_and(wl<lower+b, wl>lower-a)
    d = np.logical_and(wl>upper-a, wl<upper+ b)
    e = np.logical_or(c, d)

    return np.sqrt(np.mean(arr_flux[e] ** 2))-1

#time, wl, equivlent width, fwhm, fwhm error

def v_fwhm(sigma, gamma, model_type):
    
    # Full width at half max
    
    if model_type == 'voigt':
        
        # https://en.wikipedia.org/wiki/Voigt_profile
        
        # Olivero, J. J.; R. L. Longbothum (February 1977). "Empirical fits to the Voigt line width: A brief review". 
        # Journal of Quantitative Spectroscopy and Radiative Transfer. 17 (2): 233–236. 
        # Bibcode:1977JQSRT..17..233O. doi:10.1016/0022-4073(77)90161-3. ISSN 0022-4073

        f_G = 2.355 * sigma
        
        f_L = 2 * gamma
        
        f_V = 0.5346 * f_L + ( 0.2166 * f_L**2 + f_G**2 )**0.5
        
        return f_V

    else:

        return 2.355 * sigma

def eqw(wl, amp_f, avg_f, sd_f, lsd_f, cont_f, model_type):
    
    # Integral (Newton) based calculation of Eq. Width
    
    wl_hi = np.linspace(wl[0], wl[-1], len(wl) * 1000)

    dwl = wl_hi[1] - wl_hi[0]

    if model_type == 'voigt':
        f = line_model([amp_f, avg_f, sd_f, lsd_f, cont_f], wl_hi, model_type)
    else:
        f = line_model([amp_f, avg_f, sd_f, cont_f], wl_hi, model_type)
        #f = line_model([amp_f, avg_f, sd_f, 1], wl_hi, model_type)

    fsum = np.sum(cont_f - f)
    
    ew = dwl * fsum
    
    return ew



def stdFiltIt(arr_1d, weights_1d, sdms, box_wid, sudo_noise, plot_q):

    if plot_q:
        fig = plt.gcf()
        fig.clf()
        plt.title('Filter plot')
        xx = np.arange(len(arr_1d))
        plt.plot(xx, arr_1d, '.k')

    fit = arr_1d + np.nan

    for sdm in sdms:

        gi = np.logical_and(np.isfinite(weights_1d), np.isfinite(arr_1d))

        fit[gi] = medfilt(arr_1d[gi], int(box_wid))

        dy_sd = np.std(arr_1d[gi] - fit[gi]) * sdm

        #print(sdm, dy_sd, box_wid)

        gi = np.logical_and(np.abs(arr_1d - fit) <= dy_sd, gi)

        if plot_q:
            plt.plot(xx[np.logical_not(gi)], arr_1d[np.logical_not(gi)], 'xr')
            plt.plot(xx, fit + dy_sd, ':g')
            plt.plot(xx, fit - dy_sd, ':r')
            plt.grid(True)

        arr_1d[np.logical_not(gi)] = np.nan

    if plot_q:
        plt.show()

    return arr_1d, gi

def line_model(vars0, wl, model_type):
    
    # Make a spectral line using a specific profile

    if model_type == 'gauss':

        return Gaussian1D(vars0[0], vars0[1], vars0[2])(wl) + vars0[3]

    else:
        
        f_G = 2.355 * vars0[3]
        f_L = 2 * vars0[2]
        
        return Voigt1D(vars0[1], vars0[0], f_G, f_L)(wl) + vars0[4]


def minfun2(v, wl, r, weights, model_type):
    
    return np.sum( ( line_model(v, wl, model_type) - r )**2 * weights )

def normal(File, wl_bound, i):

    hdu = fits.open(File)

    flux = hdu[0].data

    wl_start_val = 7333.316
    wl_end_val = 7427.839

    drop = flux>1

    flux = flux[drop]

    wl_change_per_pixel = (wl_end_val-wl_start_val)/4096

    pixel_index = np.arange(len(flux))

    wl = (wl_start_val + pixel_index * wl_change_per_pixel)

    line_idx_bnds = [0, 0]


    crop =  np.logical_and(wl>wl_bounds[i][0], wl<wl_bounds[i][1])

    wl_cropped = wl[crop]

    flux_cropped = flux[crop]


    box_wid = 37
    sudo_noise = 1.0
    plot_q = False

    flux_cropped2 = np.copy(flux_cropped)

    __, gi = stdFiltIt(flux_cropped2, np.ones_like(flux_cropped2), [3,2,1.6,1.4], box_wid, sudo_noise, plot_q)

    #print(gi)

    mask =  np.logical_or(wl_cropped>7389.05, wl_cropped<7388.85)


    cont_fit = csaps.UnivariateCubicSmoothingSpline(wl_cropped[np.logical_and(gi, mask)], flux_cropped[np.logical_and(gi, mask)], smooth=0.991)(wl_cropped)

    flux_cropped_norm = flux_cropped / cont_fit

    if 0:
        plt.figure(figsize=(15, 10))

        plt.subplot(2,1, 1)

        plt.plot(wl_cropped, flux_cropped, '.-k')
        plt.plot(wl_cropped, cont_fit, '-r')
        plt.title('Continuum Fit')
        plt.xlabel('WL [ang]')
        plt.ylabel('Flux [adu]')

        plt.subplot(2,1, 2)

        plt.plot(wl_cropped, flux_cropped_norm, '.-k')
        plt.title('Normalized Spectrum')
        plt.xlabel('WL [ang]')
        plt.ylabel('Flux [adu]')

        plt.show()

        """plt.figure(figsize=(15,7))
        plt.plot(wl_cropped, flux_cropped_norm, '.-k')
        #plt.title('Final Normalized & Cropped Spectrum')
        plt.title('Final Spectrum')
        plt.xlabel('WL [ang]')
        plt.ylabel('Flux [adu]')
        plt.show()"""


     # import voight fitting routine
    vf = p.spectrum.models.inherited_voigtfitter.voigt_fitter()
    
    # import spectrum to routine

    flux_copy = np.copy(flux_cropped_norm)
    wl_nm = np.copy(wl_cropped)
    flux_err = np.zeros_like(flux_copy)+0.005

    #print(wl_cropped)
    sp = p.Spectrum(data=flux_copy, xarr=wl_nm, error = flux_err)#, unit=flux_unit_str)

    sp.xarr.units = 'angstrom' 
    sp.xarr.xtype = 'wavelength'
    
    # plot spectrum
        
    xmin = None
    xmax = None

    sp.plotter(xmin=xmin, xmax=xmax, ymin=0, errstyle='fill',color='grey' ) 
    
    # set continuum

    exclude_min = wl_bounds[i][0]+1.30  # Continuum fit mask lower bound
    exclude_max = wl_bounds[i][1]-1.30 # Continuum fit mask upper bound


    sp.baseline(xmin=xmin, xmax=xmax,exclude=[exclude_min,exclude_max],subtract=False,
                reset_selection=False,hightlight_fitregion=False,order=0)
    
    # Give initial guesses and let algo know how many lines to fit (Fit perfromed here too)
    #  guesses = [amplitude, wl_center, gauss_sigma, lor_sigma]

    wl_guess = (wl_bounds[i][0]+wl_bounds[i][1])*.5

    sp.specfit(guesses=[-0.15, wl_guess, .05, .05],
                limitedmax=[True,False,True,True],  # Apply upper bound limits of our guesses?
                limitedmin=[True,False,True,True],  # Apply lower bound limits of our guesses?
                limits=[(-.5,0),(wl_guess-.2, wl_guess+.2),(0,.1),(0,.1)],  # (lower, upper) bound limits of our guesses
               plot=True, fittype='voigt', color='g', 
               vheight=True, components=True)
    
 
    # Plot individual line fits
    sp.specfit.plot_components(add_baseline=True,component_yoffset=0.0)
    
    # Update plot
    sp.plotter.refresh()
    
    # Measure FWHM of each line fit

        
    ############################
    
    #amp_f, avg_f, sd_f, lsd_f, cont_f, fwhms, eqws = fit_sl(SV, LB, UB, 
    #                                              wl, flux, flux_err, 
    #                                              weights=weights, 
    #                                             constraints=constraints, 
    #                                            model_type=model_type)
    #
    #  fit_sl(sv, lb, ub, wl, flux, flux_err, weights, constraints, model_type)

    # Param 1)  Gaussian amplitude (must be negative values for absorption lines!)

    constraints = []

    amp_sv = -0.2  # Starting guess (SV)
    amp_lb = -0.4  # Lower bound (LB)
    amp_ub = -0.1  # Upper bound (UB)


    # Param 2)  Gaussian wavelength center

    avg = (wl_bound[i][0]+wl_bound[i][1])/2;
    avg_sv = avg
    avg_lb = avg-.2
    avg_ub = avg+.2


    # Param 3)  Gaussian standard deviation
    sd_sv = 0.1
    sd_lb = 0.0
    sd_ub = 1.0


    # Param 4)  Lorentzian standard deviation
    lsd_sv = 0.1
    lsd_lb = 0.0
    lsd_ub = 1.0


    # Param 5)  Continuum level
    cont_sv = 1.0
    cont_lb = 0.999
    cont_ub = 1.001

    sv = [amp_sv, avg_sv, sd_sv, lsd_sv, cont_sv]
    lb = [amp_lb, avg_lb, sd_lb, lsd_lb, cont_lb]
    ub = [amp_ub, avg_ub, sd_ub, lsd_ub, cont_ub]

    weights = np.ones(len(wl_cropped), dtype=float)

    p3 = minimize(minfun2, 
                  sv, 
                  method='SLSQP', #'L-BFGS-B', 
                  bounds=Bounds(lb, ub), 
                  constraints=constraints, 
                  args=(wl_cropped, flux_cropped, weights, "voigt"), 
                  options={'maxiter': 1000, 
                           'ftol': 9e-3, 
                           'iprint': 1, 
                           'disp': False, 
                           'eps': 1e-9}
                 )
    
    print('Solver message = ', p3.message)
    print('Solver success = ', p3.success)
    print('Solver iterations = ', p3.nfev)
    print("WL center = ", avg)

    v = np.copy(p3.x)
    sd_f = v[2]
    lsd_f = v[3]
    print(sd_f,lsd_f)
    fwhms = v_fwhm(sd_f,lsd_f, "voigt")


    # Update plot
    #sp.plotter.refresh()

    # Measure equivalant width of each line fit
    A = sp.specfit.EQW(plot=True, plotcolor='g', fitted=False, 
                   components=True, 
                   annotate=True, loc='lower left', xmin=None, xmax=None) # continuum=1.0, 
    # Update plot

    #sp.plotter.refresh()

    #plt.show();

    # Set plot axes labels
    xarr_fit_units = 'angstrom'
    plt.ylabel('Normalized Flux')
    plt.xlabel('Wavelength [ang])')

    return A[0], (rms(flux_copy, wl_cropped, wl_bounds[i][0],wl_bounds[i][1], i )), fwhms#, A[1]

    
wl_bounds= [[0, 80000],
           [7367.7, 7373.5],
           [7391, 7395],
           [7394.8, 7399.8]]


#print(normal(File[2], wl_bounds, 3))


if 1:
    w, h = 3, 3;
    eqw = [[0 for x in range(w)] for y in range(h)] 
    error = [[0 for x in range(w)] for y in range(h)] 
    fwhm = [[0 for x in range(w)] for y in range(h)]

    for f in range(len(File)):
        for i in range(3):
            eqw[f][i], error[f][i], fwhm[f][i] = normal(File[f], wl_bounds, i+1)

            eqw[f][i] = round(eqw[f][i], 5)

            error[f][i] = round(error[f][i], 5)

            fwhm[f][i]*=2.355

    for f in range(len(File)):
        for i in range(3):
            print(eqw[f][i], " ", error[f][i], " ", fwhm[f][i])

        print("\n")

    pd.DataFrame(eqw).to_csv(r"C:\Users\David Zhang\Desktop\STTP\Data\eqw.csv")
    pd.DataFrame(error).to_csv(r"C:\Users\David Zhang\Desktop\STTP\Data\error.csv")
    pd.DataFrame(fwhm).to_csv(r"C:\Users\David Zhang\Desktop\STTP\Data\fwhm.csv")
#print('Eq. width = ', normal(File[2], wl_bounds, 3))


#for i in range(len(File)):
#   print('Eq. width = ', normal(File[i], wl_bounds, i+1))


