# This notebook shows how to fit a single spectral line with a Voigt profile
# The Full width at half max (FWHM), Equivalent Width, and fit error are 
# calculated as well.
# By Kevin Willis 28/1/2020

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import Voigt1D, Gaussian1D
from scipy.interpolate import UnivariateSpline
import csaps
from scipy.signal import medfilt

import pandas as pd 

from os import listdir
from os.path import isfile, join

# Set default plot figure size
plt.rcParams['figure.figsize'] = (15.0, 8.0)



def n2s(num, sd): 

    return str(np.around(num, sd))


def wlr2ir(wl, wl_bounds):
    
    line_idx_bnds = [0, 0]

    line_idx_bnds[0] = np.argwhere(np.abs(wl - wl_bounds[0]) == np.nanmin(np.abs(wl - wl_bounds[0])))
    line_idx_bnds[1] = np.argwhere(np.abs(wl - wl_bounds[1]) == np.nanmin(np.abs(wl - wl_bounds[1])))

    # What indicies span within the wavelength bounds?
    return np.arange(line_idx_bnds[0], line_idx_bnds[1] + 1)


def rms(y):

    gi = np.isfinite(y)

    y = y[gi]

    return np.sqrt(np.mean(y ** 2))

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

def fwhm(sigma, gamma, model_type):
    
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

''' 
def eqw(amp, sigma):
    
    # Eq. width
    
    return np.abs(amp * sigma * np.sqrt (2 * np.pi))
''';

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


def fit_sl(sv, lb, ub, wl, flux, flux_err, weights, constraints, model_type):
    
    # Fit a single spectral line with a Gaussian or Voigt profile
    
    #-----------------------------------------------------------------------------------
    
    # INPUTS
    # * SV          = All inital guesses (a.k.a. starting values) of the params to solve
    # * LB          = All lower bounds of the params to solve
    # * UB          = All upper bounds of the params to solve
    # * wl          = Wavelength array
    # * flux        = Flux array
    # * flux_err    = Flux error array
    # * weights     = Weights for each wavelength/flux value
    # * constraints = Constraints the solver needs to obey
    # * model_type  = Gaussian ('gauss') or Voigt ('voigt') fit?

    # OUTPUTS
    # * amp_f   = Gaussian amplitude
    # * avg_f   = Gaussian wavelength center
    # * sd_f    = Gaussian standard deviation
    # * lsd_f   = Lorentzian standard deviation
    # * cont_f  = Continuum level
    # * fwhms   = Full width at half max (FWHM)
    # * eqws    = Equivalent Width
    
    #-----------------------------------------------------------------------------------
    
    # Error check 1
    if model_type == 'gauss' and len(sv) != 4:
        
        print('ERROR: Something is wrong with your starting values or boundary condition arrays (SV, LB, UB)')
        print('When model type is "gauss", each of those arrays should be of length 4')
        print('However the arrays (SV, LB, UB) have lengths = ', len(sv), len(lb), len(ub))
        return None
    
    elif model_type == 'voigt' and len(sv) != 5:
        
        print('ERROR: Something is wrong with your starting values or boundary condition arrays (SV, LB, UB)')
        print('When model type is "voigt", each of those arrays should be of length 5')
        print('However the arrays (SV, LB, UB) have lengths = ', len(sv), len(lb), len(ub))
        return None
        
    
    from scipy.optimize import minimize, Bounds
    
    plt.rcParams['figure.figsize'] = (15.0, 8.0)
    
    ############################
    
    p3 = minimize(minfun2, 
                  sv, 
                  method='SLSQP', #'L-BFGS-B', 
                  bounds=Bounds(lb, ub), 
                  constraints=constraints, 
                  args=(wl, flux, weights, model_type), 
                  options={'maxiter': 1000, 
                           'ftol': 1e-8, 
                           'iprint': 1, 
                           'disp': False, 
                           'eps': 1.4901161193847656e-8}
                 )
    
    print('Solver message = ', p3.message)
    print('Solver success = ', p3.success)
    print('Solver iterations = ', p3.nfev)
    
    v = np.copy(p3.x)
    
    
    
    ########################################################################################
    
    amp_f = v[0]
    avg_f = v[1]
    sd_f = v[2]
    
    if model_type == 'voigt':
        lsd_f = v[3]
        cont_f = v[4]
    else:
        lsd_f = None
        cont_f = v[3]
        
        
    eqws = eqw(wl, amp_f, avg_f, sd_f, lsd_f, cont_f, model_type)
    fwhms = fwhm(sd_f, lsd_f, model_type)
    
    
    # Print a summaray of the solved params
    if 0 :
        print('\n*****************************************')
        print('               Solved Params')
        print('-----------------------------------------\n')
        print('Gaussian amplitude   = ' + n2s(amp_f, 4))
        print('Gaussian WL center   = ' + n2s(avg_f, 4))
        print('Gaussian SD          = ' + n2s(sd_f, 4))
        
        if model_type == 'voigt':
            
            print('Lorentzian SD        = ' + n2s(lsd_f, 4))
        
        print('Continuum flux level = ' + n2s(cont_f, 4))
        
        print('\n-------------------------------\n')
        
        print('Full width at half max (FWHM) = ' + n2s(fwhms, 4))
        print('Equivalent Width              = ' + n2s(eqws, 4))
        
        print('\n*****************************************\n')
        
        
        plot_fit2(v, wl, flux, flux_err, weights, fwhms, eqws, amp_f, avg_f, sd_f, cont_f)
    
    
    if model_type == 'voigt':
        return amp_f, avg_f, sd_f, lsd_f, cont_f, fwhms, eqws
    else:
        return amp_f, avg_f, sd_f, cont_f, fwhms, eqws
    

def minfun2(v, wl, r, weights, model_type):
    
    return np.sum( ( line_model(v, wl, model_type) - r )**2 * weights )

    
def plot_fit2(v, wl, flux, flux_err, weights, fwhms, eqws, amp_f, avg_f, sd_f, cont_f):
    
    # Plot the spectral line fit
    
    mn = np.arange(len(wl), dtype=int)
    
    #plt.figure()
    #plt.plot(wl, flux, '-k')
    #plt.plot(wl, line_model(v, wl, model_type), '-m')
    
        
    ########################################################################################

    plt.figure(figsize=(15, 5))
    #plt.subplot()
    #plt.plot(wl, flux, '-k')
    plt.fill_between(wl[mn], flux[mn] - flux_err[mn], flux[mn] + flux_err[mn], 
             step='mid', 
             color=[0.85, 0.85, 0.85], 
             label='Flux Error')
    plt.step(wl[mn], flux[mn], '-', c=[0.5, 0.5, 0.5], where='mid', linewidth=1, label='Flux')

    wl_hi = np.linspace(wl[mn][0], wl[mn][-1], 1000)

    plt.plot(wl_hi, line_model(v, wl_hi, model_type), '-k', label='Fit')
    
    # overlay eqw
    plt.fill_between([avg_f - eqws/2, avg_f + eqws/2], [0, 0], [1, 1], 
         step='mid', 
         color=[0.0, 1.0, 0.0, 0.01])
    plt.arrow(avg_f, cont_f, eqws/2, 0.0, length_includes_head=1, color=[0,0,0], head_width=0.02, head_length=0.09)
    plt.arrow(avg_f, cont_f, -eqws/2, 0.0, length_includes_head=1, color=[0,0,0], head_width=0.02, head_length=0.09)
    plt.text(avg_f, cont_f + 0.02, 'W = ' + n2s(eqws, 3), horizontalalignment='center')

    # overlay fwhm
    if model_type != 'voigt':
        
        hm = cont_f + amp_f / 2
        #plt.fill_between([avg_f - fwhms/2, avg_f + fwhms/2], [0, 0], [hm, hm], 
        #         step='mid', 
        #         color=[0.0, 0.0, 1.0, 0.02])
        plt.arrow(avg_f, hm, fwhms/2, 0.0, length_includes_head=1, color=[0,0,0], head_width=0.02, head_length=0.09)
        plt.arrow(avg_f, hm, -fwhms/2, 0.0, length_includes_head=1, color=[0,0,0], head_width=0.02, head_length=0.09)
        plt.text(avg_f, hm + 0.01, 'FWHM', horizontalalignment='center')
        plt.text(avg_f, hm - 0.055, n2s(fwhms, 3), horizontalalignment='center')
    

    plt.xlim(wl[mn][0], wl[mn][-1])
    plt.ylim(bottom=0.0)
    plt.xlabel('WL')
    plt.ylabel('Flux (ADU)')
    plt.legend()

    ########################################################################################
    ## Error plot

    derr = flux[mn] - line_model(v, wl, model_type)
    
    plt.figure(figsize=(15, 3))
    plt.title('Fit Error | RMS = ' + n2s(rms(derr), 3))
    plt.fill_between(wl[mn], -flux_err[mn], flux_err[mn], 
             step='mid', 
             color=[0.85, 0.85, 0.85], 
             label='Flux Error')
    #plt.plot(wl[mn], derr, '-', c=[0.2, 0, 0, 1])
    plt.plot(wl[mn], derr, '.-k', label='Fit Error')
    plt.xlim(wl[mn][0], wl[mn][-1])
    plt.xlabel('WL')
    plt.ylabel('Fit Error (ADU)')
    plt.legend()
        
        
def line_model(vars0, wl, model_type):
    
    # Make a spectral line using a specific profile

    if model_type == 'gauss':

        return Gaussian1D(vars0[0], vars0[1], vars0[2])(wl) + vars0[3]

    else:
        
        f_G = 2.355 * vars0[3]
        f_L = 2 * vars0[2]
        
        return Voigt1D(vars0[1], vars0[0], f_G, f_L)(wl) + vars0[4]
    
    
def measure_error(wl, flux, fwhms, 
                  amp_f, avg_f, sd_f, cont_f, 
                  left_lb, left_ub, 
                  right_lb, right_ub):

    # Measure the error of a spectral line fit
    
    #-----------------------------------------------------------------------------------
    
    # INPUTS
    # * wl        = Wavelength array
    # * flux      = Flux array
    # * fwhms     = Solved Full width at half max (FWHM)
    # * amp_f     = Solved Gaussian amplitude
    # * avg_f     = Solved Gaussian wavelength center
    # * sd_f      = Solved Gaussian standard deviation
    # * cont_f    = Solved Continuum level
    # * left_lb   = Lower bound wavelength on left side of line
    # * left_ub   = Upper bound wavelength on left side of line
    # * right_lb  = Lower bound wavelength on right side of line
    # * right_ub  = Upper bound wavelength on right side of line

    # OUTPUTS
    # * eqw_err   = Error of Equivalent Width measurement
    # * noise_rms = Measured noise
    
    #-----------------------------------------------------------------------------------
    
    
    #if np.any(avg_f + sd_f * 3 > right_ub) or np.any(avg_f - sd_f * 3 < left_lb):
        #print('WARNING: You are calculating noise inside the bounds of the spectral line.')
    
    
    # If some of the boundary conditions are too close to the spectral 
    # line, they will automaticaly be changed
    if np.any(avg_f + sd_f * 3 > right_lb):
        
        right_lb = avg_f + sd_f * 3.1
    
    if np.any(avg_f - sd_f * 3 < left_ub):
        
        left_ub = avg_f - sd_f * 3.1
    
    
    # Get indicies of the wavelength bounds that were supplied
    gi_left = wlr2ir(wl, [left_lb, left_ub])
    gi_right = wlr2ir(wl, [right_lb, right_ub])
    gi_all = np.ravel(np.concatenate((gi_left, gi_right)))
    
    
    # Calculate Noise
    noise_rms = rms(flux[gi_all] - cont_f)

    
    # Calculate error of measurement
    eqw_err = fwhms * noise_rms
    
    
    # Plot measured noise range
    plt.figure(figsize=(15, 4))
    plt.plot(wl, cont_f + np.zeros_like(wl), '--g', label='Continuum')
    plt.plot(wl, flux, '.-k', label='Normalized spectrum')
    plt.plot(wl[gi_left], flux[gi_left], '.-r', label='Noise measure ranges')
    plt.plot(wl[gi_right], flux[gi_right], '.-r')#, label='Right noise measure range')
    
    plt.plot([avg_f, avg_f], [cont_f + amp_f, cont_f], '-b')
    plt.plot(np.array([avg_f, avg_f]) + sd_f * 3, [cont_f + amp_f, cont_f], ':r')
    plt.plot(np.array([avg_f, avg_f]) - sd_f * 3, [cont_f + amp_f, cont_f], ':r', label='Spectral line 3 sigma bounds')
    
    plt.title('Noise Measurement Plot  |  Noise = ' + n2s(noise_rms, 4) + '  |  Equivalent Width error = ' + n2s(eqw_err, 3))
    plt.xlabel('WL')
    plt.ylabel('Flux')
    plt.legend()
    
    return eqw_err, noise_rms

def solve(File, filenum, linenum):
        ################################################################
    ################################################################
                    # HOW TO FIT A VOIGT PROFILE
    ################################################################
    ################################################################

    # First we need to create a synthetic spectral line for this demonstration

    # Set the parameters for your synthetic spectral line below


    #----------------------------------------------------------------

    # Specify we want to fit a Gaussian

    from astropy.io import fits


    #Data

    wl_bounds= [[0, 80000],
               [7368.6, 7377.5],
               [7386.5, 7396.05],
               [7396.8, 7403]]

    hdu = fits.open(File[filenum])

    flux = hdu[0].data

    wl_start_val = 7333.316
    wl_end_val = 7427.839

    drop = flux>1

    pixel_index = np.arange(len(flux))

    flux = flux[drop]

    wl_change_per_pixel = (wl_end_val-wl_start_val)/4096


    wl = (wl_start_val + pixel_index * wl_change_per_pixel)

    wl = wl[drop]

    line_idx_bnds = [0, 0]


    crop =  np.logical_and(wl>wl_bounds[linenum][0], wl<wl_bounds[linenum][1])

    wl_cropped = wl[crop]

    flux_cropped = flux[crop]


    # Spectral line

    flux_err = np.zeros(len(wl_cropped))



    flux = hdu[0].data  

    # Plot the synthetic spectral line
    if 0:
        plt.figure()
        plt.plot(wl_cropped, flux_cropped, '.-g', label='With noise')
        plt.title('Synthetic Spectral Line')
        plt.xlabel('WL')
        plt.ylabel('Flux')
        plt.legend()
    #########################################################

    box_wid = 41
    sudo_noise = 1.0
    plot_q = False

    flux_cropped2 = np.copy(flux_cropped)

    __, gi = stdFiltIt(flux_cropped2, np.ones_like(flux_cropped2), [3,2,1.6,1.4], box_wid, sudo_noise, plot_q)

    #print(gi)

    mask =  np.logical_or(wl_cropped>7389.05, wl_cropped<7388.85)


    cont_fit = csaps.UnivariateCubicSmoothingSpline(wl_cropped[np.logical_and(gi, mask)], flux_cropped[np.logical_and(gi, mask)], smooth=0.980)(wl_cropped)

    flux_cropped_norm = flux_cropped / cont_fit


    #smoothing needs to be fixed!!!
    if 0:
        plt.figure()
        plt.plot(wl_cropped, flux_cropped, '.-g', label='With noise')
        plt.plot(wl_cropped, cont_fit, '.-g', label='With noise')
        plt.title('Synthetic Spectral Line')
        plt.xlabel('WL')
        plt.ylabel('Flux')
        plt.legend()
        plt.show()

    #########################################################

    # Set parameter initial guesses and boundary conditions

    #########################################################
    #########################################################

    # You will need to follow these concepts in your own code
    #########################################################


    # Specify we want to fit a Voigt profile
    model_type = 'voigt'


    #########################################################

    # The following parameters are solved to completely describe our 
    # spectral line. We must set valid initial guesses (SV) and 
    # boundary conditions for each parameter. Initial gusses must
    # be inside the range of your lower and upper bounds (LB, UB) such that
    # the following is valid:  LB < SV < UB


    # Param 1)  Gaussian amplitude (must be negative values for absorption lines!)
    amp_sv = -0.2  # Starting guess (SV)
    amp_lb = -0.4  # Lower bound (LB)
    amp_ub = -0.1  # Upper bound (UB)


    # Param 2)  Gaussian wavelength center

    avg = (wl_bounds[linenum][0]+wl_bounds[linenum][1])/2;
    avg_sv = avg
    avg_lb = avg-.2
    avg_ub = avg+.2


    # Param 3)  Gaussian standard deviation
    sd_sv = 0.1
    sd_lb = 0.01
    sd_ub = 0.5


    # Param 4)  Lorentzian standard deviation
    lsd_sv = 0.1
    lsd_lb = 0.01
    lsd_ub = 0.5


    # Param 5)  Continuum level
    cont_sv = 1.0
    cont_lb = 0.999
    cont_ub = 1.001


    # Each wavelength can be weighted so that the solver will
    # focus on fitting certain areas better than others. In this
    # example, we want all flux values to be treated with equal
    # importance in the fit. Therefore, all weights will be 1.
    #
    # If we wanted to perform weigthing on a specifc WL range, we
    # could use the following code. In this example, we want the
    # solver to ignore flux in the WL range [5001.8, 5003.6].
    # To do this we would set the weights to zero in this range:
    #
    #>   weights = np.ones(len(wl), dtype=float)
    #>   weights[ wlr2ir( wl, [5001.8, 5003.6] ) ] = 0.0

    weights = np.ones(len(wl_cropped), dtype=float)

    for i in range(len(weights)):
        if np.logical_and(wl_cropped[i]>avg-.1,wl_cropped[i]<avg+.1):
            weights[i] = weights[i]+1
        
    #weight center higher

    # It is possible to add constaints on the solver. 
    # We dont need any for this simple case and you probably dont either.
    constraints = []


    # Put the parameters into a form that the solver can use
    SV = [amp_sv, avg_sv, sd_sv, lsd_sv, cont_sv]
    LB = [amp_lb, avg_lb, sd_lb, lsd_lb, cont_lb]
    UB = [amp_ub, avg_ub, sd_ub, lsd_ub, cont_ub]
    # Run the solver!
    # This will give plot and text outputs showing the results of the fit.
    # We are not done! We need to also determine the error of 
    # our measurements. We will do that next.

    # INPUTS
    # * SV          = All inital guesses (a.k.a. starting values) of the params to solve
    # * LB          = All lower bounds of the params to solve
    # * UB          = All upper bounds of the params to solve
    # * wl          = Wavelength array
    # * flux        = Flux array
    # * flux_err    = Flux error array
    # * weights     = Weights for each wavelength/flux value
    # * constraints = Constraints the solver needs to obey
    # * model_type  = Gaussian ('gauss') or Voigt ('voigt') fit?

    # OUTPUTS
    # * amp_f   = Gaussian amplitude
    # * avg_f   = Gaussian wavelength center
    # * sd_f    = Gaussian standard deviation
    # * lsd_f   = Lorentzian standard deviation
    # * cont_f  = Continuum level
    # * fwhms   = Full width at half max (FWHM)
    # * eqws    = Equivalent Width

    amp_f, avg_f, sd_f, lsd_f, cont_f, fwhms, eqws = fit_sl(SV, LB, UB, 
                                                       wl_cropped, flux_cropped_norm, flux_err, 
                                                       weights=weights, 
                                                       constraints=constraints, 
                                                       model_type=model_type)
                                                       # Measure the error of your Equivalent Width measurement

    left_lb = wl_bounds[linenum][0]+.1
    left_ub = (wl_bounds[linenum][0]+avg*3)/4
    right_lb = (wl_bounds[linenum][1]+3*avg)/4
    right_ub = wl_bounds[linenum][1]-.1


    # INPUTS
    # * wl        = Wavelength array
    # * flux      = Flux array
    # * fwhms     = Solved Full width at half max (FWHM)
    # * amp_f     = Solved Gaussian amplitude
    # * avg_f     = Solved Gaussian wavelength center
    # * sd_f      = Solved Gaussian standard deviation
    # * cont_f    = Solved Continuum level
    # * left_lb   = Lower bound wavelength on left side of line
    # * left_ub   = Upper bound wavelength on left side of line
    # * right_lb  = Lower bound wavelength on right side of line
    # * right_ub  = Upper bound wavelength on right side of line

    # OUTPUTS
    # * eqw_err   = Error of Equivalent Width measurement
    # * noise_rms = Measured noise


    eqw_err, noise_rms = measure_error(wl_cropped, flux_cropped_norm, fwhms, 
                                       amp_f, avg_f, sd_f, cont_f, 
                                       left_lb, left_ub, 
                                       right_lb, right_ub)


    print('Error of Equivalent Width measurement = ' + n2s(eqw_err, 3)) 
    # Lets see how well we recovered the actual spectral line

    FV = [amp_f, avg_f, sd_f, lsd_f, cont_f]
    wl_hi = np.linspace(wl_cropped[0], wl_cropped[-1], len(wl_cropped) * 100)

    plt.figure()
    plt.plot(wl_cropped, flux_cropped_norm, '.-k', label='With noise')

    gauss_amp = amp_f
    gauss_mean = avg_f
    gauss_sd = sd_f
    lorz_sd = lsd_f
    continuum_level = 1;

    if 0:
        plt.fill_between([avg_f - eqws/2, avg_f + eqws/2], [0, 0], [1, 1], 
                 step='mid', 
                 color=[0.0, 1.0, 0.0, 0.04])
        plt.arrow(avg_f, cont_f, eqws/2, 0.0, length_includes_head=1, color=[0,0,0], head_width=0.02, head_length=0.09)
        plt.arrow(avg_f, cont_f, -eqws/2, 0.0, length_includes_head=1, color=[0,0,0], head_width=0.02, head_length=0.09)
        plt.text(avg_f, cont_f + 0.02, 'W = ' + n2s(eqws, 3), horizontalalignment='center')

        if model_type != 'voigt':
            hm = cont_f + amp_f / 2
            #plt.fill_between([avg_f - fwhms/2, avg_f + fwhms/2], [0, 0], [hm, hm], 
            #         step='mid', 
            #         color=[0.0, 0.0, 1.0, 0.02])
            plt.arrow(avg_f, hm, fwhms/2, 0.0, length_includes_head=1, color=[0,0,0], head_width=0.02, head_length=0.09)
            plt.arrow(avg_f, hm, -fwhms/2, 0.0, length_includes_head=1, color=[0,0,0], head_width=0.02, head_length=0.09)
            plt.text(avg_f, hm + 0.01, 'FWHM', horizontalalignment='center')
            plt.text(avg_f, hm - 0.035, n2s(fwhms, 3), horizontalalignment='center')

    if 0:

        plt.plot(wl_hi, line_model(FV, wl_hi, model_type), '-r', label='Fit')
        plt.plot(wl_hi, line_model([gauss_amp, gauss_mean, gauss_sd, lorz_sd, continuum_level], 
                                   wl_hi, model_type), ':c', label='Actual Noiseless Line')

        plt.title('Synthetic Spectral Line')
        plt.xlabel('WL')
        plt.ylabel('Flux')
        plt.legend()



        # Plot error between solved fit and actual solution

        AV = [gauss_amp, gauss_mean, gauss_sd, lorz_sd, continuum_level]

        derr = line_model(AV, wl_hi, model_type) - line_model(FV, wl_hi, model_type)

        plt.figure(figsize=(15, 3))
        plt.title('(Actual - Fit) Error | RMS = ' + n2s(rms(derr), 3))
        #plt.plot(wl[mn], derr, '-', c=[0.2, 0, 0, 1])
        plt.plot(wl_hi, derr, '-k', label='Fit Error')
        plt.xlim(wl_cropped[0], wl_cropped[-1])
        plt.xlabel('WL')
        plt.ylabel('Fit Error (ADU)')
        plt.grid()
        plt.legend()
        #plt.show()

    return n2s(eqws, 5), n2s(fwhms, 5), n2s(eqw_err, 5)


model_type = 'voigt' 

File = [join(r"C:\Users\David Zhang\Desktop\STTP\MJD_57734", f) for f in listdir(r"C:\Users\David Zhang\Desktop\STTP\MJD_57734") if isfile(join(r"C:\Users\David Zhang\Desktop\STTP\MJD_57734", f))]


#solve(File, 2, 2)

if 1:
    w, h = 3, len(File);
    eqwP = [[0 for x in range(w)] for y in range(h)] 
    errorP = [[0 for x in range(w)] for y in range(h)] 
    fwhmP = [[0 for x in range(w)] for y in range(h)]

    for f in range(len(File)):
        for i in range(3):
            print(f)
            eqwP[f][i], fwhmP[f][i], errorP[f][i] = solve(File, f, i+1)

    for f in range(len(File)):
        s = File[f].find(".")
        print("Time:", File[f][s: s+10], File[f][s+12: s+14] ,"-",File[f][s+17: s+19])
        for i in range(3):
            print("L", i+1, ":")
            print("EQW:", eqwP[f][i],  " FWHM:", fwhmP[f][i]," RMS Error:", errorP[f][i])

        print("\n")

    pd.DataFrame(eqwP).to_csv(r"C:\Users\David Zhang\Desktop\STTP\Data\eqw.csv")
    pd.DataFrame(errorP).to_csv(r"C:\Users\David Zhang\Desktop\STTP\Data\error.csv")
    pd.DataFrame(fwhmP).to_csv(r"C:\Users\David Zhang\Desktop\STTP\Data\fwhm.csv")