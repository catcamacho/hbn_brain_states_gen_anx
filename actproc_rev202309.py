#! /bin/python3

'''
This script processes cifti dense timeseries data for using in parcel or vertex-wise activation
analyses.

INPUTS:
    ts : cifti dense timeseries data (e.g., "task-movieTP_bold1_AP_Atlas.dtseries.nii")
    motderivs : motion parameter first order derivatives (e.g., "Movement_Regressors_dt.txt")

OUTPUTS:
    procts : processed dense timeseries cifti
    parcelts : processed data by parcel of a specified atlas
'''

import os
import nibabel as nib
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from subprocess import check_call
from glob import glob
import sys

ts = sys.argv[1]
motderivs = sys.argv[2]

working_dir = os.path.dirname(ts)
os.chdir(working_dir)
prefix = os.path.basename(working_dir)

# study-specific variables
atlas_dlabel = '/Users/catcamacho/Library/CloudStorage/Box-Box/CCP/HBN_study/proc/null_lL_WG33/Gordon333_SeitzmanSubcortical.32k_fs_LR.dlabel.nii'
TR = nib.load(ts).header.get_axis(0).step # in seconds
threshold = 0.9 # FD threshold
lowpass = 0.1 # in Hz
highpass = 0.008 # in Hz
fs = 1/TR
notch_low = 18/60 # breaths per second, lower range
notch_high = 30/60 # breaths per second, higher range

# resample data if needed before processing
if '32k' not in ts:
    check_call(['wb_command', '-cifti-resample',ts,'COLUMN', atlas_dlabel, 'COLUMN', 'ADAP_BARY_AREA', 'CUBIC',
                ts.replace('.dtseries','.32k_fs_LR.dtseries')])
    ts = ts.replace('.dtseries','.32k_fs_LR.dtseries')
# functions
def demean_detrend(ts):
    img = nib.load(ts)
    data = img.get_fdata()
    data_dd = signal.detrend(data,axis=0,type = 'linear')
    dd_ts = ts.replace('.32k_fs_LR','_demean_detrend.32k_fs_LR')
    img_dd = nib.cifti2.cifti2.Cifti2Image(data_dd,(img.header.get_axis(0),img.header.get_axis(1)))
    img_dd.to_filename(dd_ts)
    return(dd_ts)

def make_noise_ts(dd_ts, notch_low, notch_high, fs, motderivs, threshold=threshold):
    data_dd = nib.load(dd_ts).get_fdata()
    globalsig = np.mean(data_dd, axis=1)
    globalsig = np.expand_dims(globalsig, axis=1)

    # motderivs have 12 columns, first 6 is x,y,z,pitch,yaw,roll and the next 6 is their derivatives
    # FD = sum(absolute(derivatives)) (Power et al. 2012 NeuroImage)
    # load movement metrics, apply respiratory notch filter, & compute filtered FD (Fair et al. 2020 NeuroImage)
    # The Petersen lab used to use R and R' but now they use the 24-parameter friston regressors which is [R,    R^2,R(t-1), R(t-1)^2] Power et al. 2014
    notchb, notcha = signal.butter(2, [notch_low, notch_high], 'bandstop', fs=fs)
    dR = np.loadtxt(motderivs)[:,6:] # motion derivatives
    dR[:,3:] = 50*(np.pi/180)*dR[:,3:] # convert degrees to mm, 50 is the assumed head radius
    dR = signal.filtfilt(notchb, notcha, dR, axis=0) # filtering for respiration

    FD = np.sum(np.absolute(dR),axis=1)
    FD_file = ts.replace('Atlas.32k_fs_LR.dtseries.nii','FD.txt')
    np.savetxt(FD_file, FD) # saving the FD vector for future use

    # 24 params friston regressors
    R = np.loadtxt(motderivs)[:,:6]
    R_lag1 = np.roll(R,shift=1, axis=0)
    R_lag1[1,:] = 0;
    nuisance = np.hstack((R,R**2,R_lag1,R_lag1**2,globalsig))

	# create timeseries of volumes to censor
    vols_to_censor = FD>threshold
    n_vols = np.sum(vols_to_censor)
    if n_vols > 0:
        spikes = np.zeros((len(FD),n_vols))
        b = 0
        for a in range(0,len(FD)):
            if vols_to_censor[a]==1:
                spikes[a,b] = 1
                b = b + 1
        nuisance = np.hstack((nuisance, spikes))

    denoise_mat = os.path.abspath(motderivs.replace('Movement_Regressors_dt','nuissance_thresh{0}'.format(threshold)))
    np.savetxt(denoise_mat, nuisance)
    return(denoise_mat, FD_file)


def nuisance_regression(dd_ts, denoise_mat, threshold=threshold):
    # same as Kardan 2022, use a relatively relaxed threshold to remove the frames in calculating nuissance     coefficients,
    # also see Fair et al. 2020 NeuroImage " Nuisance regressor beta weights were calculated solely on the basis of frames with low
    #movement but regression was applied to all frames in order to preserve the temporal structure of the data prior to filtering in the time domain."
    data_dd = nib.load(dd_ts).get_fdata()
    nuisance = np.loadtxt(denoise_mat)
    inv_mat = np.linalg.pinv(nuisance)
    coefficients = np.dot(inv_mat,data_dd)
    yhat = np.dot(nuisance,coefficients)
    resid_data = data_dd-yhat

    weights = dd_ts.replace('.32k_fs_LR','_denoisecoeff{0}.32k_fs_LR'.format(threshold))
    denoised_ts = dd_ts.replace('.32k_fs_LR','_resid{0}.32k_fs_LR'.format(threshold))

    # make cifti header to save residuals and coefficients
    ax1 = nib.load(dd_ts).header.get_axis(0)
    ax2 = nib.load(dd_ts).header.get_axis(1)
    header = (ax1,ax2)
    # save outputs
    resid_image = nib.cifti2.cifti2.Cifti2Image(resid_data, header)
    resid_image.to_filename(denoised_ts)

    ax1.size = nuisance.shape[1]
    header = (ax1, ax2)
    coeff_image = nib.cifti2.cifti2.Cifti2Image(coefficients, header)
    #coeff_image.to_filename(weights)
    return(denoised_ts)


# run the script
dd_ts = demean_detrend(ts)
print('  + timeseries rescaled.')
denoise_mat, FD_file = make_noise_ts(dd_ts, notch_low, notch_high, fs, motderivs)
print('  + nuissance regressors made.')
denoised_ts = nuisance_regression(dd_ts, denoise_mat)
print('  + data are denoised.')


# apply gordon parcels to the timeseries
check_call(['wb_command','-cifti-parcellate', denoised_ts, atlas_dlabel,
            'COLUMN', denoised_ts.replace('.32k_fs_LR.dtseries','_gordon.32k_fs_LR.ptseries'), '-only-numeric'])
print('  + data parcellated.')

