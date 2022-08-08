# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 09:45:20 2022

@author: Hello
"""

from turtle import shape
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from brainda.datasets import AlexMI
from brainda.paradigms import MotorImagery
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.decomposition import FBCSP
from brainda.algorithms.decomposition.base import generate_filterbank
import mne 
class freq_analysis():
    def __init__(self,data, meta,event,srate,latency=0,channel = 'all'):
        sub_meta=meta[meta['event']==event]
        event_id=sub_meta.index.to_numpy()
        self.data_length=np.round(data.shape[2]/srate)
        if channel=='all':
            self.data=data[event_id,:,:]
        else:
            self.data=data[event_id,channel,:]
        self.latency=latency
        self.fs=srate
    
    def topo(self,data,ch_names,srate=-1,ch_types = 'eeg'):
        if srate == -1:
            srate = self.fs
        info  = mne.create_info(ch_names=ch_names, sfreq = srate,ch_types = ch_types)
        evoked = mne.EvokedArray(data, info)
        evoked.set_montage('standard_1005')
        mne.viz.plot_topomap(evoked.data[:, 0], evoked.info,show=True)

    def Stacking_average(self, data = [], _axis = 0):
        if data ==[]:
            data = self.data
        data_mean = data
        data_mean = np.mean(data, axis=_axis)
        return data_mean
    
    def power_spectrum_periodogram(self,x):
        f, Pxx_den = signal.periodogram(x, self.fs)
        # plt.plot(f, Pxx_den)
        # plt.show()
        return f,Pxx_den

    def sum_y(self,x,y,x_inf,x_sup):
        sum_A=[]
        for i,freq in enumerate(x):
            if freq<=x_sup and freq>=x_inf:
                sum_A.append(y[i])     
        return np.mean(sum_A)

    def SNR(self,data=[]): 
        if data ==[]:
            data = self.data
        Y = fft(data, len(data))
        Y = np.abs(Y)
       
        f=np.linspace(0,len(data),num=len(data),endpoint=False)
        sum_A=[]
        snr_y=[]
        

        for j,center_freq in enumerate(f):
            for i,freq in enumerate(f):
                if abs(freq-center_freq)<=1.1:
                    sum_A.append(Y[i])
            snr_y.append(20*np.log(Y[j]/np.mean(sum_A)))
            sum_A=[]
        return f,snr_y
        

dataset = AlexMI()
paradigm = MotorImagery(
    channels=None,
    events=['right_hand', 'feet'],
    intervals=[(0, 3)], # 3 seconds
    srate=128
)

def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(3, 60, l_trans_bandwidth=2,h_trans_bandwidth=5,
        phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches

def epochs_hook(epochs, caches):
    # do something with epochs object
    print(epochs.event_id)
    caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
    return epochs, caches

def data_hook(X, y, meta, caches):
    # retrive caches from the last stage
    print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
    # do something with X, y, and meta
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches

paradigm.register_raw_hook(raw_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)

X, y, meta = paradigm.get_data(
    dataset,
    subjects=[8],
    return_concat=True,
    n_jobs=None,
    verbose=False)

sample1=freq_analysis(X,meta,event='right_hand',srate=128)
mean_data=sample1.Stacking_average(data = [],_axis=0)
y_list=[]
for i in range(mean_data.shape[0]):
    f,den=sample1.power_spectrum_periodogram(mean_data[i])
    y=sample1.sum_y(f,den,8,12)
    y_list.append(y)
y_list=np.array(y_list)
y_list=y_list[:,None]
sample1.topo(y_list,[
        'Fpz','F7','F3','Fz','F4','F8',
        'T7','C3','C4','T8',
        'P7','P3','Pz','P4','P8'
    ])





