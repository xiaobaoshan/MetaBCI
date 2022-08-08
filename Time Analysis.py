import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from brainda.datasets import AlexMI
from brainda.paradigms import MotorImagery
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.decomposition import FBCSP
from brainda.algorithms.decomposition.base import generate_filterbank
import mne
import matplotlib.pyplot as plt

class Feature_Extract():
    def __init__(self,data, meta, event, srate, latency = 0.0 , channel = 'all'):
        sub_meta = meta[ meta['event']==event ]
        event_id = sub_meta.index.to_numpy()
        self.data_length = np.round(data.shape[2]/srate)
        if channel == 'all':
            self.data = data[event_id, :, :]
        else:
            self.data = data[event_id,channel,:]
        self.latency = latency
        self.fs = srate

        # return self.data

    def Stacking_average(self, data = [], _axis = 0):  # data 维度：试次*导联*时间
        if data == []:
            data = self.data
        data_mean = data
        data_mean = np.mean(data, axis=_axis)  # 在试次维度平均
        return data_mean

    def Peak_amplitude(self, data=[], time_start=0, time_end=1):
        if data == []:
            data = self.Stacking_average()
        # data_mean = np.mean(data, 0)
        a_max = np.max(data[time_start:time_end])
        return a_max

    def Average_amplitude(self, data=[], time_start=0, time_end=1):
        if data == []:
            data = self.Stacking_average()
        a_mean = np.mean(data[time_start:time_end])
        return a_mean

    def Peak_Latency(self,data=[], time_start=0, time_end=1):
        if data == []:
            data = self.data
        peak_amp = self.Peak_amplitude(data, time_start, time_end)
        peak_loc = np.argmax(data[time_start:time_end])+time_start
        return peak_loc , peak_amp

    def Average_Latency(self,data=[], time_start=0, time_end=1):
        if data == []:
            data = self.data
        ave_amp = self.Average_amplitude(data, time_start, time_end)
        sample = time_end-time_start-1
        half_average = np.sum(data) / 2
        integal = []
        for samp_i in range(sample):
            integal.append(np.abs(np.sum(data[0:samp_i])-half_average))
        ave_loc = np.argmin(integal)+time_start

        return ave_loc, ave_amp

    def Plot_single_trial(self, data, sample_num, axes=None,
                          amp_mark = False, time_start=0, time_end=1):
        latency = self.latency
        data_mean = data
        import matplotlib.pyplot as plt
        fs = self.fs
        # data_length = self.data_length
        ax = axes if axes else plt.gca()
        t = np.arange(latency,sample_num/fs+latency,1/fs)
        plt.plot(t,data_mean)
        if amp_mark:
            func_str = 'self.'+amp_mark.capitalize()+'_Latency'
            loc,amp = eval(func_str)(data_mean,time_start,time_end)
            plt.scatter(t[loc],amp,c='r',marker='o')
            pass
        return loc,amp,ax

    def Plot_topomap(self,data,point,ch_names,fig,srate=-1, ch_types = 'eeg',
                               axes = None):
        if srate == -1:
            srate = self.fs
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
        # (n_epochs, n_chans, n_times)
        # data = np.mean(SSVEP_Slides[:,8,:,:],axis=0)
        evoked = mne.EvokedArray(data, info)
        evoked.set_montage('standard_1005')
        ax = axes if axes else plt.gca()
        l = 0.92
        b = 0.2
        w = 0.015
        h = 0.6
        aximage, countour = mne.viz.plot_topomap(evoked.data[:, point],
                                                 evoked.info, show=False, cmap='jet',
                                                 vmin=0, vmax=0.5, contours=6)
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        plt.colorbar(aximage, cax=cbar_ax)

        return aximage












