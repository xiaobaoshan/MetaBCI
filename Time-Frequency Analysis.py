import numpy as np
from scipy.signal import detrend, stft
import mne
from scipy.signal import hilbert
import matplotlib.pyplot as plt

class MWT():
    def __init__(self, xtimes, Fs, omega, sigma):
        self.xtimes = xtimes
        self.Fs = Fs
        self.omega = omega
        self.sigma = sigma
        self.N_F = self.xtimes.shape[0]

    def fit(self, X, Y):
        pass

    def transform(self, X):
        '''
        Calculate Morlet wavelet transform
        :param X: ndarray (nTrials, nChannels, nTimes)
        :return:
            S_features: ndarray (nTrials, nChannels, N_F, nTimes), complex values of Morlet wavelet transform
            P_features: ndarray (nTrials, nChannels, N_F, nTimes), squared magnitude of MWT (scaleogram)
        '''
        print('Calculating Morlet Wavelet Transform ... ')
        nTrials = X.shape[0]
        nChannels = X.shape[1]
        nTimes = X.shape[2]
        S_features = np.zeros((nTrials, nChannels, self.N_F, nTimes))
        P_features = np.zeros_like(S_features)
        for trial_i in range(nTrials):
            data = X[trial_i, :, :]
            P_features[trial_i, :, :, :], S_features[trial_i, :, :, :] = self.morlet_wavelet(data)

        return P_features, S_features

    def morlet_wavelet(self, data):
        '''
        This is a sub-function to calculate Morlet wavelet transform of one trial.
        :param data: ndarray (nChannels, nTimes)
        :return:
            S_features: ndarray (nChannels, N_F, nTimes), complex values of Morlet wavelet transform
            P_features: ndarray (nChannels, N_F, nTimes), squared magnitude of MWT (scaleogram)
        '''
        data = detrend(data, axis=1, type='linear')
        N_T = data.shape[1]
        N_C = data.shape[0]
        f = self.xtimes/self.Fs
        S = np.zeros((N_C, self.N_F, N_T))
        P = np.zeros((N_C, self.N_F, N_T))

        L_hw = N_T
        for fi in range(self.N_F):
            scaling_factor = self.omega/f[fi]
            u = (-np.arange(-L_hw, L_hw+1))/scaling_factor
            hw = np.sqrt(1/scaling_factor)*np.exp(-(u**2)/(2*self.sigma**2))*np.exp(1j*2*np.pi*self.omega*u)
            for ci in range(N_C):
                S_full = np.convolve(data[ci, :], hw.conjugate())
                S[ci, fi, :] = S_full[L_hw: L_hw+N_T]

        P = np.abs(S)**2
        return P, S


class STFT():
    def __init__(self, fs=1.0, window='hann', nperseg=256, noverlap=None,
                 nfft=None, detrend=False, return_onesided=True,
                 boundary='zeros', padded=True, axis=- 1):
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_oneside = return_onesided
        self.boundary = boundary
        self.padded = padded
        self.axis = axis

    def fit(self, X, Y):
        pass

    def transform(self, X):
        '''
        Calculate STFT
        :param X: ndarray (nTrials, nChannels, nTimes)
        :return:
        '''
        f, t, Zxx = stft(X, self.fs, self.window, self.nperseg, self.noverlap, self.nfft,
                        self.detrend, self.return_oneside, self.boundary, self.padded, self.axis)
        return f, t, Zxx


class Topoplot():
    def __init__(self, sfreq, chan_names, ch_types='eeg'):
        self.sfreq = sfreq
        self.chan_names = chan_names
        self.ch_types = ch_types

    def fit(self, X, Y):
        pass

    def transform(self, X):
        montage_type = mne.channels.make_standard_montage('standard_1020', head_size=0.1)
        epoch_info = mne.create_info(ch_names=self.chan_names, ch_types=self.ch_types, sfreq=self.sfreq)
        epoch_info.set_montage(montage_type)

        vmax = np.max(X) # The value specifying the upper bound of the color range.
        vmin = np.min(X)  # The value specifying the lower bound of the color range.

        #%% 画图
        fig, ax = plt.subplots(1,  figsize=(4, 4),
                                sharex=True, sharey=True)
        im, cn = mne.viz.plot_topomap(X, epoch_info, axes=ax,
                                    show=False, vmax=vmax, vmin=vmin, cnorm=None)
        ax.set_title('topomap', fontsize=25)
        plt.colorbar(im)
        plt.show()

class fun_hilbert():
    def __init__(self, fs, N=None, axis=-1):
        self.N = N
        self.axis = axis

    def fit(self, X, Y):
        pass

    def transform(self, X):
        analytic_signal = hilbert(X, self.N, self.axis)
        realEnv = np.real(analytic_signal)
        imagEnv = np.imag(analytic_signal)
        angle = np.angle(analytic_signal)
        return analytic_signal, realEnv, imagEnv, angle


