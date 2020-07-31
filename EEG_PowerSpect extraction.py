'''
EEG 데이터로부터 Power spectral 얻어내는 코드

delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz).


'''

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    import numpy as np
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

# 주요 함수들 import

import pyedflib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal


bp_all_dataset = pd.DataFrame()
# edf 파일로부터 생체데이터 취득 및 변수에 저장
for i in range(1,2):
    for j in range(1,4): 
        file_name = f'C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\psg\\{i}-{j}.edf'
        if os.path.isfile(file_name):
            file = os.path.join(file_name)
            f = pyedflib.EdfReader(file)
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            sigbufs = np.zeros((n, f.getNSamples()[0]))
            for n in np.arange(n):
                sigbufs[n, :] = f.readSignal(n)

            # 생체데이터 Plot 그리기
            sf = 512
            # 측정 위치 선정
            Node_place = 1
            time = np.arange(sigbufs[Node_place,:].size)/sf
            sample_number = int(np.size(time)/sf)
            
            fig, ax = plt.subplots(1, 1, figsize = (12, 4 ))
            plt.plot(time, sigbufs[Node_place,:], lw =1.5, color = 'k')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Voltage')
            plt.xlim([time.min(), time.max()])
            plt.title(f'{i}-{j} data')
            sns.despine()
            
            # 생체 데이터 BandPower 값 구하기
            '''
            delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz).
            '''
            delta = [0.5, 4]
            theta = [4, 8]
            alpha = [8, 12]
            beta = [12, 30]
            gamma = [30, 100]
            
            bp_dataset = pd.Series(index=[f'bp_delta_{i}-{j}',f'bp_theta_{i}-{j}',f'bp_alpha_{i}-{j}',f'bp_beta_{i}-{j}'])
            for n in range(0,sample_number):
                N = int(n * 512)
                
                bp_delta = bandpower(sigbufs[Node_place,N:N + 512], sf, delta, relative=True)
                bp_theta = bandpower(sigbufs[Node_place,N:N+512], sf, theta, relative=True)
                bp_alpha = bandpower(sigbufs[Node_place,N:N+512], sf, alpha, relative=True)
                bp_beta = bandpower(sigbufs[Node_place,N:N+512], sf, beta, relative=True)
                bp_all = [bp_delta, bp_theta, bp_alpha, bp_beta]
                s1 = pd.Series(bp_all,index=[f'bp_delta_{i}-{j}',f'bp_theta_{i}-{j}',f'bp_alpha_{i}-{j}',f'bp_beta_{i}-{j}'])
                
                bp_dataset = pd.concat([bp_dataset, s1],axis=1)
            bp_all_dataset= bp_all_dataset.append(bp_dataset)
                
            
     

    
