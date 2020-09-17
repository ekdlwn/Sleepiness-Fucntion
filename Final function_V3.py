import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog, ICA)
# import pyedflib
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.integrate import simps
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

def bandpower(data, sf, band, window_sec=None, relative=False):
    '''Compute the average power of the signal x in a specific frequency band.
    
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
    '''
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
            


# KSS 점수 자동 변경 용 변수
n = -1

# 빈 DataFrame
Final_dataset = pd.DataFrame([])

# rejection 설정 조건
reject = dict(eeg=100e-6, # V (EEG channels)
              eog=300e-6 # V (EOG channels)
              )

# 생체 데이터 BandPower 값 구하기           

delta = [0.2, 4]
theta = [4, 8]
alpha = [8, 12]
beta = [12, 30]
gamma = [30, 100]
            

for i in range(1,2):
    for j in range(2,3):
        # PSG 데이터 리딩
        sample_data_folder = f'C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\psg\\{i}-{j}.edf'
        n += 1
        if os.path.isfile(sample_data_folder):
            sample_data_raw_file = os.path.join(sample_data_folder)
            raw = mne.io.read_raw_edf(sample_data_raw_file, exclude=['Cam-Sync'],eog=['EOG-V','EOG-H'],preload=True,stim_channel='PVT')
            
            # KSS 데이터 리딩
            KSS_data_file ='C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\KSS.csv'
            KSS_data = np.loadtxt(KSS_data_file, dtype = int, delimiter=',')
            
            # ECG 데이터 채널 변경
            raw.set_channel_types({'ECG': 'ecg','EMG': 'emg'})
            
            # 데이터에 몽타주 그려주기
            raw.set_montage('standard_1020')
            
            mapping={'Fz':'EEG001','Cz':'EEG002','C3':'EEG003','C4':'EEG004', 'Pz':'EEG005', 'EOG-V':'EOG001','EOG-H':'EOG002'}
            raw.rename_channels(mapping)
            
            # 데이터 frequency 저장
            sfreq = raw.info['sfreq']
            
            
            #데이터 리샘플링
            raw.resample(sfreq=256)
            
            # BAD 채널 제거
            raw.interpolate_bads(reset_bads = False)
            

            #%%
            # STI 시간 Annotation
            sti_events = mne.find_events(raw, stim_channel='PVT', output = 'onset', min_duration = .06, shortest_event = 256 )
            onsets = sti_events[:,0] / raw.info['sfreq']-0.5
            durations = [2] * len(onsets)
            descriptions = ['bad_stimulation'] * len(sti_events)
            
            onsets = np.append(onsets,[0])
            durations.append(10)
            descriptions.append("bad_ten")
            sti_annot = mne.Annotations(onsets, durations, descriptions)
            raw.set_annotations(sti_annot)
            
            raw.plot(start = 20)
            
#%%
            # eeg_data = raw.copy().pick_types(meg=False, eeg=True, exclude=[])
            # eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads=False)
            ''' Bad channel 제거 확인할 수 있는 가시화

            for title, data in zip(['orig.', 'interp.'], [eeg_data, eeg_data_interp]):
                fig = data.plot(butterfly=True, color='#00000102', bad_color='r')
                fig.subplots_adjust(top=0.9)
                fig.suptitle(title, size='xx-large', weight='bold')
            '''
            
            '''# 데이터 정보, raw. info
            n_time_samps = raw.n_times
            time_secs = raw.times
            ch_names = raw.ch_names
            n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
            print('the (cropped) sample data object has {} time samples and {} channels.'
                  ''.format(n_time_samps, n_chan))
            print('The last time sample is at {} seconds.'.format(time_secs[-1]))
            print('The first few channel names are {}.'.format(', '.join(ch_names[:3])))
            print()  # insert a blank line in the output
            
            # some examples of raw.info:
            print('bad channels:', raw.info['bads'])  # chs marked "bad" during acquisition
            print(raw.info['sfreq'], 'Hz')            # sampling frequency
            print(raw.info['description'], '\n')      # miscellaneous acquisition info
            
            print(raw.info)
            '''


            '''# Bad data span 제거
            eog_events = mne.preprocessing.find_eog_events(raw)
            onsets = eog_events[:, 0] / raw.info['sfreq'] - 0.25
            durations = [0.5] * len(eog_events)
            descriptions = ['bad blink'] * len(eog_events)
            blink_annot = mne.Annotations(onsets, durations, descriptions,
                              orig_time=raw.info['meas_date'])
            raw.set_annotations(blink_annot)
            
            eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
            raw.plot(events=eog_events)
            '''
            
            '''  # Artifact 검사
            ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, ch_name = 'ECG',reject = reject)
            ecg_epochs.plot_image(combine='mean')
            avg_ecg_epochs = ecg_epochs.average()
            avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))
            avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
            
            eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2),reject = reject)
            eog_epochs.plot_image(combine='mean')
            eog_epochs.average().plot_joint()
            '''
            
            '''#STI 한 내용 찾는 과정
            events = mne.find_events(raw, stim_channel='PVT',output = 'onset', min_duration = 0.04,shortest_event =10)
            '''
            raw_unfil = raw.copy()
            # Slow drift 제거 및 데이터 필터링
            # EEG_channels = mne.pick_types(raw.info, eeg = True, ecg= True, eog = True)
            # raw.plot(duration=10, order=EEG_channels, proj=False, n_channels=len(EEG_channels), remove_dc=False,start=10)
            raw.filter(l_freq = 0.5, h_freq = 50)


            ''' # Bad channel 제거
            raw.crop(tmin = 0, tmax = 3).load_data()
            eeg_data = raw.copy().pick_types(meg=False, eeg = True, exclude = [])
            eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads = False)
            
            for title, data in zip(['org.', 'interp.'], [eeg_data, eeg_data_interp]):
                fig = data.plot(butterfly=True, color = '#00000022', bad_color = 'r')
                fig.subplots_adjust(top = 0.9)
                fig.subtitle(title, size = 'xx-large', weight = 'bold')
            '''
            ''' filter 예시
        
            for cutoff in (0.1, 0.2):
                raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
                fig = raw_highpass.plot(duration=10, order=EEG_channels, proj=False,
                                        n_channels=len(EEG_channels), remove_dc=False,start=10)
                fig.subplots_adjust(top=0.9)
                fig.suptitle('High-pass filtered at {} Hz'.format(cutoff), size='xx-large',
                             weight='bold')
            '''
            '''# 평균 reference로 데이터 이상한 내용 제거
            raw.set_eeg_reference('average', projection=True)
            for title, proj in zip(['Original', 'Average'], [False, True]):
                fig = raw.plot(proj=proj, n_channels=len(raw),start = 10)
                # make room for title
                fig.subplots_adjust(top=0.9)
                fig.suptitle('{} reference'.format(title), size='xx-large', weight='bold')
            '''
            ''' ICA를 이용한 데이터 전처리
            ica = ICA(n_components=5, random_state=97)
            ica.fit(raw)
            
            # ica.plot_sources(raw)
            # ica.plot_overlay(raw, exclude=[0], picks='eeg')
            
            ica.exclude = []
            # find which ICs match the EOG pattern
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG001',threshold =1.4 )
            ica.exclude = eog_indices
            
            # barplot of ICA component "EOG match" scores
            ica.plot_scores(eog_scores)
            
            ica.plot_properties(raw, picks=eog_indices)
            
            # plot ICs applied to raw data, with EOG matches highlighted
            ica.plot_sources(raw)
            
            # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
            ica.plot_sources(eog_evoked)
            
            ica.apply(raw)
            raw.plot(duration = 10, start = 20)
            '''
            '''# Power-line 잡음 제거
            def add_arrows(axes):
                # add some arrows at 60 Hz and its harmonics
                for ax in axes:
                    freqs = ax.lines[-1].get_xdata()
                    psds = ax.lines[-1].get_ydata()
                    for freq in (60, 120, 180, 240):
                        idx = np.searchsorted(freqs, freq)
                        # get ymax of a small region around the freq. of interest
                        y = psds[(idx - 4):(idx + 5)].max()
                        ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                                  width=0.1, head_width=3, length_includes_head=True)
                        
            EEG_picks = mne.pick_types(raw.info,eeg = True, ecg= True, eog = True)  # meg=True, eeg=False are the defaults
            freqs = (60, 120, 180, 240)
            raw_notch = raw.copy().notch_filter(freqs=freqs, picks=EEG_picks)
            for title, data in zip(['Un', 'Notch '], [raw, raw_notch]):
                fig = data.plot_psd(fmax=250, average=True)
                fig.subplots_adjust(top=0.85)
                fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
                add_arrows(fig.axes[:2])
                
            raw.notch_filter(freqs=freqs, picks=EEG_picks)
            '''
            '''# ECG에 의한 artifcat 검사
            ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
            ecg_epochs.plot_image(combine='mean')
            
            avg_ecg_epochs = ecg_epochs.average()
            avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
            
            # EOG에 의한 artifact 검사
            eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
            eog_epochs.plot_image(combine='mean')
            eog_epochs.average().plot_joint()
                    
            # EOG annotations 진행
            eog_events = mne.preprocessing.find_eog_events(raw)
            onsets = eog_events[:, 0] / raw.info['sfreq'] - 0.25
            durations = [0.5] * len(eog_events)
            descriptions = ['bad blink'] * len(eog_events)
            blink_annot = mne.Annotations(onsets, durations, descriptions,
                                          orig_time=raw.info['meas_date'])
            raw.set_annotations(blink_annot)
            # EOG anootation 진행한 내용 plot화
            eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
            raw.plot(events=eog_events, order=eeg_picks)
            '''

            # EOG and ECG artifact repair
            
            # pick some channels that clearly show heartbeats and blinks
            regexp = 'EEG'
            artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
            # raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
            # ECG artifact repair
            
            ecg_evoked = create_ecg_epochs(raw,ch_name ='ECG').average()
            ecg_evoked.apply_baseline((None, None))
            # ecg_evoked.plot_joint()
            ecg_projs, events = compute_proj_ecg(raw, n_eeg=1, ch_name ='ECG',reject=None)
            
            '''for title in ('Without', 'With'):
                if title == 'With':
                    raw.add_proj(ecg_projs)
                fig = raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
                fig.subplots_adjust(top=0.9)  # make room for title
                fig.suptitle('{} ECG projectors'.format(title), size='xx-large',
                             weight='bold')
            '''
            #EOG artifact repair
            eog_evoked = create_eog_epochs(raw).average()
            eog_evoked.apply_baseline((None, None))
            # eog_evoked.plot_joint()
            eog_projs, _ = compute_proj_eog(raw, n_eeg=1,reject=None, no_proj=True)
            mne.viz.plot_projs_topomap(eog_projs, info=raw.info)
            
            '''for title in ('Without', 'With'):
                if title == 'With':
                    raw.add_proj(eog_projs)
                fig = raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
                fig.subplots_adjust(top=0.9)  # make room for title
                fig.suptitle('{} EOG projectors'.format(title), size='xx-large',
                             weight='bold')
            '''

            # 필터링 완료한 데이터 복사
            raw_filed = raw.copy()
            
            # EEG 데이터로부터 Power spectral 얻어내는 코드
            # 주요 함수들 import
            Data = raw_filed[0:5,:]
            # Data = raw_unfil[0:5, :]
            sigbufs = Data[0]
            bp_region_dataset = pd.DataFrame()
            bp_all_dataset = pd.DataFrame()

            # Power spectral density 가시화
            # Define window length (5 seconds)
            win = 20 * sfreq
            freqs, psd = signal.welch(sigbufs[0,:], sfreq, nperseg=win)
            
            # Plot the power spectrum
            sns.set(font_scale=1.2, style='white')
            plt.figure(figsize=(8, 4))
            plt.plot(freqs, psd, color='k', lw=2)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power spectral density (V^2 / Hz)')
            plt.ylim([0, psd.max() * 1.1])
            plt.title("Welch's periodogram")
            plt.xlim([0, 50])
            sns.despine()

            
            # Define delta lower and upper limits
            low, high = 0.5, 4
            
            # Find intersecting values in frequency vector
            idx_delta = np.logical_and(freqs >= low, freqs <= high)
            
            # Plot the power spectral density and fill the delta area
            plt.figure(figsize=(7, 4))
            plt.plot(freqs, psd, lw=2, color='k')
            plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power spectral density (uV^2 / Hz)')
            plt.xlim([0, 50])
            plt.ylim([0, psd.max() * 1.1])
            plt.title("Welch's periodogram")
            sns.despine()
            
            # Frequency resolution
            freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
            
            # Compute the absolute power by approximating the area under the curve
            delta_power = simps(psd[idx_delta], dx=freq_res)
            print('Absolute delta power: %.3f uV^2' % delta_power)
            
            # Relative delta power (expressed as a percentage of total power)
            total_power = simps(psd, dx=freq_res)
            delta_rel_power = delta_power / total_power
            print('Relative delta power: %.3f' % delta_rel_power)
            
            # FFT 분석 간격 설정
            window_sec = 5*sfreq
            
            bp_merge_dataset = pd.DataFrame([])
            # 데이터 쪼개는 window 크기
            delta_time = 1
            # 데이터 전체 시간
            Data_logging_time = 600
            for time in range(0,int(Data_logging_time/delta_time)):
                frame = time * 256 * delta_time
                delta_frame = delta_time * 256
                bp_all_dataset = pd.DataFrame([])
                for Node_place in range(len(sigbufs[:,0])):
                    # bp_delta = bandpower(sigbufs[Node_place,frame:frame+delta_frame], sfreq, delta, window_sec)
                    bp_theta = bandpower(sigbufs[Node_place,frame:frame+delta_frame], sfreq, theta, window_sec)
                    bp_alpha = bandpower(sigbufs[Node_place,frame:frame+delta_frame], sfreq, alpha, window_sec)
                    bp_beta = bandpower(sigbufs[Node_place,frame:frame+delta_frame], sfreq, beta, window_sec)
                    
                    bp_relative_theta = bp_theta/(bp_theta+bp_alpha+bp_beta)
                    bp_relative_alpha = bp_alpha/(bp_theta+bp_alpha+bp_beta)
                    bp_relative_beta = bp_beta/(bp_theta+bp_alpha+bp_beta)
                    
                    
                    bp_all = np.array([[bp_relative_theta, bp_relative_alpha, bp_relative_beta]])
                    s1 = pd.DataFrame(bp_all,columns=[f'bp_theta_{Node_place}',f'bp_alpha_{Node_place}',f'bp_beta_{Node_place}'],index = [f'{i}-{j}-{time}'])
                    bp_all_dataset = pd.concat([bp_all_dataset, s1], axis = 1)
                k1 = pd.DataFrame([KSS_data[n]],columns = ['KSS'], index = [f'{i}-{j}-{time}'])
                bp_all_dataset = pd.concat([bp_all_dataset, k1], axis = 1)
                bp_merge_dataset = bp_merge_dataset.append(bp_all_dataset)
                
            Final_dataset = Final_dataset.append(bp_merge_dataset)
            

        
Final_dataset.to_csv('C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\Final_result.csv')

    
'''#%%
    
data = sigbufs[0, 0: 512*5]
waveletname = 'sym5'
 
fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6,6))
for ii in range(5):
    (data, coeff_d) = pywt.dwt(data, waveletname)
    axarr[ii, 0].plot(data, 'r')
    axarr[ii, 1].plot(coeff_d, 'g')
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])
plt.tight_layout()
plt.show()

#%%
import os
import pywt
#from wavelets.wave_python.waveletFunctions import *
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def plot_wavelet(time, signal, scales, 
                 waveletname = 'cmor', 
                 cmap = plt.cm.seismic, 
                 title = 'Wavelet Transform (Power Spectrum) of signal', 
                 ylabel = 'Period (years)', 
                 xlabel = 'Time'):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()

def plot_signal_plus_average(time, signal, average_over = 5):
    fig, ax = plt.subplots(figsize=(15, 3))
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Signal Amplitude', fontsize=18)
    ax.set_title('Signal + Time Average', fontsize=18)
    ax.set_xlabel('Time', fontsize=18)
    ax.legend()
    plt.show()
    
def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave
    
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def plot_fft_plus_power(time, signal):
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt
    
    fig, ax = plt.subplots(figsize=(15, 3))
    variance = np.std(signal)**2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2     # FFT power spectrum
    ax.plot(f_values, fft_values, 'r-', label='Fourier Transform')
    ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
    ax.set_xlabel('Frequency [Hz / year]', fontsize=18)
    ax.set_ylabel('Amplitude', fontsize=18)
    ax.legend()
    plt.show()

# dataset = sigbufs[0,:]
df_nino = pd.DataFrame(sigbufs[0,0:256*1])
N = df_nino.shape[0]
t0=0
dt=256
time = np.arange(0, N) * dt + t0
signal = df_nino.values.squeeze()

scales = np.arange(1, 56)
plot_signal_plus_average(time, signal)
plot_fft_plus_power(time, signal)
plot_wavelet(time, signal, scales)
'''