import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)
# import pyedflib
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.integrate import simps

# PSG 데이터 리딩
sample_data_folder = 'C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\psg'
sample_data_raw_file = os.path.join(sample_data_folder, '1-2.edf')
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

# 데이터 60초만 그래프로 출력
raw.crop(tmax=60.)

#%% 데이터 정보, raw. info
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

#%% 평균 reference로 데이터 이상한 내용 제거
raw.set_eeg_reference('average', projection=True)
for title, proj in zip(['Original', 'Average'], [False, True]):
    fig = raw.plot(proj=proj, n_channels=len(raw))
    # make room for title
    fig.subplots_adjust(top=0.9)
    fig.suptitle('{} reference'.format(title), size='xx-large', weight='bold')

#%% ICA 진행 한 후, 필요없는 ICA 채널 제거하는 방법
"""
ica = mne.preprocessing.ICA(n_components=5, random_state = 97)
ica.fit(raw)
ica.exclude = [1,2]
ica.plot_properties(raw, picks=ica.exclude)
#%% ICA 에서 뇌파 데이터로 되돌리는 과정
orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)

orig_raw.plot(start=20)
raw.plot(start=20)
"""
#%% STI 한 내용 찾는 과정
events = mne.find_events(raw, stim_channel='PVT',output = 'onset', min_duration = 0.04,shortest_event =10)
print(events[:5])

fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)

raw.plot(events=events, duration=60, color='gray',
         event_color={1080: 'r', 1084: 'g', 1230: 'b', 127909: 'm', 127910: 'y'})

#%% 과도하게 튀는 데이터 영역 구분 후, 데이터 제거 과정
'''
# 데이터 한계점 설정
reject_criteria = dict(
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, reject=reject_criteria, preload = True)


sample1_epochs = epochs['127909']
sample2_epochs = epochs['127910']

#%% Time-frequency analysis

frequencies = np.arange(7, 30, 3)
power = mne.time_frequency.tfr_morlet(sample2_epochs, n_cycles = 2, return_itc = False, freqs=frequencies, decim =3)
power.plot(['Fz'])

#%% Estimating evoked responses
sample1_evoked = sample1_epochs.average()
sample2_evoked = sample2_epochs.average()

mne.viz.plot_compare_evokeds(dict(auditory=sample1_evoked, visual=sample2_evoked),
                             legend='upper left', show_sensors='upper right')
'''
#%% Slow drift 제거
EEG_channels = mne.pick_types(raw.info, eeg = True, ecg= True, eog = True)
raw.plot(duration=5, order=EEG_channels, proj=False, n_channels=len(EEG_channels), remove_dc=False,start=10)

for cutoff in (0.1, 0.2):
    raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
    fig = raw_highpass.plot(duration=5, order=EEG_channels, proj=False,
                            n_channels=len(EEG_channels), remove_dc=False,start=10)
    fig.subplots_adjust(top=0.9)
    fig.suptitle('High-pass filtered at {} Hz'.format(cutoff), size='xx-large',
                 weight='bold')
    
# 필터 작동
raw.filter(l_freq=0.2, h_freq=None)
    
#%%
# Power-line 잡음 제거
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
    
# 파워라인 필터 작용
raw.notch_filter(freqs=freqs, picks=EEG_picks)
#%% ECG에 의한 artifcat 검사
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
ecg_epochs.plot_image(combine='mean')

avg_ecg_epochs = ecg_epochs.average()
avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])

#%% EOG에 의한 artifact 검사
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine='mean')
eog_epochs.average().plot_joint()


#%% EOG annotations 진행
"""eog_events = mne.preprocessing.find_eog_events(raw)
onsets = eog_events[:, 0] / raw.info['sfreq'] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ['bad blink'] * len(eog_events)
blink_annot = mne.Annotations(onsets, durations, descriptions,
                              orig_time=raw.info['meas_date'])
raw.set_annotations(blink_annot)
# EOG anootation 진행한 내용 plot화
eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
raw.plot(events=eog_events, order=eeg_picks)
"""
#%% EOG and ECG artifact repair
# pick some channels that clearly show heartbeats and blinks
regexp = 'EEG'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
#%% ECG artifact repair

ecg_evoked = mne.preprocessing.create_ecg_epochs(raw)
# ecg_evoked.plot_joint()
# ecg_evoked.apply_baseline((None, None))
# ecg_evoked.plot_joint()

ecg_projs, events = compute_proj_ecg(raw, n_eeg=3, ch_name ='ECG',reject=None)

raw.del_proj()
for title in ('Without', 'With'):
    if title == 'With':
        raw.add_proj(ecg_projs)
    fig = raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} ECG projectors'.format(title), size='xx-large',
                 weight='bold')
    
#%% EOG artifact repair
eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline((None, None))
eog_evoked.plot_joint()
eog_projs, _ = compute_proj_eog(raw, n_eeg=1,reject=None, no_proj=True)
mne.viz.plot_projs_topomap(eog_projs, info=raw.info)

for title in ('Without', 'With'):
    if title == 'With':
        raw.add_proj(eog_projs)
    fig = raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} EOG projectors'.format(title), size='xx-large',
                 weight='bold')
    
#%% 필터링 완료한 데이터 복사
raw_filed = raw.copy()

# 데이터 파형 그리기
raw_filed.plot_psd()

#%% Extracting data by index 
sampling_freq = raw_filed.info['sfreq']
start_stop_seconds = np.array([11, 13])
start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
channel_index = 0
raw_filed_selection = raw_filed[channel_index, start_sample:stop_sample]
print(raw_filed_selection)

x = raw_filed_selection[1]
y = raw_filed_selection[0].T
plt.plot(x, y)

#%%
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



Data = raw_filed[0:5,:]
sigbufs = Data[0]
bp_all_dataset = pd.DataFrame()
# edf 파일로부터 생체데이터 취득 및 변수에 저장
# for i in range(1,2):
#     for j in range(1,4): 
        # file_name = f'C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\psg\\{i}-{j}.edf'
        # if os.path.isfile(file_name):
        # file = os.path.join(file_name)
        # f = pyedflib.EdfReader(file)
        # n = f.signals_in_file
        # signal_labels = f.getSignalLabels()
        # sigbufs = np.zeros((n, f.getNSamples()[0]))
        # for n in np.arange(n):
        #     sigbufs[n, :] = f.readSignal(n)

        # 생체데이터 Plot 그리기
        # 측정 위치 선정
Node_place = 3
time = np.arange(sigbufs[Node_place,:].size)/sfreq
sample_number = int(np.size(time)/sfreq)

fig, ax = plt.subplots(1, 1, figsize = (12, 4 ))
plt.plot(time, sigbufs[Node_place,:], lw =1.5, color = 'k')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.xlim([10, 40])
# plt.title(f'{i}-{j} data')
sns.despine()


# Power spectral density 가시화

# Define window length (4 seconds)
win = 5 * sfreq
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
plt.xlim([0, 10])
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



# 생체 데이터 BandPower 값 구하기
'''
delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz).
'''
window_sec = 5*sfreq
delta = [0.2, 4]
theta = [4, 8]
alpha = [8, 12]
beta = [12, 30]
gamma = [30, 100]

# bp_delta = bandpower(sigbufs[0,:],sfreq,delta,window_sec, relative=True)

# bp_dataset = pd.Series(index=[f'bp_delta_{i}-{j}',f'bp_theta_{i}-{j}',f'bp_alpha_{i}-{j}',f'bp_beta_{i}-{j}'])

for Node_place in range(len(sigbufs[:,0])):
        
    bp_delta = bandpower(sigbufs[Node_place,:], sfreq, delta, window_sec, relative=True)
    bp_theta = bandpower(sigbufs[Node_place,:], sfreq, theta, window_sec, relative=True)
    bp_alpha = bandpower(sigbufs[Node_place,:], sfreq, alpha, window_sec, relative=True)
    bp_beta = bandpower(sigbufs[Node_place,:], sfreq, beta, window_sec, relative=True)
    bp_all = [bp_delta, bp_theta, bp_alpha, bp_beta]
    s1 = pd.DataFrame(bp_all,index=['bp_delta', 'bp_theta','bp_alpha','bp_beta'])
    k1 = pd.DataFrame([KSS_data[0]],columns = ['KSS'])
    
    s1 = s1.transpose()
    
    s1.reset_index()
    k1.reset_index()
    
    bp_all_dataset = pd.concat([s1, k1], axis = 1)
    
bp_all_dataset


# bp_all_dataset= bp_all_dataset.append(bp_dataset)
        
    
 

    


#%% Rejecting Epochs based on channel ampitude
"""
eject_criteria = dict(mag=3000e-15,     # 3000 fT
                       grad=3000e-13,    # 3000 fT/cm
                       eeg=100e-6,       # 100 µV
                       eog=200e-6)       # 200 µV

flat_criteria = dict(mag=1e-15,          # 1 fT
                     grad=1e-13,         # 1 fT/cm
                     eeg=1e-6)           # 1 µV

epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, reject_tmax=0,
                    reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=False, preload=True)
epochs.plot_drop_log()
"""
#%% ECG anootations 진행
"""
ecg_events = mne.preprocessing.find_ecg_events(raw)
onsets = ecg_events[:,0] / raw.info['sfreq'] - 0.25
durations = [0.5] * len(ecg_epochs)
descriptons = ['bad Heartbeat'] * len(ecg_events)
Heartbeat_annot = mne.Annotations(onsets, durations, descriptions, orig_time=raw.info['meas_date'])

raw.set_annotations(Heartbeat_annot)

# ECG annotation 진행한 내용 plot화
eeg_picks = mne.pick_types(raw.info, eeg=True)
raw.plot(events=ecg_events, order=eeg_picks)
"""