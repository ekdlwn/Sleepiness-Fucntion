# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:06:39 2020

@author: ekdlw
"""

import numpy as np
import mne

# 그래프 scaling
Plot_scaling = dict(mag=1e-12, grad=4e-11, eeg=20e-4, eog=150e-6, ecg=5e-4,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4, whitened=1e2)

# CSV file 데이터 읽어서 Numpy array로 입력
data = np.loadtxt(r'C:\Users\ekdlw\Desktop\Bio Data Analysis\CSV_file\1230240719_2.csv',delimiter = ',')
data = np.transpose(data)


# Some information about the channels
ch_name = ['Fp1', 'Fp2', 'F3', 'F4', 'P4', 'P3', 'O1', 'O2', 'EOG-V', 'EOG-H', 'EMG', 'None_1', 'ECG', 'RESP', 'PPG', 'SpO2', 'HR', 'GSR', 'None_2', 'None_3', 'None_4', 'None_5', 'None_6', 'TimeStamps']

# 데이터에 몽타주 그려주기
raw.set_montage('standard_1020')
# Sampling rate of the machine
sfreq = 500 # Hz

# create the info structure needed by MNE
info = mne.create_info(ch_name, sfreq)

# Finally, create the Raw object
raw = mne.io.RawArray(data, info)

# raw 데이터 채널 타입 변경
raw.set_channel_types({'Fp1':'eeg','Fp2':'eeg','F3':'eeg','F4':'eeg','P4':'eeg','P3':'eeg','O1':'eeg','O2':'eeg','EOG-V':'eog','EOG-H':'eog','EMG':'emg','RESP':'resp','ECG':'ecg'})

# raw 데이터 채널 버리기
raw.drop_channels(['None_1','None_2','None_3','None_4','None_5','None_6','GSR','HR','SpO2','PPG','TimeStamps'])


# plot it!
raw.plot(scalings = 'auto')