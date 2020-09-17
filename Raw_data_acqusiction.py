import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

n = -1
Data = np.zeros(6)
Data = np.reshape(Data,(1,6))
for i in range(1,15):
    for j in range(1,4):
        
        sample_data_folder = f'C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\psg\\{i}-{j}.edf'

        n += 1
        if os.path.isfile(sample_data_folder):
            # if 문 들어간 횟수
            counter = 0
            sample_data_raw_file = os.path.join(sample_data_folder)
            raw = mne.io.read_raw_edf(sample_data_raw_file, exclude=['Cam-Sync'],eog=['EOG-V','EOG-H'],preload=True,stim_channel='PVT' )
            
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
            
            raw.filter(l_freq = 0.5, h_freq = 50)
            
            N_KSS = [KSS_data[n]] * 76800
            N_KSS = np.array(N_KSS)
            N_KSS = np.reshape(N_KSS,[76800,1])
            
            Data_cache = raw[0:7,:]
            Data_cache = np.array(Data_cache[0])
            Data_cache = np.transpose(Data_cache)
            Data_cache = Data_cache[-76801:-1,0:5]
            
            Data_cache = np.hstack((Data_cache,N_KSS))
            Data = np.vstack((Data,Data_cache))
            
dataset = pd.DataFrame(Data)
dataset.colums = ["Fz", "Cz", "C3", "C4", "Pz"]
dataset.to_csv(f'C:\\Users\\ekdlw\\Desktop\\20.07.16\\result_raw_total.csv')