import pyedflib
import numpy as np
import os
sigbufs = []
file_name = os.path.join('C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\psg\\4-3.edf')
f = pyedflib.EdfReader(file_name)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
    
 # 생체데이터 Plot 그리기
sf = 512
# 측정 위치 선정
Node_place = 7
time = np.arange(sigbufs[Node_place,0:sf*30].size)/sf
sample_number = int(np.size(time)/sf)

fig, ax = plt.subplots(1, 1, figsize = (12, 4 ))
plt.plot(time, sigbufs[Node_place,0:sf*30], lw =1.5, color = 'k')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.xlim([time.min(), time.max()])
plt.title('EOG-V data')
sns.despine()

#%%
from scipy import signal

# Define window length (4 seconds)
win = 4 * sf
freqs, psd = signal.welch(sigbufs[1,0:15359], sf, nperseg=win, noverlap = 1)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8,4))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.xlim([0, 26])
sns.despine()

#%%

delta = [0.5, 4]
theta = [4, 8]
alpha = [8, 12]
beta = [12, 26]

# Define delta lower and upper limits
low, high = 0.5, 4

# Find intersecting values in frequency vector
idx_delta = np.logical_and(freqs >= low, freqs <= high)

idx_delta = np.logical_and(freqs >= delta[0], freqs <= delta[1])
idx_theta = np.logical_and(freqs >= theta[0], freqs <= theta[1])
idx_alpha = np.logical_and(freqs >= alpha[0], freqs <= alpha[1])
idx_beta = np.logical_and(freqs >= beta[0], freqs <= beta[1])

# Plot the power spectral density and fill the delta area
plt.figure(figsize=(7, 4))
plt.plot(freqs, psd, lw=2, color='k')
plt.fill_between(freqs, psd, where=idx_delta, color='sky blue')
plt.fill_between(freqs, psd, where=idx_theta, color='red')
plt.fill_between(freqs, psd, where=idx_alpha, color='green')
plt.fill_between(freqs, psd, where=idx_beta, color='yellow')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (uV^2 / Hz)')
plt.xlim([0, 30])
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
sns.despine()

#%%
from scipy.integrate import simps

# Frequecny resolution
freq_res = freqs[1] - freqs[0] # = 1/4 = 0.25

# Compute the absolute power by approximating the area under the curve
delta_power = simps(psd[idx_delta], dx = freq_res)
theta_power = simps(psd[idx_theta], dx = freq_res)
alpha_power = simps(psd[idx_alpha], dx = freq_res)
beta_power = simps(psd[idx_beta], dx = freq_res)

print('Absolute delta power: %.3f uV^2' % delta_power)
print('Absolute theta power: %.3f uV^2' % theta_power)
print('Absolute alpha power: %.3f uV^2' % alpha_power)
print('Absolute beta power: %.3f uV^2' % beta_power)

#%%
# Relative delta power (experessed as a percentage of total power)
total_power = simps(psd, dx=freq_res)
delta_rel_power = delta_power / total_power
theta_rel_power = theta_power / total_power
alpha_rel_power = alpha_power / total_power
beta_rel_power = beta_power / total_power

print('Relative delta power: %.3f' % delta_rel_power)
print('Relative theta power: %.3f' % theta_rel_power)
print('Relative alpha power: %.3f' % alpha_rel_power)
print('Relative beta power: %.3f' % beta_rel_power)
#%%


