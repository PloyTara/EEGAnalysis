import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def analyze_stft(data):
    eeg = data['eeg']
    fs = int(data['fs'].squeeze())
    ch_data = eeg[0, :]

    f, t, Zxx = stft(ch_data, fs=fs, nperseg=256)
    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()
