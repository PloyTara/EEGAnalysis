import numpy as np
import matplotlib.pyplot as plt
import pywt

def analyze_wavelet(data):
    eeg = data['eeg']
    ch_data = eeg[0, :]
    
    scales = np.arange(1, 128)
    coef, freqs = pywt.cwt(ch_data, scales, 'morl')

    plt.figure()
    plt.imshow(np.abs(coef), extent=[0, ch_data.shape[0], 1, 128], cmap='jet', aspect='auto',
               vmax=np.percentile(np.abs(coef), 99))
    plt.gca().invert_yaxis()
    plt.title('Wavelet Transform (CWT)')
    plt.xlabel('Samples')
    plt.ylabel('Scales')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.show()
