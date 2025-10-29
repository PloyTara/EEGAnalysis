import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def analyzed_dynamic_psd(data, fs, window_size=1.0, step_size=0.25, freq_min=2,freq_max=50, selected_channels=None, mode='average', nperseg=2048):
    samples_per_win = int(window_size * fs)
    sample_step = int(step_size * fs)
    n_windows = int((data.shape[0] - samples_per_win) / sample_step) + 1

    if selected_channels is None:
        selected_channels = list(range(data.shape[1]))

    if mode == 'average':
        data_proc = np.mean(data[:, selected_channels], axis=1, keepdims=True)
        channel_labels = ['Average']
    else:
        data_proc = data[:, selected_channels]
        channel_labels = [f"Ch {ch+1}" for ch in selected_channels]

    psd_all = []
    max_freqs = []

    for ch_idx in range(data_proc.shape[1]):
        psd_matrix = []
        time_points = []

        for i in range(n_windows):
            start = i * sample_step
            end = start + samples_per_win
            segment = data[start:end, ch_idx]
            freqs, psd = welch(segment, fs=fs, nperseg=min(nperseg, len(segment)))

            freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
            psd_matrix.append(10 * np.log10(psd[freq_mask]))
            time_points.append(start / fs)

        psd_matrix = np.array(psd_matrix).T
        psd_all.append((freqs[freq_mask], time_points, psd_matrix))

        max_power = psd_matrix.mean(axis=1)
        max_freq = freqs[freq_mask][np.argmax(max_power)]
        max_freqs.append((channel_labels[ch_idx], max_freq))
    
    return psd_all, max_freqs
