# fbcca.py
import numpy as np
from sklearn.cross_decomposition import CCA
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, lfilter

def filter_bank(signal, fs):
    bands = [(6, 90), (6, 50), (6, 30), (6, 20)]
    filtered = []
    for low, high in bands:
        b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
        filtered.append(lfilter(b, a, signal, axis=0))
    return filtered

def analyze_fbcca(data, baseline_data, fs, freq_range, ax=None, threshold_multiplier=1.5, selected_channels=None, x_tick_spacing=0.2):
    if selected_channels is None:
        selected_channels = [0, 1]

    freqs = np.arange(freq_range[0], freq_range[1] + x_tick_spacing, x_tick_spacing)
    best_corr = {}
    exceed_channels = {}

    if ax is not None:
        ax.set_title("FBCCA Correlation vs Frequency")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Sum of Correlations")
        ax.grid(True)
        ax.xaxis.set_major_locator(MultipleLocator(base=x_tick_spacing))
        ax.tick_params(axis='x', rotation=45)

    for ch in selected_channels:
        signal = data[:, ch].reshape(-1, 1)
        baseline_signal = baseline_data[:, ch].reshape(-1, 1)
        n_samples = signal.shape[0]
        t = np.arange(n_samples) / fs

        filtered_signals = filter_bank(signal, fs)
        corr_list = []

        for f in freqs:
            ref = np.hstack([
                np.sin(2 * np.pi * f * t).reshape(-1, 1),
                np.cos(2 * np.pi * f * t).reshape(-1, 1)
            ])

            score_f = 0
            for fb in filtered_signals:
                cca = CCA(n_components=1)
                try:
                    cca.fit(ref, fb)
                    X_c, Y_c = cca.transform(ref, fb)
                    score_f += np.corrcoef(X_c.T, Y_c.T)[0, 1]
                except:
                    score_f += 0  # ในกรณีที่ CCA ไม่สามารถคำนวณได้

            corr_list.append(score_f)

        best_idx = np.argmax(corr_list)
        best_freq = freqs[best_idx]
        best_val = corr_list[best_idx]
        best_corr[f'Channel {ch+1}'] = (best_freq, best_val)

        baseline_corr = np.mean(corr_list)
        if best_val > threshold_multiplier * baseline_corr:
            exceed_channels[f'Channel {ch+1}'] = True
        else:
            exceed_channels[f'Channel {ch+1}'] = False

        if ax is not None:
            ax.plot(freqs, corr_list, label=f'Channel {ch+1}')

    if ax is not None:
        ax.legend()

    return best_corr, [ch for ch, exceeded in exceed_channels.items() if exceeded]
