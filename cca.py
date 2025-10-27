import numpy as np
from sklearn.cross_decomposition import CCA
from matplotlib.ticker import MultipleLocator

def analyze_cca(data, baseline_data, fs, freq_range, ax=None, threshold_multiplier=1.5, selected_channels=None, x_tick_spacing=0.2):
    if selected_channels is None:
        selected_channels = [0, 1]

    # ใช้ช่วงความถี่ตาม x_tick_spacing จาก GUI
    freqs = np.arange(freq_range[0], freq_range[1] + x_tick_spacing, x_tick_spacing)
    best_corr = {}
    exceed_channels = {}

    if ax is not None:
        ax.set_title("CCA Correlation vs Frequency")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Canonical Correlation")
        ax.grid(True)
        ax.xaxis.set_major_locator(MultipleLocator(base=x_tick_spacing))
        ax.tick_params(axis='x', rotation=45)

    for ch in selected_channels:
        signal = data[:, ch].reshape(-1, 1)
        baseline_signal = baseline_data[:, ch].reshape(-1, 1)

        corr_list = []
        for f in freqs:
            ref = np.hstack([
                np.sin(2 * np.pi * f * np.arange(len(signal)) / fs).reshape(-1, 1),
                np.cos(2 * np.pi * f * np.arange(len(signal)) / fs).reshape(-1, 1)
            ])

            cca = CCA(n_components=1)
            cca.fit(ref, signal)
            X_c, Y_c = cca.transform(ref, signal)
            corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
            corr_list.append(corr)

        # ค่าที่ดีที่สุด
        best_idx = np.argmax(corr_list)
        best_freq = freqs[best_idx]
        best_val = corr_list[best_idx]
        best_corr[f'Channel {ch+1}'] = (best_freq, best_val)

        # เปรียบเทียบกับ threshold
        baseline_corr = np.mean(corr_list)
        if best_val > threshold_multiplier * baseline_corr:
            exceed_channels[f'Channel {ch+1}'] = True
        else:
            exceed_channels[f'Channel {ch+1}'] = False

        # วาดกราฟ
        if ax is not None:
            ax.plot(freqs, corr_list, label=f'Channel {ch+1}') #marker='o', 

    if ax is not None:
        ax.legend()

    # ส่งคืนค่าความถี่ที่ดีที่สุด และรายชื่อ channel ที่เกิน threshold
    return best_corr, [ch for ch, exceeded in exceed_channels.items() if exceeded]
