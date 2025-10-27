import numpy as np
from scipy.signal import welch
from matplotlib.ticker import MultipleLocator


def analyze_psd(data, baseline_data, fs, freq_range, ax=None, x_tick_spacing=1.0, threshold_multiplier=1.5, selected_channels=None):
    if selected_channels is None:
        selected_channels = [0, 1]

    # กำหนดช่วงความถี่ที่จะวิเคราะห์ตามระยะห่างแกน X
    # target_freqs = np.arange(freq_range[0], freq_range[1] + x_tick_spacing, x_tick_spacing)
    # target_freqs = target_freqs[target_freqs <= freq_range[1]]

    num_points = int(round((freq_range[1] - freq_range[0]) / x_tick_spacing)) + 1
    target_freqs = np.round(np.linspace(freq_range[0], freq_range[1], num=num_points), 10)


    best_freqs = {}
    exceed_channels = []

    if ax is not None:
        ax.set_title('Power Spectral Density')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power/Frequency (V²/Hz)')
        ax.grid(True)
        ax.xaxis.set_major_locator(MultipleLocator(base=x_tick_spacing))
        ax.tick_params(axis='x', rotation=45)

    # ประมวลผลแต่ละ channel ที่เลือก
    for ch in selected_channels:
        ch_data = data[:, ch]
        b_data = baseline_data[:, ch]

        freqs, psd = welch(ch_data, fs=fs, nperseg=1024)
        _, b_psd = welch(b_data, fs=fs, nperseg=1024)

        # คำนวณค่า psd เฉพาะ target_freqs ที่ต้องการ
        psd_interp = np.interp(target_freqs, freqs, psd)
        b_psd_interp = np.interp(target_freqs, freqs, b_psd)

        # ค่าความถี่ที่มีพลังงานสูงสุดในช่วงนี้
        max_idx = np.argmax(psd_interp)
        best_freq = target_freqs[max_idx]
        best_freqs[f'Channel {ch+1}'] = best_freq

        # ตรวจสอบ threshold
        if np.max(psd_interp) > threshold_multiplier * np.max(b_psd_interp):
            exceed_channels.append(f'Channel {ch+1}')

        # วาดกราฟ
        if ax is not None:
            ax.plot(target_freqs, psd_interp, label=f'Channel {ch+1}') #marker='o', 

    if ax is not None:
        ax.legend()

    return best_freqs, exceed_channels
