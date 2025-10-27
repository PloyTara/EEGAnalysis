import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

def analyze_dynamic_psd(data, baseline_data, fs, freq_band, ax, x_tick_spacing, threshold_multiplier, selected_channels):
    results = {}
    exceed_channels = []

    window_size = fs * 2
    step_size = fs // 2

    for ch in selected_channels:
        ch_data = data[ch, :]
        base_data = baseline_data[ch, :]

        # ✅ Zero-pad ถ้าข้อมูลไม่พอ
        if len(ch_data) < window_size:
            pad_width = window_size - len(ch_data)
            ch_data = np.pad(ch_data, (0, pad_width), 'constant')

        if len(base_data) < window_size:
            pad_width = window_size - len(base_data)
            base_data = np.pad(base_data, (0, pad_width), 'constant')

        times, powers = [], []
        freqs = None

        # ✅ Dynamic PSD
        for start in range(0, len(ch_data) - window_size + 1, step_size):
            segment = ch_data[start:start+window_size]
            f, Pxx = welch(segment, fs=fs, nperseg=min(window_size, len(segment)))
            if freqs is None:
                freqs = f
            powers.append(Pxx)
            times.append(start / fs)

        if freqs is None or len(powers) == 0:
            continue

        powers = np.array(powers).T  # freq x time

        # ✅ ตรวจสอบความถี่ที่อยู่ในช่วง
        band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        if not np.any(band_mask):
            continue

        band_powers = powers[band_mask, :]
        mean_power = band_powers.mean(axis=1)

        # ✅ คำนวณ baseline
        f_base, Pxx_base = welch(base_data[:window_size], fs=fs, nperseg=min(window_size, len(base_data)))
        base_mask = (f_base >= freq_band[0]) & (f_base <= freq_band[1])

        if np.any(base_mask):
            base_power = Pxx_base[base_mask].mean()
        else:
            base_power = 0  # ถ้าไม่มีค่าช่วงนี้ ให้ baseline = 0

        # ✅ หาความถี่ที่มี power สูงสุด
        max_idx = np.argmax(mean_power)
        max_freq = freqs[band_mask][max_idx]

        if base_power > 0 and mean_power[max_idx] > base_power * threshold_multiplier:
            exceed_channels.append(f"Ch{ch+1}")

        results[f"Ch{ch+1}"] = max_freq

        if powers.size == 0 or len(times) == 0 or freqs is None:
            ax.set_title(f"Ch{ch+1} Dynamic PSD (No Data)")
            continue

        # ✅ Plot Dynamic PSD
        powers = np.array(powers).T
        pcm = ax.pcolormesh(times, freqs, powers, shading='gouraud')
        fig = ax.get_figure()
        fig.colorbar(pcm, ax=ax, label='Power')
        ax.set_ylabel('Freq [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title(f'Ch{ch+1} Dynamic PSD')

    return results, exceed_channels
