import numpy as np
from matplotlib.ticker import MultipleLocator

def analyze_fft(data, fs, freq_range, ax=None, x_tick_spacing=1.0, selected_channels=None, threshold_multiplier=1.5):
    n = data.shape[0]
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_result = {}
    fft_exceed = []

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_in_range = freqs[mask]
    power_all = np.abs(np.fft.rfft(data, axis=0))**2
    power_in_range = power_all[mask]

    # สร้าง bin centers: 6.0, 6.2, ..., 10.0 ตาม x_tick_spacing
    bin_centers = np.arange(freq_range[0], freq_range[1] + x_tick_spacing, x_tick_spacing)

    if ax is not None:
        ax.set_title("FFT Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.grid(True)
        ax.xaxis.set_major_locator(MultipleLocator(base=x_tick_spacing))
        ax.tick_params(axis='x', rotation=45)

    for ch in selected_channels:
        ch_power = power_in_range[:, ch]

        bin_power = []
        for center in bin_centers:
            bin_mask = (freqs_in_range >= center - x_tick_spacing/2) & (freqs_in_range < center + x_tick_spacing/2)
            if np.any(bin_mask):
                avg = np.mean(ch_power[bin_mask])
                bin_power.append(avg)
            else:
                bin_power.append(0)

        bin_power = np.array(bin_power)

        best_idx = np.argmax(bin_power)
        best_freq = bin_centers[best_idx]
        fft_result[f'Channel {ch+1}'] = best_freq

        # Threshold checking
        mean_power = np.mean(bin_power)
        std_power = np.std(bin_power)
        threshold = mean_power + threshold_multiplier * std_power
        if bin_power[best_idx] > threshold:
            fft_exceed.append(f'Channel {ch+1}')

        if ax is not None:
            ax.plot(bin_centers, bin_power, label=f'Channel {ch+1}')

    if ax is not None:
        ax.legend()

    return fft_result, fft_exceed
