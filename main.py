import tkinter as tk
from tkinter import filedialog, messagebox
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from psd import analyze_psd
from cca import analyze_cca
from fft import analyze_fft
from fbcca import analyze_fbcca
from normalization import normalize_data

def load_mat_file(path):
    data = scipy.io.loadmat(path)
    return data['data']

def run_analysis():
    mat_path = mat_path_var.get()
    baseline_path = baseline_path_var.get()
    fs = int(fs_entry.get())
    fft_band = list(map(float, fft_entry.get().split('-')))
    psd_band = list(map(float, psd_entry.get().split('-')))
    cca_band = list(map(float, cca_entry.get().split('-')))
    fbcca_band = list(map(float, fbcca_entry.get().split('-')))
    x_tick_spacing = float(x_tick_entry.get())
    threshold_multiplier = float(threshold_entry.get())
    norm_method = norm_var.get()

    selected_channels = [i for i, var in enumerate(channel_vars) if var.get() == 1]
    if not selected_channels:
        messagebox.showwarning("Warning", "กรุณาเลือกอย่างน้อย 1 channel ก่อนเริ่มวิเคราะห์")
        return

    data = load_mat_file(mat_path)
    baseline_data = load_mat_file(baseline_path)

    data = normalize_data(data, norm_method)
    baseline_data = normalize_data(baseline_data, norm_method)

    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig, axs = plt.subplots(4, 1, figsize=(10, 9))
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95, hspace=0.8)

    fft_result, fft_exceed = analyze_fft(data, fs, fft_band, axs[0], x_tick_spacing, selected_channels, threshold_multiplier)
    psd_result, psd_exceed = analyze_psd(data, baseline_data, fs, psd_band, axs[1], x_tick_spacing, threshold_multiplier, selected_channels)
    cca_result, cca_exceed = analyze_cca(data, baseline_data, fs, cca_band, axs[2], threshold_multiplier, selected_channels)
    fbcca_result, fbcca_exceed = analyze_fbcca(data, baseline_data, fs, cca_band, axs[3], threshold_multiplier, selected_channels)

    text = ""
    text += f"○ FFT Exceeds Threshold: {', '.join(fft_exceed) if fft_exceed else 'None'}\n"
    for ch, freq in fft_result.items():
        text += f" - {ch}: {freq:.2f} Hz\n"

    text += f"\n○ PSD Exceeds Threshold: {', '.join(psd_exceed) if psd_exceed else 'None'}\n"
    for ch, freq in psd_result.items():
        text += f" - {ch}: {freq:.2f} Hz\n"

    text += f"\n○ CCA Exceeds Threshold: {', '.join(cca_exceed) if cca_exceed else 'None'}\n"
    for ch, freq in cca_result.items():
        text += f" - {ch}: {freq[0]:.2f} Hz (corr={freq[1]:.3f})\n"

    text += f"\n○ FBCCA Exceeds Threshold: {', '.join(fbcca_exceed) if fbcca_exceed else 'None'}\n"
    for ch, freq in fbcca_result.items():
        text += f" - {ch}: {freq[0]:.2f} Hz (corr={freq[1]:.3f})\n"

    result_label.config(text=text)

    plot_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    plot_canvas_widget = plot_canvas.get_tk_widget()
    plot_canvas_widget.pack()
    plot_canvas.draw()
    plot_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))
    

def browse_mat_file():
    filename = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    mat_path_var.set(filename)

def browse_baseline_file():
    filename = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    baseline_path_var.set(filename)

root = tk.Tk()
root.title("EEG Analysis GUI")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

control_frame = tk.Frame(main_frame)
control_frame.pack(side=tk.LEFT, anchor='n', padx=10, pady=10)

tk.Label(control_frame, text="เลือกไฟล์วิเคราะห์ .mat:").grid(row=0, column=0, sticky='e')
mat_path_var = tk.StringVar()
tk.Entry(control_frame, textvariable=mat_path_var, width=50).grid(row=0, column=1)
tk.Button(control_frame, text="เลือกไฟล์", command=browse_mat_file).grid(row=0, column=2)

tk.Label(control_frame, text="เลือกไฟล์ baseline .mat:").grid(row=1, column=0, sticky='e')
baseline_path_var = tk.StringVar()
tk.Entry(control_frame, textvariable=baseline_path_var, width=50).grid(row=1, column=1)
tk.Button(control_frame, text="เลือกไฟล์", command=browse_baseline_file).grid(row=1, column=2)

tk.Label(control_frame, text="Sampling Rate (Hz):").grid(row=2, column=0, sticky='e')
fs_entry = tk.Entry(control_frame)
fs_entry.insert(0, "2000")
fs_entry.grid(row=2, column=1)

tk.Label(control_frame, text="ช่วงวิเคราะห์ FFT (เช่น 6-10):").grid(row=3, column=0, sticky='e')
fft_entry = tk.Entry(control_frame)
fft_entry.insert(0, "6-10")
fft_entry.grid(row=3, column=1)

tk.Label(control_frame, text="ช่วงวิเคราะห์ PSD (เช่น 6-10):").grid(row=4, column=0, sticky='e')
psd_entry = tk.Entry(control_frame)
psd_entry.insert(0, "6-10")
psd_entry.grid(row=4, column=1)

tk.Label(control_frame, text="ช่วงวิเคราะห์ CCA (เช่น 6-10):").grid(row=5, column=0, sticky='e')
cca_entry = tk.Entry(control_frame)
cca_entry.insert(0, "6-10")
cca_entry.grid(row=5, column=1)

tk.Label(control_frame, text="ช่วงวิเคราะห์ FBCCA (เช่น 6-10):").grid(row=6, column=0, sticky='e')
fbcca_entry = tk.Entry(control_frame)
fbcca_entry.insert(0, "6-10")
fbcca_entry.grid(row=6, column=1)

tk.Label(control_frame, text="ความถี่ที่ห่างของแกน X (Hz):").grid(row=7, column=0, sticky='e')
x_tick_entry = tk.Entry(control_frame)
x_tick_entry.insert(0, "0.2")
x_tick_entry.grid(row=7, column=1)

tk.Label(control_frame, text="Threshold Multiplier:").grid(row=8, column=0, sticky='e')
threshold_entry = tk.Entry(control_frame)
threshold_entry.insert(0, "1.5")
threshold_entry.grid(row=8, column=1)

# Normalization dropdown
tk.Label(control_frame, text="Normalization:").grid(row=9, column=0, sticky='e')
norm_var = tk.StringVar(value="raw")
norm_menu = tk.OptionMenu(control_frame, norm_var, "raw", "min-max", "z-score")
norm_menu.grid(row=9, column=1, sticky='w')

# Channel selection
tk.Label(control_frame, text="เลือก Channels (สูงสุด 20):").grid(row=10, column=0, sticky='ne', pady=(10, 0))
channel_frame = tk.Frame(control_frame)
channel_frame.grid(row=10, column=1, columnspan=2, sticky='w', pady=(10, 0))
channel_vars = [tk.IntVar(value=1 if i == 0 else 0) for i in range(20)]
for i in range(20):
    cb = tk.Checkbutton(channel_frame, text=f"Ch {i+1}", variable=channel_vars[i])
    cb.grid(row=i//5, column=i%5, sticky='w')

tk.Button(control_frame, text="เริ่มวิเคราะห์", command=run_analysis).grid(row=11, column=1, pady=10)

result_label = tk.Label(control_frame, justify="left", anchor="w")
result_label.grid(row=12, column=0, columnspan=3, sticky="w")

canvas_frame = tk.Frame(main_frame)
canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame)
scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

plot_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=plot_frame, anchor="nw")

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

plot_frame.bind("<Configure>", on_configure)

root.mainloop()
