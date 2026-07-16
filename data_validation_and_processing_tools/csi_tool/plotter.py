import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import platform
import subprocess
from datetime import datetime
import pandas as pd
from scipy.interpolate import PchipInterpolator

"""
================================================================================
MODULE: plotter.py - CSI Data Visualization (Multi-Device Amplitude & Spectrogram)
================================================================================

INPUT DATA STRUCTURE:
================================================================================
The module now receives a `devices_list` in `plot_csi_data`. Each element is a dict:
  {
      'name': str,              # Tên hiển thị của thiết bị (VD: 'ESP32_Tx', 'ASUS_Rx1')
      'dev_type': str,          # Loại thiết bị: 'esp' hoặc 'asus'
      'csi_matrices': list      # Danh sách dict chứa data rx_index, timestamp, amplitude...
  }

LAYOUT:
- Cột (Columns): Tương ứng với số lượng thiết bị truyền vào (VD: 3 thiết bị -> 3 cột)
- Hàng (Rows): Tương ứng với số lượng antenna lớn nhất trong các thiết bị (VD: có Asus -> 4 hàng)
- Các thiết bị ít antenna hơn (như ESP32) sẽ tự động ẩn các đồ thị ở hàng dưới.
================================================================================
"""

IMAGE_DIR = "image"
basename_file = None

def _ensure_image_dir():
    """Tạo thư mục lưu ảnh nếu chưa tồn tại."""
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

def _get_filepath(filename):
    """Tạo đường dẫn đầy đủ lưu vào thư mục ảnh."""
    _ensure_image_dir()
    return os.path.join(IMAGE_DIR, filename)

def _open_image_with_os(filepath):
    """Mở ảnh bằng trình xem ảnh mặc định của hệ điều hành (tiến trình độc lập)."""
    try:
        abs_filepath = os.path.abspath(filepath)
        if platform.system() == 'Windows':
            os.startfile(abs_filepath)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.Popen(['open', abs_filepath])
        else:  # Linux
            subprocess.Popen(['xdg-open', abs_filepath])
    except Exception as e:
        print(f"[Cảnh báo] Không thể tự động mở ảnh: {e}")
def _pchip_interpolate(signal):
    signal = np.asarray(signal, dtype=float).copy()
    signal[signal == 0] = np.nan
    valid = ~np.isnan(signal)

    if np.sum(valid) < 2:
        return signal

    x = np.arange(len(signal))
    f = PchipInterpolator(
        x[valid],
        signal[valid],
        extrapolate=True,
    )

    signal[~valid] = f(x[~valid])

    return signal

def _hampel_filter(data, window_size=5, threshold=3.0):
    """Hampel filter - bộ lọc robust dùng median để loại bỏ outliers."""
    filtered = np.copy(data)
    half_win = window_size // 2
    
    for i in range(half_win, len(data) - half_win):
        window = data[i - half_win : i + half_win + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        
        if mad > 0:
            deviation = np.abs(data[i] - median) / mad
            if deviation > threshold:
                filtered[i] = median
    
    return filtered

def _butterworth_filter(data, order=4, cutoff=0.1):
    """Butterworth low-pass filter."""
    b, a = signal.butter(order, cutoff, btype='low')
    filtered = signal.filtfilt(b, a, data)
    return filtered

def _apply_preprocessing(amplitude):
    """
    Áp dụng tiền xử lý CSI:
    - Packet mất (=0) -> NaN
    - Interpolate NaN
    - Normalize theo từng cột
    - Hampel filter
    - Butterworth low-pass
    """

    amplitude = amplitude.astype(float).copy()

    # ==========================================================
    # Packet mất (=0) -> NaN
    # ==========================================================
    amplitude[amplitude == 0] = np.nan

    # ==========================================================
    # 1D
    # ==========================================================
    if amplitude.ndim == 1:

        if np.isnan(amplitude).all():
            return amplitude

        # ------------------------------------------------------
        # Interpolate NaN
        # ------------------------------------------------------
        amplitude = (
            pd.Series(amplitude)
            .interpolate(method='linear')
            .bfill()
            .ffill()
            .values
        )

        # ------------------------------------------------------
        # Normalize
        # ------------------------------------------------------
        amp_min = np.min(amplitude)
        amp_max = np.max(amplitude)

        if amp_max > amp_min:
            normalized = (amplitude - amp_min) / (amp_max - amp_min)
        else:
            normalized = np.zeros_like(amplitude)

        # ------------------------------------------------------
        # Hampel
        # ------------------------------------------------------
        hampel_filtered = _hampel_filter(
            normalized,
            window_size=11,
            threshold=3.0
        )

        # ------------------------------------------------------
        # Butterworth
        # ------------------------------------------------------
        butterworth_filtered = _butterworth_filter(
            hampel_filtered,
            order=4,
            cutoff=0.25
        )

        # ------------------------------------------------------
        # Restore scale
        # ------------------------------------------------------
        restored = butterworth_filtered

        if amp_max > amp_min:
            restored = (
                butterworth_filtered
                * (amp_max - amp_min)
                + amp_min
            )

        return restored

    # ==========================================================
    # 2D
    # ==========================================================
    butterworth_filtered = np.copy(amplitude)

    for i in range(amplitude.shape[1]):

        col = amplitude[:, i]

        if np.isnan(col).all():
            continue

        # ------------------------------------------------------
        # Interpolate NaN
        # ------------------------------------------------------
        col = (
            pd.Series(col)
            .interpolate(method='linear')
            .bfill()
            .ffill()
            .values
        )

        # ------------------------------------------------------
        # Normalize theo từng subcarrier
        # ------------------------------------------------------
        amp_min = np.min(col)
        amp_max = np.max(col)

        if amp_max > amp_min:
            normalized = (col - amp_min) / (amp_max - amp_min)
        else:
            normalized = np.zeros_like(col)

        # ------------------------------------------------------
        # Hampel
        # ------------------------------------------------------
        hampel_filtered = _hampel_filter(
            normalized,
            window_size=11,
            threshold=3.0
        )

        # ------------------------------------------------------
        # Butterworth
        # ------------------------------------------------------
        filtered = _butterworth_filter(
            hampel_filtered,
            order=4,
            cutoff=0.25
        )

        # ------------------------------------------------------
        # Restore scale
        # ------------------------------------------------------
        if amp_max > amp_min:
            filtered = (
                filtered
                * (amp_max - amp_min)
                + amp_min
            )

        butterworth_filtered[:, i] = filtered

    return butterworth_filtered

def plot_csi_data(base_name, devices_list, plot_preprocess: bool = False, plot_single_sub=None, enCutSubUnused: bool = False):
    """
    Định tuyến dữ liệu để vẽ (Amplitude & Spectrogram) cho danh sách các thiết bị.
    Layout: N Hàng (số lượng antenna tối đa) x M Cột (số lượng thiết bị).
    """
    global basename_file
    basename_file = base_name

    if not devices_list:
        print("[Lỗi] Không có dữ liệu danh sách thiết bị để vẽ.")
        return
    
    processed_devices = []
    max_antennas = 1
    
    # 1. Trích xuất và cấu trúc hóa dữ liệu cho từng thiết bị
    for dev_info in devices_list:
        dev_name = dev_info.get('name', 'Unknown_Device')
        dev_type = dev_info.get('dev_type', 'esp')
        csi_matrices = dev_info.get('csi_matrices', None)
        
        if not csi_matrices or csi_matrices[0] is None:
            print(f"[Cảnh báo] Bỏ qua thiết bị {dev_name} vì không có dữ liệu CSI.")
            continue
            
        rx_data = csi_matrices[0]
        num_antennas = rx_data.get('num_antennas', 1)
        if num_antennas > max_antennas:
            max_antennas = num_antennas
            
        if dev_type == 'esp':
            amplitude = np.array(rx_data['amplitude'], dtype=float)
        else:  # asus
            amplitude = [np.array(arr, dtype=float) for arr in rx_data['amplitude']]
            
        # Timestamp logic
        timestamp = rx_data.get('timestamp', np.arange(len(amplitude) if isinstance(amplitude, np.ndarray) else len(amplitude[0])))
        time_axis = np.arange(len(timestamp))
        
        actual_plot_sub = plot_single_sub
        is_sub_disabled = False
        
        # --- ÁP DỤNG CẮT BỎ SUBCARRIER CHO ASUS ---
        if dev_type == 'asus' and enCutSubUnused:
            for ant in range(num_antennas):
                amplitude[ant] = np.delete(amplitude[ant], np.s_[28:37], axis=1)
                
            if plot_single_sub is not None:
                if 28 <= plot_single_sub <= 36:
                    is_sub_disabled = True
                    print(f"[{dev_name}] Cảnh báo: Subcarrier {plot_single_sub} nằm trong dải đã cắt bỏ (28-36).")
                elif plot_single_sub > 36:
                    actual_plot_sub = plot_single_sub - 9
                    
        processed_devices.append({
            'name': dev_name,
            'type': dev_type,
            'num_antennas': num_antennas,
            'amplitude': amplitude,
            'timestamp': timestamp,
            'time_axis': time_axis,
            'actual_plot_sub': actual_plot_sub,
            'is_sub_disabled': is_sub_disabled
        })

    if not processed_devices:
        print("[Lỗi] Không có thiết bị hợp lệ nào để vẽ sau khi xử lý.")
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. Định tuyến theo chế độ (Single Subcarrier hoặc Full Subcarrier)
    if plot_single_sub is not None:
        plot_multi_single_sub_amplitude(processed_devices, max_antennas, plot_single_sub, plot_preprocess, timestamp_str)
        # plot_multi_single_sub_spectrogram(processed_devices, max_antennas, plot_single_sub, plot_preprocess, timestamp_str)
    else:
        plot_multi_full_heatmap(processed_devices, max_antennas, plot_preprocess, timestamp_str)
        # plot_multi_full_spectrogram(processed_devices, max_antennas, plot_preprocess, timestamp_str)


# ==============================================================================
# 1. HÀM VẼ CHO 1 SUBCARRIER (MULTI-DEVICE)
# ==============================================================================

def plot_multi_single_sub_amplitude(devices, max_antennas, sub_idx_label, plot_preprocess, timestamp_str):
    """Vẽ biểu đồ biên độ Line Plot cho 1 subcarrier trên nhiều thiết bị (Cột: Device, Hàng: Antenna)."""
    num_devices = len(devices)
    fig, axes = plt.subplots(max_antennas, num_devices, figsize=(6 * num_devices, 4 * max_antennas), squeeze=False)
    
    for col, dev in enumerate(devices):
        dev_name = dev['name']
        dev_type = dev['type']
        num_ant = dev['num_antennas']
        time_axis = dev['time_axis']
        amplitude = dev['amplitude']
        actual_sub = dev['actual_plot_sub']
        is_disabled = dev['is_sub_disabled']
        
        for row in range(max_antennas):
            ax = axes[row, col]
            
            # Ẩn đồ thị nếu ăng-ten vượt quá cấu hình thiết bị
            if row >= num_ant:
                ax.set_visible(False)
                continue
                
            if is_disabled:
                ax.text(0.5, 0.5, f"Subcarrier {sub_idx_label} vô hiệu hóa", ha='center', va='center')
                ax.set_title(f'{dev_name} - Ant {row} - Sub {sub_idx_label}')
                continue
            
            # Lấy data tương ứng
            if dev_type == 'esp':
                amp_raw = amplitude[:, actual_sub]
            else:  # asus
                amp_raw = amplitude[row][:, actual_sub]
                
            if np.isnan(amp_raw).all():
                ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
                continue

            # ------------------------------------------------------
            # PCHIP interpolation
            # ------------------------------------------------------
            if plot_preprocess:
                print(f"================ Plot PCHIP interpolation =====================")
                amp_raw = _pchip_interpolate(amp_raw)
            
            ax.plot(time_axis, amp_raw, label='Raw', linewidth=1.5, alpha=0.7)
            
            if plot_preprocess:
                amp_processed = _apply_preprocessing(amp_raw.reshape(-1, 1)).flatten()
                ax.plot(time_axis, amp_processed, label='Preprocessed', linewidth=2.0, alpha=0.9)
            
            # Định dạng UI cho subplot
            # ------------------------------------------------------
            # Fixed Y-axis
            # ------------------------------------------------------
            if dev_type == "esp":
                ax.set_ylim(0, 50)
            else:
                ax.set_ylim(0, 5000)
                
            ax.set_title(f'[{dev_name}] - Ant {row} - Sub {sub_idx_label}')
            if row == num_ant - 1:
                ax.set_xlabel('Time (packets)')
            if col == 0:
                ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"csi_single_amp_{basename_file}_sub{sub_idx_label}_{timestamp_str}.png"
    filepath = _get_filepath(filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"[Thông báo] Đã lưu Amplitude Plot (Multi) vào file: {filepath}")
    _open_image_with_os(filepath)
    plt.close(fig)


def plot_multi_single_sub_spectrogram(devices, max_antennas, sub_idx_label, plot_preprocess, timestamp_str):
    """Vẽ STFT Spectrogram cho 1 subcarrier trên nhiều thiết bị (Cột: Device, Hàng: Antenna)."""
    num_devices = len(devices)
    fig, axes = plt.subplots(max_antennas, num_devices, figsize=(6 * num_devices, 5 * max_antennas), squeeze=False)
    
    for col, dev in enumerate(devices):
        dev_name = dev['name']
        dev_type = dev['type']
        num_ant = dev['num_antennas']
        timestamp = dev['timestamp']
        amplitude = dev['amplitude']
        actual_sub = dev['actual_plot_sub']
        is_disabled = dev['is_sub_disabled']
        
        # Tính tần số Fs (Hz) riêng cho từng thiết bị
        fs_hz = 1.0
        if len(timestamp) > 1 and not np.array_equal(timestamp, np.arange(len(timestamp))):
            duration_sec = (timestamp[-1] - timestamp[0]) / 1000000.0
            if duration_sec > 0:
                fs_hz = len(timestamp) / duration_sec
                
        for row in range(max_antennas):
            ax_spec = axes[row, col]
            
            if row >= num_ant:
                ax_spec.set_visible(False)
                continue
                
            if is_disabled:
                ax_spec.text(0.5, 0.5, f"Subcarrier vô hiệu hóa", ha='center', va='center')
                ax_spec.set_title(f'{dev_name} - Ant {row} - Sub {sub_idx_label}')
                continue
                
            if dev_type == 'esp':
                amp_raw = amplitude[:, actual_sub]
            else:
                amp_raw = amplitude[row][:, actual_sub]
                
            if np.isnan(amp_raw).all():
                ax_spec.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
                continue

            if plot_preprocess:
                amp_raw = _pchip_interpolate(amp_raw)
                
            if plot_preprocess:
                amp_raw = _apply_preprocessing(amp_raw.reshape(-1, 1)).flatten()
                
            signal_1d = amp_raw - np.nanmean(amp_raw)
            n_samples = len(signal_1d)
            if n_samples < 16:
                ax_spec.text(0.5, 0.5, "Data quá ngắn cho STFT", ha='center', va='center')
                continue
                
            nperseg = min(256, max(16, n_samples // 8))
            noverlap = int(nperseg * 0.9)
            
            f, t, Sxx = signal.spectrogram(signal_1d, fs=fs_hz, nperseg=nperseg, noverlap=noverlap)
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)
            
            vmin = np.nanpercentile(Sxx_dB, 2)
            vmax = np.nanpercentile(Sxx_dB, 98)
            
            im_spec = ax_spec.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
            ax_spec.set_title(f'[{dev_name}] - Ant {row} - Sub {sub_idx_label} Spectrogram')
            if row == num_ant - 1:
                ax_spec.set_xlabel('Time (Seconds)')
            if col == 0:
                ax_spec.set_ylabel('Frequency (Hz)')
            plt.colorbar(im_spec, ax=ax_spec, label='Power (dB)')

    plt.tight_layout()
    filename = f"csi_single_spec_{basename_file}_sub{sub_idx_label}_{timestamp_str}.png"
    filepath = _get_filepath(filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"[Thông báo] Đã lưu Spectrogram 1 Sub (Multi) vào file: {filepath}")
    _open_image_with_os(filepath)
    plt.close(fig)


# ==============================================================================
# 2. HÀM VẼ CHO TẤT CẢ SUBCARRIER (MULTI-DEVICE)
# ==============================================================================

def plot_multi_full_heatmap(devices, max_antennas, plot_preprocess, timestamp_str):
    """Vẽ Heatmap liền mạch toàn bộ subcarrier cho nhiều thiết bị."""
    num_devices = len(devices)
    fig, axes = plt.subplots(max_antennas, num_devices, figsize=(6 * num_devices, 5 * max_antennas), squeeze=False)
    
    for col, dev in enumerate(devices):
        dev_name = dev['name']
        dev_type = dev['type']
        num_ant = dev['num_antennas']
        amplitude = dev['amplitude']
        
        for row in range(max_antennas):
            ax_amp = axes[row, col]
            
            if row >= num_ant:
                ax_amp.set_visible(False)
                continue
                
            if dev_type == 'esp':
                amp_data = amplitude
            else:
                amp_data = amplitude[row]
                
            if plot_preprocess:
                amp_data = _apply_preprocessing(amp_data)
                
            vmin = np.nanpercentile(amp_data, 2)
            vmax = np.nanpercentile(amp_data, 98)
            
            im_amp = ax_amp.imshow(amp_data.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            ax_amp.set_title(f'[{dev_name}] - Ant {row} - Heatmap')
            if row == num_ant - 1:
                ax_amp.set_xlabel('Time (packets)')
            if col == 0:
                ax_amp.set_ylabel('Subcarrier Index (Adjusted)')
            plt.colorbar(im_amp, ax=ax_amp, label='Amplitude')

    plt.tight_layout()
    filename = f"csi_full_heatmap_{basename_file}_{timestamp_str}.png"
    filepath = _get_filepath(filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"[Thông báo] Đã lưu Heatmap Full (Multi) vào file: {filepath}")
    _open_image_with_os(filepath)
    plt.close(fig)


def plot_multi_full_spectrogram(devices, max_antennas, plot_preprocess, timestamp_str):
    """Vẽ Spectrogram kết hợp STFT toàn dải subcarrier cho nhiều thiết bị."""
    num_devices = len(devices)
    fig, axes = plt.subplots(max_antennas, num_devices, figsize=(6 * num_devices, 5 * max_antennas), squeeze=False)
    
    for col, dev in enumerate(devices):
        dev_name = dev['name']
        dev_type = dev['type']
        num_ant = dev['num_antennas']
        timestamp = dev['timestamp']
        amplitude = dev['amplitude']
        
        fs_hz = 1.0
        if len(timestamp) > 1 and not np.array_equal(timestamp, np.arange(len(timestamp))):
            duration_sec = (timestamp[-1] - timestamp[0]) / 1000000.0
            if duration_sec > 0:
                fs_hz = len(timestamp) / duration_sec
                
        for row in range(max_antennas):
            ax_spec = axes[row, col]
            
            if row >= num_ant:
                ax_spec.set_visible(False)
                continue
                
            if dev_type == 'esp':
                amp_data = amplitude
            else:
                amp_data = amplitude[row]
                
            if plot_preprocess:
                amp_data = _apply_preprocessing(amp_data)
                
            # Gộp không gian để phân tích toàn dải
            amp_dynamic = amp_data - np.nanmean(amp_data, axis=0)
            signal_1d = np.nanmean(amp_dynamic, axis=1)
            
            n_samples = len(signal_1d)
            if n_samples < 16:
                ax_spec.text(0.5, 0.5, "Data quá ngắn cho STFT", ha='center', va='center')
                continue
                
            nperseg = min(256, max(16, n_samples // 8))
            noverlap = int(nperseg * 0.9)
            
            f, t, Sxx = signal.spectrogram(signal_1d, fs=fs_hz, nperseg=nperseg, noverlap=noverlap)
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)
            
            vmin = np.nanpercentile(Sxx_dB, 2)
            vmax = np.nanpercentile(Sxx_dB, 98)
            
            im_spec = ax_spec.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
            ax_spec.set_title(f'[{dev_name}] - Ant {row} - Full Band Spec')
            if row == num_ant - 1:
                ax_spec.set_xlabel('Time (Seconds)')
            if col == 0:
                ax_spec.set_ylabel('Frequency (Hz)')
            plt.colorbar(im_spec, ax=ax_spec, label='Power (dB)')

    plt.tight_layout()
    filename = f"csi_full_spec_{basename_file}_{timestamp_str}.png"
    filepath = _get_filepath(filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"[Thông báo] Đã lưu Spectrogram Full (Multi) vào file: {filepath}")
    _open_image_with_os(filepath)
    plt.close(fig)