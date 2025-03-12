import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

def get_max_supported_sample_rate(device_index):
    test_sample_rates = [192000, 96000, 48000, 44100, 22050, 16000, 8000]
    for rate in test_sample_rates:
        try:
            sd.check_input_settings(device=device_index, samplerate=rate)
            return rate
        except Exception:
            continue
    return None

def select_input_device():
    try:
        default_device = sd.default.device[0] if sd.default.device and sd.default.device[0] is not None else None
    except Exception:
        default_device = None
        
    if default_device is not None:
        try:
            dev_info = sd.query_devices(default_device)
            sr = dev_info['default_samplerate']
            sd.check_input_settings(device=default_device, samplerate=sr)
            print(f"Using default input device: {default_device} - {dev_info['name']}")
            return default_device
        except Exception as e:
            print(f"Default input device {default_device} not working: {e}")
    
    print("Scanning for working input devices...")
    devices = sd.query_devices()
    working_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            try:
                sr = dev['default_samplerate']
                sd.check_input_settings(device=i, samplerate=sr)
                max_rate = get_max_supported_sample_rate(i)
                working_devices.append((i, dev, max_rate))
            except Exception:
                continue
                
    if not working_devices:
        raise Exception("No working input devices found!")
        
    print("Available working input devices:")
    for idx, dev, max_rate in working_devices:
        print(f"  Index: {idx}, Name: {dev['name']}, Default Samplerate: {dev['default_samplerate']}, Max Supported Rate: {max_rate}, Max Channels: {dev['max_input_channels']}")
    chosen = input("Enter the device index to use (or press Enter to use the first available): ")
    try:
        if chosen.strip() == "":
            return working_devices[0][0]
        chosen = int(chosen)
        if any(idx == chosen for idx, _, _ in working_devices):
            return chosen
        else:
            print("Invalid selection, using first available device.")
            return working_devices[0][0]
    except Exception:
        print("Invalid input, using first available device.")
        return working_devices[0][0]

def record_audio(duration=5, samplerate=16000, channels=1, device=None):
    try:
        if device is None:
            device = select_input_device()
        else:
            dev_info = sd.query_devices(device)
            if dev_info['max_input_channels'] < 1:
                raise Exception(f"Selected device {device} does not support input.")
            sr = dev_info['default_samplerate']
            sd.check_input_settings(device=device, samplerate=sr)
        chosen_info = sd.query_devices(device)
        print(f"Using input device: {device} - {chosen_info['name']}")
        
        # Try using the desired samplerate first; if not, fall back to device's default
        try:
            sd.check_input_settings(device=device, samplerate=samplerate)
            used_samplerate = samplerate
        except Exception:
            default_sr = chosen_info['default_samplerate']
            sd.check_input_settings(device=device, samplerate=default_sr)
            print(f"Desired samplerate {samplerate} not supported, falling back to device default {default_sr}")
            used_samplerate = default_sr
            
        print(f"Recording at {used_samplerate} Hz for {duration} seconds...")
        recording = sd.rec(int(duration * used_samplerate), samplerate=used_samplerate,
                           channels=channels, dtype='float32', device=device)
        sd.wait()
        return np.squeeze(recording), used_samplerate
    except Exception as e:
        print("Error recording audio:", e)
        return None, None

def save_wav(filename, data, samplerate):
    try:
        if np.max(np.abs(data)) > 0:
            scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        else:
            scaled = np.int16(data)
        write(filename, samplerate, scaled)
        print(f"Audio saved as {filename}")
    except Exception as e:
        print("Error saving WAV file:", e)
