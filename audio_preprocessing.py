import numpy as np
import librosa
import scipy.signal as signal

def preprocess_audio(audio, samplerate, normalize=True, equalize=True, noise_reduction=True):
    """
    Preprocess audio to make it more robust to microphone distance and environment.
    
    Args:
        audio: Audio array (1D numpy array)
        samplerate: Sample rate of the audio
        normalize: Whether to normalize the volume
        equalize: Whether to apply equalization
        noise_reduction: Whether to apply noise reduction
        
    Returns:
        Preprocessed audio array
    """
    # Skip processing if audio is too short
    if len(audio) < samplerate // 2:  # Less than 0.5 seconds
        return audio
        
    # Make a copy to avoid modifying the original
    processed = np.copy(audio)
    
    # 1. Remove DC offset
    processed = processed - np.mean(processed)
    
    # 2. Volume normalization
    if normalize:
        # Check if audio has content (not silence)
        if np.abs(processed).max() > 0.001:
            # Normalize to have RMS of 0.1
            rms = np.sqrt(np.mean(processed**2))
            if rms > 0:
                target_rms = 0.1
                processed = processed * (target_rms / rms)
                
                # Hard limit to prevent clipping
                max_val = np.abs(processed).max()
                if max_val > 0.95:
                    processed = processed * (0.95 / max_val)
    
    # 3. Frequency equalization to reduce microphone distance effects
    if equalize:
        # Slight high-pass filter to remove rumble and DC
        nyquist = samplerate // 2
        high_pass_freq = 80  # Hz
        b, a = signal.butter(2, high_pass_freq / nyquist, 'highpass')
        processed = signal.filtfilt(b, a, processed)
        
        # Apply a 3-band EQ to boost mid-range frequencies (where speech is)
        if len(processed) > samplerate // 4:  # At least 0.25 seconds
            try:
                # Convert to spectrogram
                D = librosa.stft(processed)
                S_db = librosa.amplitude_to_db(np.abs(D))
                
                # Enhance mid-range frequencies (300Hz-3kHz)
                low_bin = int(300 * D.shape[0] / nyquist)
                high_bin = int(3000 * D.shape[0] / nyquist)
                
                # Boost mid-range by 3dB, reduce low and high 
                S_db[:low_bin, :] -= 1.0  # Slight reduction of lows
                S_db[low_bin:high_bin, :] += 3.0  # Boost mids
                S_db[high_bin:, :] -= 1.0  # Slight reduction of highs
                
                # Convert back to audio
                S_modified = librosa.db_to_amplitude(S_db)
                phase = np.angle(D)
                D_modified = S_modified * np.exp(1j * phase)
                processed = librosa.istft(D_modified, length=len(processed))
            except Exception as e:
                print(f"Equalization failed, skipping: {e}")
    
    # 4. Simple noise reduction
    if noise_reduction and len(processed) > samplerate:
        try:
            # Estimate noise from first 0.5 seconds (assuming it starts with a bit of silence)
            noise_sample = processed[:int(samplerate * 0.5)]
            noise_profile = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)
            
            # Apply simple spectral subtraction
            D = librosa.stft(processed)
            mag = np.abs(D)
            phase = np.angle(D)
            
            # Subtract noise profile from each frame (with floor)
            mag_reduced = np.maximum(mag - noise_profile[:, np.newaxis] * 1.5, 0.01 * mag)
            
            # Reconstruct signal
            D_reduced = mag_reduced * np.exp(1j * phase)
            processed = librosa.istft(D_reduced, length=len(processed))
        except Exception as e:
            print(f"Noise reduction failed, skipping: {e}")
    
    return processed

def multi_distance_enrollment(recordings, samplerates):
    """
    Process a set of enrollment recordings to handle multiple microphone distances.
    
    Args:
        recordings: List of audio arrays
        samplerates: List of sample rates (or single sample rate for all)
        
    Returns:
        List of processed recordings
    """
    if isinstance(samplerates, (int, float)):
        samplerates = [samplerates] * len(recordings)
    
    processed_recordings = []
    
    for i, (rec, sr) in enumerate(zip(recordings, samplerates)):
        # Add a "near" version (original with normalization)
        near = preprocess_audio(rec, sr, normalize=True, equalize=False, noise_reduction=False)
        processed_recordings.append(near)
        
        # Add a "far" version (simulated distance with EQ changes)
        # We'll reduce bass, add slight reverb, and lower volume
        far = preprocess_audio(rec, sr, normalize=True, equalize=True, noise_reduction=True)
        
        # Further modify to simulate distance
        # 1. Further reduce bass frequencies
        nyquist = sr // 2
        high_pass_freq = 150  # Higher cutoff to reduce more bass
        b, a = signal.butter(3, high_pass_freq / nyquist, 'highpass')
        far = signal.filtfilt(b, a, far)
        
        # 2. Add slight "distance effect"
        far = far * 0.7  # Reduce volume
        
        # 3. Simple reverb effect (if recording is long enough)
        if len(far) > sr // 2:
            try:
                # Create a simple reverb IR (impulse response)
                reverb_time = 0.1  # 100ms reverb
                ir_length = int(reverb_time * sr)
                ir = np.exp(-np.linspace(0, 10, ir_length))
                ir = ir / np.sum(ir)  # Normalize
                
                # Apply reverb via convolution
                far = signal.convolve(far, ir, mode='same')
            except Exception:
                pass  # Skip reverb if it fails
        
        # Add to processed recordings
        processed_recordings.append(far)
    
    return processed_recordings

def preprocess_for_identification(audio, samplerate):
    """
    Preprocess audio for identification to make it robust to mic distance.
    
    Args:
        audio: Audio array
        samplerate: Sample rate
        
    Returns:
        Preprocessed audio
    """
    return preprocess_audio(audio, samplerate, normalize=True, equalize=True, noise_reduction=True)