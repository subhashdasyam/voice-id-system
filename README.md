# Voice ID System

Voice ID System is a Python-based speaker recognition system that allows you to enroll your voice and later identify it from new recordings. It leverages modern speaker embedding techniques using [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) to extract robust voice features. The system is designed in a modular way for future enhancements and supports CPU, NVIDIA GPU, and AMD GPU configurations.

## Features

- **Enrollment Mode:** Record a sample of your voice, extract a speaker embedding, and save it as your voice imprint.
- **Identification Mode:** Record a new sample and compare its embedding with the enrolled voice imprint using cosine similarity.
- **Device Auto-Selection:** Automatically selects a working audio input device or allows manual selection via a command-line option.
- **CPU/GPU Support:** Runs on CPU by default; supports NVIDIA and AMD GPUs if available.
- **Modular Design:** Easily extend or enhance the system (e.g., integrate advanced deep learning models) with separate modules.

## Project Structure

```
voice-id-system/
 ├── main.py               # Main entry point for enrollment and identification.
 ├── audio_recorder.py     # Module for recording audio and selecting the input device.
 ├── voice_imprint.py      # Module for extracting and comparing voice embeddings.
 ├── model.py              # Placeholder for future deep learning model integration.
 └── requirements.txt      # Python package dependencies.
```

## Requirements

### Python Packages

This project requires Python 3.8 or higher and the following Python packages:

- **torch** (for GPU support, if applicable)
- **torchaudio**
- **librosa**
- **numpy**
- **sounddevice**
- **scipy**
- **resemblyzer**

Install them via pip using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### System Dependencies (Linux)

For audio recording support, install the necessary system libraries. On Ubuntu/Pop!_OS (or other Debian-based systems), run:

```bash
sudo apt-get install portaudio19-dev libasound-dev
```

If your system uses Pipewire (default on many modern distros), ensure you have the compatibility packages installed:

```bash
sudo apt install pipewire-alsa
```

## How It Works

1. **Enrollment Mode (`--mode enroll`):**
   - Records your voice sample using your default (or a specified) audio input device.
   - Saves the raw audio as `recording.wav` (for reference).
   - Extracts a speaker embedding from the audio using Resemblyzer.
   - Saves the embedding (your voice imprint) to `voice_imprint.npy`.

2. **Identification Mode (`--mode identify`):**
   - Records a new audio sample.
   - Saves the sample as `recording.wav`.
   - Extracts a speaker embedding from the new recording.
   - Loads the saved voice imprint from `voice_imprint.npy`.
   - Compares the two embeddings using cosine similarity.
   - Reports whether the voice is identified based on a configurable threshold (default is 0.75).

3. **Device Selection:**
   - The system first attempts to use the default audio input device.
   - If the default device fails, it scans for available working devices and prompts you to choose one.
   - You can explicitly specify an audio device index with the `--audio-device` option.

4. **CPU, NVIDIA, AMD Support:**
   - The program runs on CPU by default.
   - Use the `--device` option (`cpu`, `nvidia`, or `amd`) to indicate your preferred computing device. The system will check GPU availability using PyTorch and fall back to CPU if necessary.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/subhashdasyam/voice-id-system.git
   cd voice-id-system
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies (Linux):**

   ```bash
   sudo apt-get install portaudio19-dev libasound-dev
   ```
   And if using Pipewire:
   ```bash
   sudo apt install pipewire-alsa
   ```

## Usage

### Enroll Your Voice

Enroll your voice to create a voice imprint. This command records 10 seconds of audio, extracts the speaker embedding, and saves it:

```bash
python main.py --mode enroll --duration 10 --device nvidia
```

*Note:* If you need to specify a particular audio device, use the `--audio-device` flag (e.g., `--audio-device 2`).

### Identify Your Voice

Identify or verify your voice by comparing a new recording with the enrolled imprint:

```bash
python main.py --mode identify --duration 5
```

The program records 5 seconds of audio, extracts the speaker embedding, and compares it with the saved imprint using cosine similarity.

## Troubleshooting

- **Audio Device Issues:**  
  If you encounter errors with the default audio device, list available devices with:

  ```bash
  python -c "import sounddevice as sd; print(sd.query_devices())"
  ```

  Then, use the `--audio-device` option to specify a working device index.

- **Permission Issues:**  
  Ensure your user account has the necessary permissions to access audio devices.

- **Threshold Tuning:**  
  If identification results are not as expected (false accepts or rejects), adjust the cosine similarity threshold in `voice_imprint.py` within the `compare_imprint` function.

- **GPU vs. CPU:**  
  The `--device` option only affects PyTorch device selection. If GPU is not available, the system automatically falls back to CPU.

## Example

![Recognized](recognized.png)
![Not Recognized](not-recognized.png)

## Future Enhancements

- **Improved Speaker Verification Models:**  
  Integrate advanced deep learning models (e.g., x-vectors, d-vectors) for more robust recognition.
- **Multiple Enrollment Samples:**  
  Allow multiple recordings during enrollment to create a more robust voice imprint.
- **Cross-Platform Testing:**  
  Expand testing and optimization for Windows and macOS.

## License

MIT

## Acknowledgements

- **Resemblyzer:** For providing a robust framework for speaker embedding extraction.
- **Open-source Community:** For the excellent libraries and tools that make this project possible.