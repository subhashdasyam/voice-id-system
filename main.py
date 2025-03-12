import argparse
import torch
from audio_recorder import record_audio, save_wav
from voice_imprint import extract_features, save_imprint, load_imprint, compare_imprint

def main():
    parser = argparse.ArgumentParser(description="Voice ID system")
    parser.add_argument("--mode", type=str, choices=["enroll", "identify"], required=True,
                        help="Mode: enroll to record and save voice imprint, identify to record and compare")
    parser.add_argument("--device", type=str, choices=["cpu", "nvidia", "amd"], default="cpu",
                        help="Device for training/inference (default: cpu)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording duration in seconds (default: 5.0)")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="Audio input device index (if not provided, the program will auto-select)")
    args = parser.parse_args()
    
    # Setup training/inference device (used here only for info, not for audio processing)
    if args.device in ["nvidia", "amd"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {args.device}")
        else:
            print("GPU not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    desired_samplerate = 16000
    print("Recording audio...")
    audio, used_samplerate = record_audio(duration=args.duration, samplerate=desired_samplerate, device=args.audio_device)
    if audio is None:
        print("Failed to record audio. Exiting.")
        return

    # Save the raw audio as a WAV file (for reference)
    save_wav("recording.wav", audio, used_samplerate)
    
    # Extract the voice embedding using Resemblyzer
    embedding = extract_features(audio, used_samplerate)
    if embedding is None:
        print("Failed to extract voice embedding.")
        return

    if args.mode == "enroll":
        # Save the voice imprint (embedding)
        save_imprint(embedding)
        print("Voice imprint saved successfully!")
    elif args.mode == "identify":
        try:
            saved_embedding = load_imprint()
        except Exception as e:
            print("Error loading saved voice imprint. Have you enrolled your voice?")
            return
        match, similarity = compare_imprint(embedding, saved_embedding)
        if match:
            print(f"Voice identified. Cosine similarity: {similarity:.4f}")
        else:
            print(f"Voice not recognized. Cosine similarity: {similarity:.4f}")
    
if __name__ == "__main__":
    main()
