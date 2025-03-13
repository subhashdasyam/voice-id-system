import argparse
import torch
import os
import sys
import time
from audio_recorder import record_audio, save_wav
from voice_imprint import extract_features, save_imprint, load_imprint, compare_imprint, combine_embeddings

def main():
    parser = argparse.ArgumentParser(description="Voice ID system with advanced speaker verification models")
    parser.add_argument("--mode", type=str, choices=["enroll", "identify"], required=True,
                        help="Mode: enroll to record and save voice imprint, identify to record and compare")
    parser.add_argument("--device", type=str, choices=["cpu", "nvidia", "amd"], default="cpu",
                        help="Device for computation (default: cpu)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds for each sample (default: 10.0)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of voice samples to record during enrollment (default: 1, recommended: 5+)")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="Audio input device index (if not provided, the program will auto-select)")
    parser.add_argument("--model", type=str, choices=["resemblyzer", "xvector", "dvector"], default="resemblyzer",
                        help="Voice embedding model to use (default: resemblyzer)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold for identification (model-dependent, typically 0.7-0.8)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model weights (for x-vector and d-vector models)")
    parser.add_argument("--combine-method", type=str, choices=["average", "weighted", "pca"], default="average",
                        help="Method to combine multiple enrollment samples (default: average)")
    args = parser.parse_args()
    
    # Check if advanced models are requested but not available
    if args.model in ["xvector", "dvector"]:
        try:
            from advanced_models import TDNN, DVectorLSTM
        except ImportError:
            print("Advanced models module not found. Make sure advanced_models.py is in the same directory.")
            choice = input("Continue with resemblyzer model instead? (y/n): ")
            if choice.lower() != 'y':
                sys.exit(1)
            args.model = "resemblyzer"
    
    # Setup device for computation
    if args.device in ["nvidia", "amd"]:
        if torch.cuda.is_available():
            compute_device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, falling back to CPU")
            compute_device = "cpu"
    else:
        compute_device = "cpu"
        print("Using CPU for computation")
    
    # Handle pretrained model path if specified
    if args.pretrained and args.model in ["xvector", "dvector"]:
        if not os.path.exists(args.pretrained):
            print(f"Pretrained model file not found: {args.pretrained}")
            print("Continuing with untrained model (embedding quality may be poor)")
    
    # Initialize sample collection
    desired_samplerate = 16000  # 16kHz is standard for speech processing
    embeddings = []
    
    if args.mode == "enroll":
        num_samples = max(1, args.samples)  # Ensure at least one sample
        print(f"Enrollment will use {num_samples} voice samples, {args.duration} seconds each")
        
        for i in range(num_samples):
            print(f"\nRecording sample {i+1} of {num_samples} ({args.duration} seconds)...")
            # Give user a moment to prepare if multiple samples
            if i > 0:
                for j in range(3, 0, -1):
                    print(f"Starting in {j}...")
                    time.sleep(1)
                    
            audio, used_samplerate = record_audio(duration=args.duration, samplerate=desired_samplerate, device=args.audio_device)
            if audio is None:
                print(f"Failed to record sample {i+1}. Skipping.")
                continue
                
            # Save the raw audio as a WAV file with an index (for reference)
            save_wav(f"recording_{i+1}.wav", audio, used_samplerate)
            
            # Extract the voice embedding using the selected model
            print(f"Extracting features from sample {i+1} using {args.model} model...")
            embedding = extract_features(audio, used_samplerate, model_type=args.model, device=compute_device)
            
            if embedding is not None:
                embeddings.append(embedding)
                print(f"Sample {i+1} processed successfully")
            else:
                print(f"Failed to extract embedding from sample {i+1}")
        
        if not embeddings:
            print("Failed to extract any voice embeddings. Exiting.")
            return
            
        # Combine multiple embeddings into a single voice imprint
        print(f"Creating voice imprint from {len(embeddings)} samples...")
        embedding = combine_embeddings(embeddings, method=args.combine_method)
        
    else:  # Identification mode - single sample is sufficient
        print(f"Recording audio for identification ({args.duration} seconds)...")
        audio, used_samplerate = record_audio(duration=args.duration, samplerate=desired_samplerate, device=args.audio_device)
        if audio is None:
            print("Failed to record audio. Exiting.")
            return

        # Save the raw audio as a WAV file (for reference)
        save_wav("recording.wav", audio, used_samplerate)
        
        # Extract the voice embedding using the selected model
        print(f"Extracting features using {args.model} model...")
        embedding = extract_features(audio, used_samplerate, model_type=args.model, device=compute_device)
        if embedding is None:
            print("Failed to extract voice embedding.")
            return

    if args.mode == "enroll":
        # Save the voice imprint (embedding)
        save_imprint(embedding, model_type=args.model, sample_count=len(embeddings), combination_method=args.combine_method)
        print(f"Voice imprint successfully saved using {args.model} model!")
        if len(embeddings) > 1:
            print(f"Created from {len(embeddings)} samples using {args.combine_method} combination method")
        print("You can now use the 'identify' mode to verify voices against this imprint.")
    
    elif args.mode == "identify":
        try:
            saved_embedding, model_type = load_imprint()
            if model_type != args.model:
                print(f"⚠️ Warning: Current model ({args.model}) differs from enrolled model ({model_type}).")
                print("Cross-model comparison may reduce accuracy. Consider re-enrolling with the same model.")
        except Exception as e:
            print("Error loading saved voice imprint. Have you enrolled your voice?")
            return
        
        # Use the provided threshold or default based on model type
        threshold = args.threshold
        
        match, similarity = compare_imprint(embedding, saved_embedding, threshold, model_type)
        
        if match:
            print(f"✅ Voice identified! Cosine similarity: {similarity:.4f}")
        else:
            print(f"❌ Voice not recognized. Cosine similarity: {similarity:.4f}")
        
        # Provide additional information about the match quality
        if similarity > 0.9:
            print("Very high confidence match.")
        elif similarity > 0.8:
            print("High confidence match.")
        elif similarity > 0.7:
            print("Moderate confidence match.")
        elif similarity > 0.6:
            print("Low confidence match.")
        else:
            print("Very low similarity.")
    
if __name__ == "__main__":
    main()