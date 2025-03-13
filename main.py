import argparse
import torch
import os
import sys
import time
from audio_recorder import record_audio, save_wav
from voice_imprint import extract_features, combine_embeddings
from voice_database import VoiceDatabase
from audio_preprocessing import preprocess_audio, multi_distance_enrollment, preprocess_for_identification

def main():
    parser = argparse.ArgumentParser(description="Voice ID system with multi-user database")
    parser.add_argument("--mode", type=str, 
                        choices=["enroll", "identify", "list", "remove", "settings"], 
                        required=True,
                        help="Mode: enroll/identify/list/remove/settings")
    parser.add_argument("--user", type=str, default=None,
                        help="User identifier/name for enrollment or removal")
    parser.add_argument("--device", type=str, choices=["cpu", "nvidia", "amd"], default="cpu",
                        help="Device for computation (default: cpu)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds for each sample (default: 10.0)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of voice samples to record during enrollment (default: 5, recommended: 10+)")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="Audio input device index (if not provided, the program will auto-select)")
    parser.add_argument("--model", type=str, choices=["resemblyzer", "xvector", "dvector"], default="resemblyzer",
                        help="Voice embedding model to use (default: resemblyzer)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold for identification (default: 0.85)")
    parser.add_argument("--combine-method", type=str, choices=["average", "weighted", "pca"], default="weighted",
                        help="Method to combine multiple enrollment samples (default: weighted)")
    parser.add_argument("--database", type=str, default="voice_database.npy",
                        help="Path to the voice database file (default: voice_database.npy)")
    parser.add_argument("--set-threshold", type=float, default=None,
                        help="[Settings mode] Set global threshold (0.0-1.0)")
    parser.add_argument("--set-margin", type=float, default=None,
                        help="[Settings mode] Set required margin between best and second best match (0.0-1.0)")
    parser.add_argument("--strict", action="store_true",
                        help="[Identification mode] Use stricter verification criteria")
    args = parser.parse_args()
    
    # Initialize the voice database
    db = VoiceDatabase(args.database)
    
    # Process based on mode
    if args.mode == "list":
        # List all enrolled users
        list_users(db)
        return
        
    elif args.mode == "remove":
        # Remove a user from the database
        if not args.user:
            print("Error: User identifier (--user) is required for remove mode.")
            return
        remove_user(db, args.user)
        return
        
    elif args.mode == "settings":
        # Update database settings
        update_settings(db, args)
        return
        
    # Check if user ID is provided for enrollment
    if args.mode == "enroll" and not args.user:
        print("Error: User identifier (--user) is required for enrollment mode.")
        print("Example: --user john_doe")
        return
    
    # Create voices directory for recordings if it doesn't exist
    os.makedirs("voices", exist_ok=True)
    
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
    
    # Process enrollment or identification
    if args.mode == "enroll":
        enroll_user(db, args, compute_device)
    elif args.mode == "identify":
        identify_speaker(db, args, compute_device)

def update_settings(db, args):
    """Update database settings."""
    settings = {}
    
    if args.set_threshold is not None:
        if 0.0 <= args.set_threshold <= 1.0:
            settings["global_threshold"] = args.set_threshold
            print(f"Global threshold set to: {args.set_threshold}")
        else:
            print("Error: Threshold must be between 0.0 and 1.0")
            
    if args.set_margin is not None:
        if 0.0 <= args.set_margin <= 1.0:
            settings["second_best_margin"] = args.set_margin
            print(f"Second-best margin set to: {args.set_margin}")
        else:
            print("Error: Margin must be between 0.0 and 1.0")
            
    if settings:
        db.update_settings(settings)
        print("Settings updated successfully.")
    else:
        # Display current settings
        current = db.get_settings()
        print("\n=== Current Database Settings ===")
        for key, value in current.items():
            print(f"{key}: {value}")
        print("\nTo update settings, use:")
        print("  --set-threshold 0.85   # Set global verification threshold")
        print("  --set-margin 0.1       # Set required margin between best and second-best match")

def list_users(db):
    """List all enrolled users in the database with details."""
    users = db.get_all_users()
    
    if not users:
        print("No users enrolled in the database.")
        return
    
    print(f"\n=== Enrolled Users ({len(users)}) ===")
    print("ID                  | Model Type  | Samples | Quality | Custom Threshold")
    print("-" * 75)
    
    for user_id, user_data in users.items():
        model_type = user_data.get("model_type", "resemblyzer")
        sample_count = user_data.get("sample_count", 1)
        quality = user_data.get("enrollment_quality", 0.0)
        threshold_offset = user_data.get("threshold_offset", 0.0)
        
        quality_str = f"{quality:.2f}" if quality > 0 else "N/A"
        threshold_str = f"+{threshold_offset:.2f}" if threshold_offset > 0 else \
                       f"{threshold_offset:.2f}" if threshold_offset < 0 else "0.00"
        
        print(f"{user_id[:20]:<20} | {model_type:<10} | {sample_count:<7} | {quality_str:<7} | {threshold_str}")
    
    print("\nUse '--mode identify' to test voice recognition.")
    
    # Show settings
    settings = db.get_settings()
    print("\n=== Verification Settings ===")
    print(f"Global threshold: {settings.get('global_threshold', 0.75)}")
    print(f"Second-best margin: {settings.get('second_best_margin', 0.1)}")
    print(f"Minimum recommended samples: {settings.get('min_samples', 5)}")

def remove_user(db, user_id):
    """Remove a user from the database."""
    user = db.get_user(user_id)
    
    if not user:
        print(f"User '{user_id}' not found in the database.")
        return
    
    confirmation = input(f"Are you sure you want to remove user '{user_id}'? (y/n): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        return
    
    success = db.remove_user(user_id)
    if success:
        print(f"User '{user_id}' has been removed from the database.")
        
        # Clean up user recordings if they exist
        user_dir = os.path.join("voices", user_id)
        if os.path.exists(user_dir):
            try:
                import shutil
                shutil.rmtree(user_dir)
                print(f"Removed recordings directory: {user_dir}")
            except Exception as e:
                print(f"Note: Could not remove recordings directory: {e}")
    else:
        print(f"Failed to remove user '{user_id}'.")

def verify_enrollment_quality(samples, args):
    """
    Verify that enrollment samples are of good quality.
    Returns True if acceptable, False if there are issues.
    """
    if len(samples) < 3:
        print("\n\u26a0\ufe0f Warning: Very few samples (less than 3) will result in poor recognition.")
        print("   Consider enrolling with more samples (recommended: 10+).")
        return False
    
    if args.duration < 5.0:
        print("\n\u26a0\ufe0f Warning: Short audio samples (<5 seconds) may reduce recognition accuracy.")
        print("   Consider using --duration 10 or higher.")
        return False
        
    return True

def enroll_user(db, args, compute_device):
    """Enroll a new user in the database with multi-distance processing."""
    num_samples = max(1, args.samples)
    print(f"Enrolling user '{args.user}' with {num_samples} voice samples, {args.duration} seconds each")
    
    # Check if user already exists
    existing_user = db.get_user(args.user)
    if existing_user:
        choice = input(f"User '{args.user}' already exists. Overwrite? (y/n): ")
        if choice.lower() != 'y':
            print("Enrollment cancelled.")
            return
        print(f"Updating voice profile for '{args.user}'...")
    
    # Create user recordings directory
    user_dir = os.path.join("voices", args.user)
    os.makedirs(user_dir, exist_ok=True)
    
    # Record samples and extract embeddings
    raw_recordings = []
    sample_rates = []
    desired_samplerate = 16000  # 16kHz is standard for speech processing
    
    # Ask about microphone distance options
    multi_distance = False
    if num_samples >= 2:
        multi_distance = input("\nWould you like to use multi-distance enrollment (more robust)? (y/n): ").lower() == 'y'
        if multi_distance:
            print("\n=== Multi-Distance Enrollment ===")
            print("For best results, alternate between:")
            print("1. Close microphone (normal speaking distance)")
            print("2. Far microphone (twice the normal distance)")
    
    for i in range(num_samples):
        print(f"\nRecording sample {i+1} of {num_samples} ({args.duration} seconds)...")
        
        # For multi-distance, give specific instructions on each sample
        if multi_distance:
            if i % 2 == 0:
                print("CLOSE MICROPHONE: Please speak with the microphone close to your mouth.")
            else:
                print("FAR MICROPHONE: Please speak with the microphone further away.")
            
        # Give user a moment to prepare if multiple samples
        if i > 0:
            for j in range(3, 0, -1):
                print(f"Starting in {j}...")
                time.sleep(1)
                
        audio, used_samplerate = record_audio(duration=args.duration, samplerate=desired_samplerate, device=args.audio_device)
        if audio is None:
            print(f"Failed to record sample {i+1}. Skipping.")
            continue
            
        # Save the raw audio
        raw_recordings.append(audio)
        sample_rates.append(used_samplerate)
            
        # Save the raw audio as a WAV file with an index (for reference)
        save_wav(os.path.join(user_dir, f"recording_{i+1}.wav"), audio, used_samplerate)
    
    if not raw_recordings:
        print("Failed to record any audio samples. Exiting.")
        return
        
    # Process recordings to handle different microphone distances
    if multi_distance:
        print("\nProcessing recordings for multi-distance robustness...")
        processed_recordings = multi_distance_enrollment(raw_recordings, sample_rates)
        print(f"Created {len(processed_recordings)} processed samples (original + distance-adjusted variants)")
    else:
        # Just normalize the audio if not doing multi-distance
        processed_recordings = [
            preprocess_audio(rec, sr, normalize=True, equalize=False, noise_reduction=False)
            for rec, sr in zip(raw_recordings, sample_rates)
        ]
    
    # Extract embeddings from all processed recordings
    embeddings = []
    
    for i, (audio, sr) in enumerate(zip(processed_recordings, sample_rates)):
        sample_type = "original" if i < len(raw_recordings) else "processed"
        print(f"Extracting features from {sample_type} sample {(i % len(raw_recordings))+1}...")
        
        embedding = extract_features(audio, sr, model_type=args.model, device=compute_device)
        
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Sample {i+1} processed successfully")
        else:
            print(f"Failed to extract embedding from sample {i+1}")
    
    if not embeddings:
        print("Failed to extract any voice embeddings. Exiting.")
        return
        
    # Verify enrollment quality
    verify_enrollment_quality(embeddings, args)
        
    # Combine multiple embeddings into a single voice imprint
    print(f"Creating voice imprint for user '{args.user}' from {len(embeddings)} samples...")
    embedding = combine_embeddings(embeddings, method=args.combine_method)
    
    # Calculate custom threshold offset if needed
    threshold_offset = 0.0
    if len(raw_recordings) < db.get_settings().get("min_samples", 5):
        threshold_offset = 0.03  # Make threshold stricter for users with few samples
    
    # Add the user to the database
    success = db.add_user(
        user_id=args.user,
        embedding=embedding,
        model_type=args.model,
        sample_count=len(embeddings),
        combination_method=args.combine_method,
        threshold_offset=threshold_offset
    )
    
    if success:
        print(f"Voice imprint for '{args.user}' successfully saved to database!")
        
        if multi_distance:
            print(f"Created with multi-distance processing for improved robustness.")
            
        print(f"Created from {len(embeddings)} samples using {args.combine_method} combination method")
        
        if threshold_offset != 0.0:
            print(f"Note: Applied threshold adjustment of +{threshold_offset:.2f} due to low sample count.")
            
        print("You can now use the 'identify' mode to verify voices.")
    else:
        print("Error saving voice imprint to database.")

def identify_speaker(db, args, compute_device):
    """Identify a speaker by comparing with enrolled voices."""
    # Check if there are any enrolled users
    user_count = db.get_user_count()
    if user_count == 0:
        print("No users enrolled in the database. Please enroll users first.")
        return
    
    # Adjust threshold for strict mode
    threshold = args.threshold
    if args.strict and threshold is None:
        # Add 0.05 to whatever the default threshold is
        threshold = db.get_settings().get("global_threshold", 0.85) + 0.05
        print(f"Strict mode enabled. Using threshold: {threshold:.2f}")
    
    # Record audio
    print(f"Recording audio for identification ({args.duration} seconds)...")
    desired_samplerate = 16000
    audio, used_samplerate = record_audio(duration=args.duration, samplerate=desired_samplerate, device=args.audio_device)
    
    if audio is None:
        print("Failed to record audio. Exiting.")
        return

    # Save the raw audio as a WAV file (for reference)
    save_wav("identify_recording.wav", audio, used_samplerate)
    
    # Process audio to make it robust to microphone distance
    print("Preprocessing audio for robust identification...")
    processed_audio = preprocess_for_identification(audio, used_samplerate)
    
    # Extract the voice embedding using the selected model
    print(f"Extracting features using {args.model} model...")
    test_embedding = extract_features(processed_audio, used_samplerate, model_type=args.model, device=compute_device)
    
    if test_embedding is None:
        print("Failed to extract voice embedding.")
        return
        
    # Compare with all voice imprints in the database
    print(f"Comparing with {user_count} enrolled users...")
    
    best_match, best_similarity, all_results = db.identify_speaker(test_embedding, threshold)
    
    # Print results for all users (sorted by similarity)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)
    print("\n=== Comparison Results ===")
    for user_id, (similarity, is_match) in sorted_results:
        if similarity >= 0:  # Skip errors
            status = "\u2713 MATCH" if is_match else "\u2717 NO MATCH"
            print(f"User '{user_id}': Similarity = {similarity:.4f} [{status}]")
    
    # Report the best match
    print("\n=== Final Result ===")
    if best_match and all_results[best_match][1]:  # If best match is above threshold
        print(f"\u2705 Voice identified as user: '{best_match}'")
        print(f"Confidence: {best_similarity:.4f}")
        
        # Provide additional information about the match quality
        if best_similarity > 0.95:
            print("Very high confidence match.")
        elif best_similarity > 0.9:
            print("High confidence match.")
        elif best_similarity > 0.85:
            print("Good confidence match.")
        elif best_similarity > 0.8:
            print("Moderate confidence match.")
        else:
            print("Low confidence match, consider re-enrolling with more samples.")
    else:
        print(f"\u274c Voice not recognized as any enrolled user.")
        if best_match:
            print(f"Best match was '{best_match}' with similarity {best_similarity:.4f}, below threshold.")
            print("Try again or adjust the threshold with --threshold option.")
    
if __name__ == "__main__":
    main()