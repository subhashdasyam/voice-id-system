from resemblyzer import VoiceEncoder
import numpy as np
import torch
import os
from sklearn.decomposition import PCA

# Import the advanced models
try:
    from advanced_models import extract_embedding
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("Advanced models not available. Only Resemblyzer will be used.")

def extract_features(audio, samplerate=16000, model_type="resemblyzer", device="cpu"):
    """
    Extract a speaker embedding from the given audio using the specified model.
    
    Args:
        audio: 1D NumPy array containing audio samples
        samplerate: Sample rate of the audio (in Hz)
        model_type: Model to use for feature extraction:
                    - "resemblyzer": Original d-vector model (default)
                    - "xvector": Time Delay Neural Network-based x-vector
                    - "dvector": LSTM-based d-vector
        device: Computing device ("cpu" or "cuda")
    
    Returns:
        Embedding vector as a NumPy array
    """
    try:
        if model_type == "resemblyzer":
            encoder = VoiceEncoder(device=device)
            embedding = encoder.embed_utterance(audio)
            return embedding
        
        elif model_type in ["xvector", "dvector"]:
            if not ADVANCED_MODELS_AVAILABLE:
                print(f"Advanced models not available. Falling back to Resemblyzer.")
                return extract_features(audio, samplerate, "resemblyzer", device)
                
            # Check if audio is long enough (minimum 3 seconds recommended for x-vectors)
            min_samples = 3 * samplerate
            if len(audio) < min_samples and model_type == "xvector":
                print(f"Warning: Audio too short for reliable {model_type} extraction.")
                print(f"Recommended minimum: 3 seconds. Got: {len(audio)/samplerate:.1f} seconds.")
                choice = input("Continue with xvector (may be less accurate) or use resemblyzer instead? (x/r): ")
                if choice.lower() != 'x':
                    print(f"Using Resemblyzer for this sample.")
                    return extract_features(audio, samplerate, "resemblyzer", device)
            
            # Use the advanced model for feature extraction
            embedding = extract_embedding(audio, samplerate, model_type, device)
            
            # If advanced model fails, fall back to Resemblyzer
            if embedding is None:
                print(f"Advanced model {model_type} failed. Falling back to Resemblyzer.")
                return extract_features(audio, samplerate, "resemblyzer", device)
                
            return embedding
        
        else:
            print(f"Unknown model type: {model_type}. Falling back to Resemblyzer.")
            return extract_features(audio, samplerate, "resemblyzer", device)
            
    except Exception as e:
        print("Error extracting voice embedding:", e)
        import traceback
        traceback.print_exc()
        return None

def combine_embeddings(embeddings, method="average"):
    """
    Combines multiple embeddings into a single voice imprint.
    
    Args:
        embeddings: List of embedding vectors
        method: Method to combine embeddings:
                - "average": Simple averaging (default)
                - "weighted": Weighted averaging based on embedding norms
                - "pca": PCA-based combination
    
    Returns:
        Combined embedding vector
    """
    if not embeddings:
        raise ValueError("No embeddings provided to combine")
        
    if len(embeddings) == 1:
        return embeddings[0]
        
    # Convert list to numpy array for easier manipulation
    embeddings_array = np.array(embeddings)
    
    if method == "average":
        # Simple averaging of embeddings
        combined = np.mean(embeddings_array, axis=0)
        
    elif method == "weighted":
        # Weighted averaging based on embedding norms (higher weight to stronger embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1)
        weights = norms / np.sum(norms)
        combined = np.sum(embeddings_array * weights[:, np.newaxis], axis=0)
        
    elif method == "pca":
        # PCA-based combination to capture main variation directions
        if len(embeddings) < 3:
            print("Too few samples for PCA, falling back to average")
            return combine_embeddings(embeddings, "average")
            
        try:
            pca = PCA(n_components=1)
            principal_component = pca.fit_transform(embeddings_array)
            combined = pca.inverse_transform(np.array([[1.0]]))[0]
        except Exception as e:
            print(f"PCA failed: {e}, falling back to average")
            return combine_embeddings(embeddings, "average")
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    # Normalize the combined embedding
    combined = combined / np.linalg.norm(combined)
    
    return combined

def save_imprint(embedding, filename="voice_imprint.npy", model_type="resemblyzer", sample_count=1, combination_method="average"):
    """
    Saves the voice imprint (embedding) to a file.
    
    Args:
        embedding: The voice embedding vector
        filename: Path to save the embedding
        model_type: Type of model used to generate the embedding
        sample_count: Number of samples used to create this imprint
        combination_method: Method used to combine multiple samples (if applicable)
    """
    try:
        # Save both the embedding and metadata about which model created it
        data = {
            "embedding": embedding,
            "model_type": model_type,
            "version": "3.0",  # Version tracking for compatibility
            "created": np.datetime64('now'),
            "sample_count": sample_count,
            "combination_method": combination_method if sample_count > 1 else "single"
        }
        np.save(filename, data)
        print(f"Saved voice imprint to {filename} using {model_type} model")
        if sample_count > 1:
            print(f"Imprint combines {sample_count} samples using {combination_method} method")
    except Exception as e:
        print("Error saving voice imprint:", e)

def load_imprint(filename="voice_imprint.npy"):
    """
    Loads the saved voice imprint (embedding) from a file.
    
    Args:
        filename: Path to the saved embedding file
        
    Returns:
        Tuple of (embedding, model_type)
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Voice imprint file {filename} not found")
            
        # Load the data
        data = np.load(filename, allow_pickle=True).item()
        
        # Handle both old and new format
        if isinstance(data, dict) and "embedding" in data:
            embedding = data["embedding"]
            model_type = data.get("model_type", "resemblyzer")  # Default for backward compatibility
            version = data.get("version", "1.0")
            print(f"Loaded voice imprint from {filename} (model: {model_type}, version: {version})")
        else:
            # For backward compatibility with old format (just the embedding)
            embedding = data
            model_type = "resemblyzer"
            sample_count = 1
            combination_method = "single"
            print(f"Loaded legacy voice imprint from {filename}")
            
        # Extract additional metadata if available
        sample_count = data.get("sample_count", 1) if isinstance(data, dict) else 1
        combination_method = data.get("combination_method", "single") if isinstance(data, dict) else "single"
        
        if sample_count > 1:
            print(f"Imprint was created from {sample_count} samples using {combination_method} method")
            
        return embedding, model_type
    except Exception as e:
        print("Error loading voice imprint:", e)
        raise e

def compare_imprint(embedding1, embedding2, threshold=None, model_type="resemblyzer"):
    """
    Compares two voice embeddings using cosine similarity.
    
    Args:
        embedding1: First voice embedding
        embedding2: Second voice embedding
        threshold: Similarity threshold for match/no-match decision
        model_type: Model type used to create embeddings (may affect threshold)
    
    Returns:
        Tuple of (match: bool, similarity: float)
    """
    from numpy.linalg import norm
    
    # Use different default thresholds based on model type
    if threshold is None:
        if model_type == "xvector":
            threshold = 0.70  # X-vectors typically need a lower threshold
        elif model_type == "dvector":
            threshold = 0.75  # Modern d-vectors
        else:
            threshold = 0.75  # Resemblyzer default
    
    # Calculate cosine similarity
    cosine_similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    match = cosine_similarity > threshold
    
    return match, cosine_similarity