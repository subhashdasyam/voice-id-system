from resemblyzer import VoiceEncoder
import numpy as np

def extract_features(audio, samplerate=16000):
    """
    Extract a speaker embedding from the given audio using Resemblyzer.
    The audio array is expected to be a 1D NumPy array at the given samplerate.
    """
    try:
        encoder = VoiceEncoder()
        # Resemblyzer expects the audio to be a 16kHz waveform.
        # It internally applies pre-processing.
        embedding = encoder.embed_utterance(audio)
        return embedding
    except Exception as e:
        print("Error extracting voice embedding:", e)
        return None

def save_imprint(embedding, filename="voice_imprint.npy"):
    """
    Saves the voice imprint (embedding) to a file.
    """
    try:
        np.save(filename, embedding)
        print("Saved voice imprint to", filename)
    except Exception as e:
        print("Error saving voice imprint:", e)

def load_imprint(filename="voice_imprint.npy"):
    """
    Loads the saved voice imprint (embedding) from a file.
    """
    try:
        embedding = np.load(filename)
        print("Loaded voice imprint from", filename)
        return embedding
    except Exception as e:
        print("Error loading voice imprint:", e)
        raise e

def compare_imprint(embedding1, embedding2, threshold=0.75):
    """
    Compares two voice embeddings using cosine similarity.
    Returns a tuple (match: bool, similarity: float), where similarity is in [0,1].
    Higher similarity indicates more likely the same speaker.
    The threshold may need to be tuned; typically, your own voice should score above 0.8.
    """
    from numpy.linalg import norm
    cosine_similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    match = cosine_similarity > threshold
    return match, cosine_similarity
