import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

class TDNN(nn.Module):
    """
    Time Delay Neural Network for x-vector extraction.
    Based on the architecture from Snyder et al. (2018).
    """
    def __init__(self, input_dim=40, embedding_dim=512):
        super(TDNN, self).__init__()
        
        # Frame-level layers with different temporal contexts
        self.frame1 = nn.Conv1d(input_dim, 512, kernel_size=5, dilation=1)
        self.frame2 = nn.Conv1d(512, 512, kernel_size=3, dilation=2)
        self.frame3 = nn.Conv1d(512, 512, kernel_size=3, dilation=3)
        self.frame4 = nn.Conv1d(512, 512, kernel_size=1)
        self.frame5 = nn.Conv1d(512, 1500, kernel_size=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1500)
        
        # Segment-level layers
        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, embedding_dim)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(embedding_dim)
        
        # Set model to evaluation mode by default
        self.eval()
    
    def forward(self, x):
        # x should be (batch, time, features)
        # Convert to (batch, features, time) for 1D convolution
        x = x.transpose(1, 2)
        
        # Frame-level layers
        x = F.relu(self.bn1(self.frame1(x)))
        x = F.relu(self.bn2(self.frame2(x)))
        x = F.relu(self.bn3(self.frame3(x)))
        x = F.relu(self.bn4(self.frame4(x)))
        x = F.relu(self.bn5(self.frame5(x)))
        
        # Statistics pooling: concatenate mean and standard deviation
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        stat_pooling = torch.cat((mean, std), dim=1)
        
        # Segment-level layers
        x = F.relu(self.bn6(self.segment6(stat_pooling)))
        x = self.bn7(self.segment7(x))
        
        # L2 normalization for embedding
        embedding = F.normalize(x, p=2, dim=1)
        
        return embedding

class DVectorLSTM(nn.Module):
    """
    LSTM-based d-vector speaker embedding network.
    Based on the architecture from Wan et al. (2018).
    """
    def __init__(self, input_dim=40, hidden_dim=256, embedding_dim=256, num_layers=3):
        super(DVectorLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Set model to evaluation mode by default
        self.eval()
    
    def forward(self, x):
        # x should be (batch, time, features)
        self.lstm.flatten_parameters()
        
        # Get LSTM outputs
        lstm_out, _ = self.lstm(x)
        
        # Get last output for embedding
        last_hidden = lstm_out[:, -1, :]
        
        # Project to embedding space and normalize
        x = self.embedding_layer(last_hidden)
        x = self.bn(x)
        embedding = F.normalize(x, p=2, dim=1)
        
        return embedding

def extract_features_from_audio(audio, samplerate, model_type, device="cpu"):
    """
    Extract speech features (MFCCs) from raw audio.
    """
    # Convert to tensor if numpy array
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.FloatTensor(audio)
    else:
        audio_tensor = audio
        
    # Ensure correct shape (batch, time)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Move to specified device
    audio_tensor = audio_tensor.to(device)
        
    # Extract MFCC features
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=samplerate,
        n_mfcc=40,
        melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40}
    ).to(device)
    
    mfccs = mfcc_transform(audio_tensor)
    
    # Transpose to (batch, time, features) for LSTM/TDNN processing
    mfccs = mfccs.transpose(1, 2)
    
    # Check if we have enough frames for the x-vector model (at least 200ms of audio)
    min_frames_needed = int(0.2 * samplerate / 160)  # 160 is hop_length
    
    if mfccs.shape[1] < min_frames_needed:
        print(f"Warning: Audio sequence is very short ({mfccs.shape[1]} frames). Minimum recommended: {min_frames_needed} frames.")
        if model_type == "xvector":
            # For x-vectors, we need sufficient context due to the dilated convolutions
            # Repeat the sequence to create enough context
            repeats_needed = int(np.ceil(min_frames_needed / mfccs.shape[1]))
            print(f"Repeating sequence {repeats_needed} times to provide sufficient context.")
            mfccs = mfccs.repeat(1, repeats_needed, 1)
    
    return mfccs

def get_model(model_type, device="cpu"):
    """
    Get the appropriate model based on the requested type.
    """
    if model_type == "xvector":
        model = TDNN().to(device)
    elif model_type == "dvector":
        model = DVectorLSTM().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set to evaluation mode
    model.eval()
    return model

def extract_embedding(audio, samplerate, model_type="xvector", device="cpu"):
    """
    Extract advanced speaker embedding from audio.
    
    Args:
        audio: NumPy array of audio samples
        samplerate: Sample rate of the audio
        model_type: "xvector" or "dvector"
        device: Computing device ("cpu" or "cuda")
    
    Returns:
        Embedding vector as NumPy array
    """
    try:
        # Convert device string to torch device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        torch_device = torch.device(device)
        
        # Get the appropriate model
        model = get_model(model_type, torch_device)
        
        # Extract features
        features = extract_features_from_audio(audio, samplerate, model_type, torch_device)
        
        # Log shape information for debugging
        print(f"Audio features shape: {features.shape}")
        
        # Extract embedding
        with torch.no_grad():
            embedding = model(features)
            
        # Return as numpy array
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_pretrained_model(model_type, model_path, device="cpu"):
    """
    Load a pretrained model from a checkpoint file.
    """
    try:
        model = get_model(model_type, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return None