import numpy as np
import librosa
import io
import soundfile as sf

def load_audio(audio_source, sr=16000):
    """
    Safely loads audio from a file path, file-like BytesIO buffer, UploadedFile, or tuple (y, sr).
    Resamples to target sr (default 16000Hz) and downmixes to mono.
    """
    if isinstance(audio_source, tuple) and len(audio_source) == 2:
        y, orig_sr = audio_source
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        return y, sr

    # Rewind file-like objects (e.g. BytesIO, Streamlit UploadedFile) if seekable
    if hasattr(audio_source, "seek"):
        try:
            audio_source.seek(0)
        except Exception:
            pass

    if isinstance(audio_source, (bytes, bytearray, io.BytesIO)) or hasattr(audio_source, "read"):
        if isinstance(audio_source, (bytes, bytearray)):
            audio_source = io.BytesIO(audio_source)
        y, orig_sr = sf.read(audio_source)
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # Convert stereo to mono
        if orig_sr != sr:
            y = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
        return y.astype(np.float32), sr

    # Assume audio_source is a file path string or path object
    y, orig_sr = librosa.load(audio_source, sr=sr, mono=True)
    return y.astype(np.float32), sr

def extract_286d_features(audio_source, sr=16000):
    """
    Extracts the exact 286-dimensional acoustic feature vector matching scaler & SVM model specs:
      - Mel-spectrogram (128 bins): Mean (128) + Std (128) = 256
      - MFCCs (13 coefficients): Mean (13) + Std (13) = 26
      - Spectral Centroid: Mean (1) + Std (1) = 2
      - Zero-Crossing Rate: Mean (1) + Std (1) = 2
      Total = 256 + 26 + 2 + 2 = 286 features.
    
    Returns:
        features_vector: np.ndarray of shape (286,)
        audio_signal: np.ndarray (resampled mono audio)
        breakdown_dict: dict containing summary components for dashboard rendering
    """
    y, sr = load_audio(audio_source, sr=sr)
    
    # Standardize length to 3 seconds to eliminate bias
    target_duration = 3.0
    target_samples = int(sr * target_duration)
    y = librosa.util.fix_length(y, size=target_samples)
        
    # 1. Mel-spectrogram: 128 bins -> mean (128) and std (128) across time axes = 256 features
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)  # 128
    mel_std = np.std(mel_db, axis=1)    # 128
    
    # 2. MFCC: 13 coefficients -> mean (13) and std (13) across time axes = 26 features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)   # 13
    mfcc_std = np.std(mfcc, axis=1)     # 13
    
    # 3. Spectral Centroid: mean (1) and std (1) = 2 features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.array([np.mean(centroid)], dtype=np.float32)  # 1
    cent_std = np.array([np.std(centroid)], dtype=np.float32)    # 1
    
    # 4. Zero-Crossing Rate (ZCR): mean (1) and std (1) = 2 features
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.array([np.mean(zcr)], dtype=np.float32)        # 1
    zcr_std = np.array([np.std(zcr)], dtype=np.float32)          # 1
    
    # Concatenate all features into single 286-D vector
    feature_vector = np.concatenate([
        mel_mean, mel_std,
        mfcc_mean, mfcc_std,
        cent_mean, cent_std,
        zcr_mean, zcr_std
    ])
    
    assert feature_vector.shape[0] == 286, f"Expected 286 features, but got {feature_vector.shape[0]}"
    
    breakdown_dict = {
        "mel_db": mel_db,
        "mel_mean_avg": float(np.mean(mel_mean)),
        "mel_std_avg": float(np.mean(mel_std)),
        "mfcc_mean_avg": float(np.mean(mfcc_mean)),
        "mfcc_std_avg": float(np.mean(mfcc_std)),
        "spectral_centroid_mean": float(cent_mean[0]),
        "spectral_centroid_std": float(cent_std[0]),
        "zcr_mean": float(zcr_mean[0]),
        "zcr_std": float(zcr_std[0]),
        "duration_sec": len(y) / sr
    }
    
    return feature_vector, y, breakdown_dict
