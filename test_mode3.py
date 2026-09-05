import io
import soundfile as sf
import librosa
import numpy as np
import joblib
from feature_extractor import extract_286d_features, load_audio

def test_mode3_synthesis():
    print("--- Testing Mode 3 Synthesis & Defense Pipeline ---")
    sim_input = "sample_audio/human_voice_sample.wav"
    
    # 1. Load baseline
    y_base, sr = load_audio(sim_input, sr=16000)
    print(f"Loaded baseline audio: len={len(y_base)}, sr={sr}")
    
    # 2. Transform (Pitch + Time stretch + Noise)
    y_synth = librosa.effects.pitch_shift(y_base, sr=sr, n_steps=4.0)
    y_synth = librosa.effects.time_stretch(y_synth, rate=1.15)
    noise = np.random.normal(0, 0.02, len(y_synth))
    y_synth += noise
    
    max_abs = np.max(np.abs(y_synth))
    if max_abs > 0:
        y_synth = y_synth / max_abs
        
    # 3. Export to immutable WAV bytes
    buf = io.BytesIO()
    sf.write(buf, y_synth, sr, format='WAV')
    synth_bytes = buf.getvalue()
    print(f"Generated synthetic WAV bytes: length = {len(synth_bytes)} bytes")
    
    # 4. Extract 286-D features & predict
    feats, y_sig, breakdown = extract_286d_features(synth_bytes, sr=16000)
    assert feats.shape == (286,), f"Expected 286 features, got {feats.shape}"
    
    scaler = joblib.load("feature_scaler.pkl")
    discriminator = joblib.load("svm_discriminator.pkl")
    
    scaled_feats = scaler.transform([feats])
    pred = discriminator.predict(scaled_feats)[0]
    probs = discriminator.predict_proba(scaled_feats)[0]
    
    print(f"Mode 3 Defense Prediction: {pred} (0=Real, 1=AI Cloned)")
    print(f"Confidence Probabilities: Real={probs[0]*100:.1f}%, AI={probs[1]*100:.1f}%")
    print("[SUCCESS] MODE 3 SYNTHESIS AND DISCRIMINATOR TEST PASSED!")

if __name__ == "__main__":
    test_mode3_synthesis()
