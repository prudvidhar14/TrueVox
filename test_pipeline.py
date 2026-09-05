import os
import joblib
import numpy as np
from feature_extractor import extract_286d_features

def test_full_pipeline():
    print("--- 1. Testing Feature Extraction Engine ---")
    feat_r, y_r, breakdown_r = extract_286d_features("sample_audio/human_voice_sample.wav")
    print(f"Human Voice Feature Shape: {feat_r.shape}")
    assert feat_r.shape == (286,), f"Expected (286,), got {feat_r.shape}"
    
    feat_s, y_s, breakdown_s = extract_286d_features("sample_audio/ai_clone_sample.wav")
    print(f"AI Clone Feature Shape: {feat_s.shape}")
    assert feat_s.shape == (286,), f"Expected (286,), got {feat_s.shape}"
    
    print("\n--- 2. Testing Scaler & Model Artifact Inference ---")
    scaler = joblib.load("feature_scaler.pkl")
    discriminator = joblib.load("svm_discriminator.pkl")
    
    scaled_r = scaler.transform([feat_r])
    pred_r = discriminator.predict(scaled_r)[0]
    probs_r = discriminator.predict_proba(scaled_r)[0]
    print(f"Real Sample Pred: {pred_r} (0=Real, 1=AI), Probs: Real={probs_r[0]*100:.1f}%, AI={probs_r[1]*100:.1f}%")
    
    scaled_s = scaler.transform([feat_s])
    pred_s = discriminator.predict(scaled_s)[0]
    probs_s = discriminator.predict_proba(scaled_s)[0]
    print(f"AI Sample Pred: {pred_s} (0=Real, 1=AI), Probs: Real={probs_s[0]*100:.1f}%, AI={probs_s[1]*100:.1f}%")
    
    print("\n--- 3. Verifying All Breakdown Features ---")
    for key, val in breakdown_r.items():
        if key != "mel_db":
            print(f"  - {key}: {val}")
            
    print("\n[SUCCESS] PIPELINE TEST PASSED FULLY!")

if __name__ == "__main__":
    test_full_pipeline()
