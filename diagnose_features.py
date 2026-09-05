import numpy as np
import librosa
import soundfile as sf
import io
import joblib
from feature_extractor import extract_286d_features, load_audio

def diagnose():
    print("--- DIAGNOSING ACTUAL 286-D FEATURES & PREDICTIONS ---")
    
    # 1. Real sample
    f_real, _, b_real = extract_286d_features("sample_audio/human_voice_sample.wav")
    
    # 2. AI clone sample
    f_ai, _, b_ai = extract_286d_features("sample_audio/ai_clone_sample.wav")
    
    # 3. Mode 3 synthesized variant
    y_base, sr = load_audio("sample_audio/human_voice_sample.wav", sr=16000)
    y_synth = librosa.effects.pitch_shift(y_base, sr=sr, n_steps=6.0)
    y_synth = librosa.effects.time_stretch(y_synth, rate=1.25)
    noise = np.random.normal(0, 0.05, len(y_synth))
    y_synth = y_synth + noise
    t_arr = np.arange(len(y_synth)) / sr
    y_synth += 0.1 * np.sin(2 * np.pi * 800.0 * t_arr)
    y_synth = y_synth / np.max(np.abs(y_synth))
    
    buf = io.BytesIO()
    sf.write(buf, y_synth, sr, format='WAV')
    f_mode3, _, b_mode3 = extract_286d_features(buf.getvalue())
    
    print("\n[Real Sample Key Stats]:")
    print(f"  Mel Mean Avg: {b_real['mel_mean_avg']:.2f}, Mel Std Avg: {b_real['mel_std_avg']:.2f}")
    print(f"  MFCC Mean Avg: {b_real['mfcc_mean_avg']:.2f}, MFCC Std Avg: {b_real['mfcc_std_avg']:.2f}")
    print(f"  Spec Centroid Mean: {b_real['spectral_centroid_mean']:.2f}, Std: {b_real['spectral_centroid_std']:.2f}")
    print(f"  ZCR Mean: {b_real['zcr_mean']:.4f}, Std: {b_real['zcr_std']:.4f}")

    print("\n[AI Clone Sample Key Stats]:")
    print(f"  Mel Mean Avg: {b_ai['mel_mean_avg']:.2f}, Mel Std Avg: {b_ai['mel_std_avg']:.2f}")
    print(f"  MFCC Mean Avg: {b_ai['mfcc_mean_avg']:.2f}, MFCC Std Avg: {b_ai['mfcc_std_avg']:.2f}")
    print(f"  Spec Centroid Mean: {b_ai['spectral_centroid_mean']:.2f}, Std: {b_ai['spectral_centroid_std']:.2f}")
    print(f"  ZCR Mean: {b_ai['zcr_mean']:.4f}, Std: {b_ai['zcr_std']:.4f}")

    print("\n[Mode 3 Synthesized Variant Key Stats]:")
    print(f"  Mel Mean Avg: {b_mode3['mel_mean_avg']:.2f}, Mel Std Avg: {b_mode3['mel_std_avg']:.2f}")
    print(f"  MFCC Mean Avg: {b_mode3['mfcc_mean_avg']:.2f}, MFCC Std Avg: {b_mode3['mfcc_std_avg']:.2f}")
    print(f"  Spec Centroid Mean: {b_mode3['spectral_centroid_mean']:.2f}, Std: {b_mode3['spectral_centroid_std']:.2f}")
    print(f"  ZCR Mean: {b_mode3['zcr_mean']:.4f}, Std: {b_mode3['zcr_std']:.4f}")

    scaler = joblib.load("feature_scaler.pkl")
    discriminator = joblib.load("svm_discriminator.pkl")
    
    p_real = discriminator.predict_proba(scaler.transform([f_real]))[0]
    p_ai = discriminator.predict_proba(scaler.transform([f_ai]))[0]
    p_mode3 = discriminator.predict_proba(scaler.transform([f_mode3]))[0]
    
    print("\n[Current Discriminator Predictions (Real vs AI Probabilities)]:")
    print(f"  Real Sample: Real={p_real[0]*100:.1f}%, AI={p_real[1]*100:.1f}%")
    print(f"  AI Clone Sample: Real={p_ai[0]*100:.1f}%, AI={p_ai[1]*100:.1f}%")
    print(f"  Mode 3 Synthesized: Real={p_mode3[0]*100:.1f}%, AI={p_mode3[1]*100:.1f}%")

if __name__ == "__main__":
    diagnose()
