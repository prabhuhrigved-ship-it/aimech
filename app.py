import os
import pickle
import joblib
import librosa
import numpy as np
import streamlit as st

# --- Constants ---
TORQUE_RPM_MODEL_PATH = "models/extra_trees_models.pkl"
CONDITION_MODEL_PATH = "models/pitch_shift_extra_trees.joblib"
CHUNK_DURATION = 2
TARGET_SR = 16000
N_MFCC = 13

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    with open(TORQUE_RPM_MODEL_PATH, "rb") as f:
        torque_rpm_models = pickle.load(f)
    rpm_model = torque_rpm_models["rpm_model"]
    torque_model = torque_rpm_models["torque_model"]
    condition_model = joblib.load(CONDITION_MODEL_PATH)
    return rpm_model, torque_model, condition_model

try:
    rpm_model, torque_model, condition_model = load_models()
except Exception as e:
    st.error(f"Failed to load models. Please check your model paths. Error: {e}")
    st.stop()

# --- Feature Extraction Functions ---
def extract_torque_rpm_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=TARGET_SR)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=TARGET_SR)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=TARGET_SR)
    zero_crossing = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    
    extra_features = np.array([
        spectral_centroid.mean(), spectral_centroid.std(),
        spectral_bandwidth.mean(), spectral_bandwidth.std(),
        spectral_rolloff.mean(), spectral_rolloff.std(),
        zero_crossing.mean(), zero_crossing.std(),
        rms.mean(), rms.std()
    ])
    
    features = np.concatenate([mfcc_mean, mfcc_std, extra_features])
    return features

def extract_condition_features(audio, sr):
    features = []
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(np.mean(spectral_rolloff))
    features.append(np.std(spectral_rolloff))
    
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    return np.array(features)

# --- Streamlit UI ---
st.set_page_config(page_title="Engine Sound Analysis", page_icon="🔧")

st.title("🔧 Engine Sound Analysis")
st.markdown("Upload an engine recording to estimate engine condition, RPM and torque using machine-learning models trained on engine acoustic signals.")

# File uploader
audio_file = st.file_uploader("Upload Engine Sound", type=["wav", "mp3", "ogg", "flac"])

if audio_file is not None:
    # Playback the uploaded audio
    st.audio(audio_file)
    
    if st.button("Analyse", type="primary"):
        with st.spinner("Analyzing engine acoustics..."):
            try:
                # librosa can load directly from Streamlit's UploadedFile object
                audio, sr = librosa.load(audio_file, sr=None, mono=True)
                
                if len(audio) == 0:
                    st.error("The uploaded audio contains no samples.")
                    st.stop()
                    
                duration = len(audio) / sr
                
                # ==========================================
                # ENGINE CONDITION
                # ==========================================
                condition_features = extract_condition_features(audio, sr)
                condition_features = condition_features.reshape(1, -1)
                condition_prediction = condition_model.predict(condition_features)[0]
                condition = str(condition_prediction)
                
                # ==========================================
                # TORQUE + RPM
                # ==========================================
                chunk_samples = int(CHUNK_DURATION * sr)
                num_chunks = len(audio) // chunk_samples
                
                if num_chunks == 0:
                    st.warning(f"Audio duration: {duration:.2f} seconds. At least {CHUNK_DURATION} seconds are required for RPM and torque prediction.")
                    rpm_final = "N/A"
                    torque_final = "N/A"
                else:
                    rpm_predictions = []
                    torque_predictions = []
                    
                    for i in range(num_chunks):
                        start = i * chunk_samples
                        end = start + chunk_samples
                        audio_chunk = audio[start:end]
                        
                        audio_chunk_16k = librosa.resample(
                            audio_chunk, 
                            orig_sr=sr, 
                            target_sr=TARGET_SR
                        )
                        features = extract_torque_rpm_features(audio_chunk_16k)
                        features = features.reshape(1, -1)
                        
                        rpm_predictions.append(rpm_model.predict(features)[0])
                        torque_predictions.append(torque_model.predict(features)[0])
                        
                    # Average predictions across all complete chunks
                    rpm_final = f"{float(np.mean(rpm_predictions)):.2f} RPM"
                    torque_final = f"{float(np.mean(torque_predictions)):.2f} Nm"
                
                # ==========================================
                # DISPLAY RESULTS
                # ==========================================
                st.subheader("Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Engine Condition", value=condition)
                col2.metric(label="Estimated RPM", value=rpm_final)
                col3.metric(label="Estimated Torque", value=torque_final)
                
                with st.expander("View Analysis Details"):
                    st.write(f"**Audio duration:** {duration:.2f} seconds")
                    st.write(f"**Sample rate:** {sr} Hz")
                    st.write(f"**Chunks analysed:** {num_chunks}")
                    st.write(f"**Chunk duration:** {CHUNK_DURATION} seconds")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {type(e).__name__}: {str(e)}")
