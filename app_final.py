import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import noisereduce as nr
import soundfile as sf
import tempfile
import os
import time
from io import BytesIO
import base64
import sounddevice as sd
from PIL import Image
from collections import deque
import threading
import queue

# Set page configuration
st.set_page_config(
    page_title="Hostage Sentiment Monitor",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS - Money Heist theme (red, black, gold)
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .header {
        color: #e50914;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .subtitle {
        color: #d4af37; /* Gold */
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    .result-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e50914;
        box-shadow: 0 4px 6px rgba(229, 9, 20, 0.3);
        margin-bottom: 20px;
    }
    .danger-alert {
        background-color: #e50914;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.5; }
    }
    .stButton button {
        background-color: #e50914;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
        border: 1px solid #d4af37;
    }
    .stButton button:hover {
        background-color: #b30710;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border-radius: 5px 5px 0px 0px;
        border: 1px solid #e50914;
        color: #d4af37;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e50914;
        color: white;
    }
    .stProgress > div > div {
        background-color: #e50914;
    }
    .stMetric {
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #d4af37;
    }
    .stMetric label {
        color: #d4af37;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_model(model_path='speech_emotion_model_final_final.h5'):
    """Load and cache the emotion prediction model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"System Error: Model loading failed - {e}")
        return None

def reduce_noise(y, sr):
    """Reduces noise from an audio signal"""
    # Apply noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    
    # Normalize
    y_denoised = librosa.util.normalize(y_denoised)
    
    return y_denoised

def extract_features(y, sr):
    """Extract audio features for emotion prediction"""
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
    mfccs_normalized = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)

    mel_spec = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=40, n_fft=2048, hop_length=512, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_normalized = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)

    features = np.vstack([mfccs_normalized, mel_spec_normalized])
    
    # Apply padding/cropping 
    max_pad_len = 150
    if features.shape[1] > max_pad_len:
        start = (features.shape[1] - max_pad_len) // 2
        features = features[:, start:start + max_pad_len]
    else:
        pad_width = max_pad_len - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    
    # Reshape for model
    features_for_model = features.reshape(1, features.shape[0], features.shape[1], 1)
    
    return features_for_model, y_trimmed, mfccs, mel_spec_db

def predict_emotion(model, features):
    """Make emotion prediction using the model"""
    # Define emotion labels and mappings
    emotion_labels = {
        'sad': 0, 'surprised': 1, 'joyfully': 2, 'euphoric': 3
    }
    idx_to_emotion = {v: k for k, v in emotion_labels.items()}
    
    # Money Heist themed sentiment mapping
    sentiment_mapping = {
        'sad': 'Alert: Hostage Distress', 
        'surprised': 'Caution: Unpredictable State', 
        'joyfully': 'Stable: Cooperative', 
        'euphoric': 'Optimal: Compliant'
    }
    
    # Make prediction
    prediction = model.predict(features, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    predicted_emotion = idx_to_emotion[predicted_class]
    predicted_sentiment = sentiment_mapping[predicted_emotion]
    
    # Create probability dictionaries
    all_probs = {idx_to_emotion[i]: float(prob) for i, prob in enumerate(prediction)}
    
    # Money Heist themed sentiment categories
    heist_sentiment_mapping = {
        'Alert: Hostage Distress': 'DANGER', 
        'Caution: Unpredictable State': 'WARNING',
        'Stable: Cooperative': 'STABLE', 
        'Optimal: Compliant': 'STABLE'
    }
    
    sentiment_category = heist_sentiment_mapping[predicted_sentiment]
    
    sentiment_probs = {
        'STABLE': sum(all_probs[e] for e, s in sentiment_mapping.items() if heist_sentiment_mapping[s] == 'STABLE'),
        'WARNING': sum(all_probs[e] for e, s in sentiment_mapping.items() if heist_sentiment_mapping[s] == 'WARNING'),
        'DANGER': sum(all_probs[e] for e, s in sentiment_mapping.items() if heist_sentiment_mapping[s] == 'DANGER')
    }
    
    return {
        'emotion': predicted_emotion,
        'sentiment': predicted_sentiment,
        'category': sentiment_category,
        'confidence': float(confidence),
        'all_probs': all_probs,
        'sentiment_probs': sentiment_probs
    }

def plot_results(result, y_trimmed, sr, mfccs, mel_spec_db):
    """Create visualization plots for the results"""
    # Set up subplots with Money Heist color theme
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), facecolor='black')
    
    # Set text color for all plots
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    # Plot waveform
    axs[0].set_facecolor('black')
    title = f"Hostage Voice Pattern - Status: {result['sentiment']} ({result['confidence']:.2f})"
    axs[0].set_title(title, color='#e50914', fontweight='bold')
    time = np.arange(0, len(y_trimmed)) / sr
    axs[0].plot(time, y_trimmed, color='#e50914')
    axs[0].set_xlabel("Time (s)", color='#d4af37')
    axs[0].set_ylabel("Amplitude", color='#d4af37')
    axs[0].grid(True, color='#333333')
    
    # Plot MFCC
    axs[1].set_facecolor('black')
    im1 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axs[1], cmap='inferno')
    axs[1].set_title("Voice Pattern Analysis", color='#e50914', fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=axs[1], format='%+2.0f dB')
    cbar1.ax.yaxis.set_tick_params(color='white')
    cbar1.outline.set_edgecolor('#d4af37')
    axs[1].set_ylabel("MFCC Features", color='#d4af37')
    
    # Plot Mel Spectrogram
    axs[2].set_facecolor('black')
    im2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axs[2], cmap='inferno')
    axs[2].set_title("Spectral Analysis", color='#e50914', fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=axs[2], format='%+2.0f dB')
    cbar2.ax.yaxis.set_tick_params(color='white')
    cbar2.outline.set_edgecolor('#d4af37')
    axs[2].set_ylabel("Frequency (Hz)", color='#d4af37')
    
    # Plot emotion probabilities
    axs[3].set_facecolor('black')
    emotions = list(result['all_probs'].keys())
    values = list(result['all_probs'].values())
    bars = axs[3].bar(emotions, values)
    axs[3].set_title("Mental State Analysis", color='#e50914', fontweight='bold')
    axs[3].set_ylabel("Probability", color='#d4af37')
    axs[3].set_ylim(0, 1)
    
    # Define colors based on Money Heist theme
    # Money Heist themed sentiment mapping
    sentiment_mapping = {
        'sad': 'Alert: Hostage Distress', 
        'surprised': 'Caution: Unpredictable State', 
        'joyfully': 'Stable: Cooperative', 
        'euphoric': 'Optimal: Compliant'
    }
    
    # Color the bars
    for i, emotion in enumerate(emotions):
        if emotion in ['joyfully', 'euphoric']:
            bars[i].set_color('#00b300')  # Green for positive
        elif emotion == 'surprised':
            bars[i].set_color('#d4af37')  # Gold for neutral
        else:  # sad
            bars[i].set_color('#e50914')  # Red for negative
    
    # Add probability values as text
    for i, prob in enumerate(values):
        axs[3].text(i, prob + 0.02, f"{prob:.2f}", ha='center', color='white')
    
    plt.tight_layout()
    return fig

def plot_sentiment_probs(result):
    """Create a separate plot for sentiment probabilities"""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')
    ax.set_facecolor('black')
    
    # Set text color
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    sentiments = list(result['sentiment_probs'].keys())
    values = list(result['sentiment_probs'].values())
    
    # Create bars with colors
    bars = ax.bar(sentiments, values)
    for i, sentiment in enumerate(sentiments):
        if sentiment == 'STABLE':
            bars[i].set_color('#00b300')  # Green
        elif sentiment == 'WARNING':
            bars[i].set_color('#d4af37')  # Gold
        else:  # DANGER
            bars[i].set_color('#e50914')  # Red
    
    ax.set_title("Threat Assessment", color='#e50914', fontweight='bold')
    ax.set_ylabel("Probability", color='#d4af37')
    ax.set_ylim(0, 1)
    ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
    
    # Add values as text
    for i, prob in enumerate(values):
        ax.text(i, prob + 0.02, f"{prob:.2f}", ha='center', color='white')
    
    plt.tight_layout()
    return fig

def process_audio_file(model, audio_file):
    """Process an uploaded audio file"""
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load audio
        y, sr = librosa.load(tmp_path, sr=22050)
        
        # Apply noise reduction
        y = reduce_noise(y, sr)
        
        # Extract features
        features, y_trimmed, mfccs, mel_spec_db = extract_features(y, sr)
        
        # Make prediction
        result = predict_emotion(model, features)
        
        # Create plots
        fig = plot_results(result, y_trimmed, sr, mfccs, mel_spec_db)
        sentiment_fig = plot_sentiment_probs(result)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return result, fig, sentiment_fig
    
    except Exception as e:
        st.error(f"Error in surveillance system: {e}")
        return None, None, None

def record_audio(duration=5, sample_rate=22050):
    """Record audio from microphone"""
    try:
        # Record audio
        st.write(f"🔴 Surveillance active. Recording for {duration} seconds...")
        progress_bar = st.progress(0)
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        
        # Show progress
        for i in range(100):
            time.sleep(duration/100)
            progress_bar.progress(i + 1)
        
        sd.wait()
        st.success("Voice capture complete!")
        
        # Flatten and process
        audio = audio.flatten()
        
        # Apply noise reduction
        audio = reduce_noise(audio, sample_rate)
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            tmp_path = tmp_file.name
            
        # Create audio playback
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
            
        return audio_bytes, tmp_path
    
    except Exception as e:
        st.error(f"Error in surveillance system: {e}")
        return None, None

# Realtime processing
class RealtimeAudioProcessor:
    def __init__(self, model, sample_rate=22050, buffer_duration=3.0, chunk_duration=0.5):
        self.model = model
        self.sample_rate = sample_rate
        self.buffer_size = int(buffer_duration * sample_rate)
        self.chunk_size = int(chunk_duration * sample_rate)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.is_running = False
        self.processing_thread = None
        self.results_queue = queue.Queue()
        
        # For visualizations
        self.emotion_history = deque(maxlen=10)
        self.sentiment_history = deque(maxlen=10)
        self.category_history = deque(maxlen=10)
        
        # For alerts
        self.danger_count = 0
        self.warning_count = 0
        
    def start(self):
        """Start real-time processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        )
        self.stream.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        """Stop real-time processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Status: {status}")
        
        # Roll buffer and add new data
        self.audio_buffer = np.roll(self.audio_buffer, -len(indata))
        self.audio_buffer[-len(indata):] = indata.flatten()
    
    def process_loop(self):
        """Background processing loop"""
        last_process_time = time.time()
        process_interval = 0.5  # seconds
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_process_time >= process_interval:
                last_process_time = current_time
                
                # Process audio buffer
                buffer_copy = np.copy(self.audio_buffer)
                
                try:
                    # Denoise
                    y_denoised = reduce_noise(buffer_copy, self.sample_rate)
                    
                    # Extract features
                    features, y_trimmed, mfccs, mel_spec_db = extract_features(y_denoised, self.sample_rate)
                    
                    # Predict emotion
                    result = predict_emotion(self.model, features)
                    
                    # Update history
                    self.emotion_history.append(result['emotion'])
                    self.sentiment_history.append(result['sentiment'])
                    self.category_history.append(result['category'])
                    
                    # Track danger and warning counts
                    if result['category'] == 'DANGER':
                        self.danger_count += 1
                    elif result['category'] == 'WARNING':
                        self.warning_count += 1
                    
                    # Add result to queue
                    self.results_queue.put((result, y_trimmed, mfccs, mel_spec_db,self.sample_rate))
                    
                except Exception as e:
                    print(f"Error in surveillance system: {e}")
            
            # Small sleep to avoid CPU hogging
            time.sleep(0.1)
    
    def get_latest_result(self):
        """Get latest processing result if available"""
        try:
            result, y_trimmed, mfccs, mel_spec_db, sr = self.results_queue.get_nowait()
            return result, y_trimmed, mfccs, mel_spec_db, sr
        except queue.Empty:
            return None
    
    def get_emotion_stats(self):
        """Get emotion and sentiment statistics from history"""
        if not self.emotion_history:
            return None
        
        emotion_counts = {}
        for emotion in set(self.emotion_history):
            emotion_counts[emotion] = self.emotion_history.count(emotion) / len(self.emotion_history)
        
        category_counts = {'STABLE': 0, 'WARNING': 0, 'DANGER': 0}
        for category in self.category_history:
            category_counts[category] += 1 / len(self.category_history)
            
        return emotion_counts, category_counts
    
    def get_alert_status(self):
        """Get alert status based on danger and warning counts"""
        if self.danger_count > 3:
            return "CRITICAL", "Multiple hostage distress signals detected! Immediate intervention required."
        elif self.danger_count > 1:
            return "HIGH", "Hostage distress detected. Prepare intervention protocol."
        elif self.warning_count > 2:
            return "ELEVATED", "Caution: Unstable subject state. Monitor closely."
        else:
            return "LOW", "Situation stable. Continue surveillance."

# Main app
def main():
    st.markdown("<h1 class='header'>🎭 'BELLA CIAO' - HOSTAGE SENTIMENT SURVEILLANCE</h1>", unsafe_allow_html=True)
    
    # Display Money Heist mask icon
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 4rem; color: #e50914;">🎭</div>
            <p class="subtitle">CASA DE PAPEL SECURITY PROTOCOL</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model_path = 'speech_emotion_model_final_final.h5'
    model = load_model(model_path)
    
    if not model:
        st.error(f"SYSTEM FAILURE: Surveillance model initialization error. Contact the Professor immediately.")
        st.stop()
    
    # Create tabs with Money Heist themed names
    tab1, tab2, tab3 = st.tabs(["📁 Surveillance Records", "🎙️ Quick Interrogation", "📡 Live Monitoring"])
    
    # File Upload Tab
    with tab1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>ANALYZE RECORDED HOSTAGE COMMUNICATIONS</p>", unsafe_allow_html=True)
        st.write("Upload audio recordings to analyze hostage mental state.")
        
        audio_file = st.file_uploader("Select recording file", type=["wav", "mp3", "ogg", "flac"])
        
        if audio_file is not None:
            st.audio(audio_file, format='audio/wav')
            
            if st.button("ANALYZE RECORDING", key="analyze_upload"):
                with st.spinner("Decoding voice patterns..."):
                    result, fig, sentiment_fig = process_audio_file(model, audio_file)
                    
                if result:
                    st.markdown("### Intelligence Analysis Results:")
                    
                    # Display alert based on sentiment
                    if result['category'] == 'DANGER':
                        st.markdown("""
                        <div class='danger-alert'>
                            ⚠️ ALERT: HOSTAGE DISTRESS DETECTED! POSSIBLE INTERVENTION REQUIRED!
                        </div>
                        """, unsafe_allow_html=True)
                    elif result['category'] == 'WARNING':
                        st.warning("⚠️ CAUTION: Subject in unpredictable state. Monitor closely.")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Mental State", result['emotion'].capitalize())
                        st.metric("Detection Confidence", f"{result['confidence']:.2f}")
                    
                    with col2:
                        st.metric("Status Assessment", result['sentiment'])
                        st.metric("Threat Level", result['category'])
                    
                    # Show visualizations
                    st.pyplot(fig)
                    st.pyplot(sentiment_fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick Recording Tab
    with tab2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>RAPID SUBJECT ASSESSMENT</p>", unsafe_allow_html=True)
        st.write("Capture a quick 5-second voice sample to analyze hostage mental state.")
        
        if st.button("INITIATE VOICE CAPTURE", key="record_btn"):
            audio_bytes, tmp_path = record_audio(duration=5)
            
            if audio_bytes and tmp_path:
                st.audio(audio_bytes, format='audio/wav')
                
                with st.spinner("Analyzing voice patterns..."):
                    # Load and process the recording
                    y, sr = librosa.load(tmp_path, sr=22050)
                    
                    # Extract features
                    features, y_trimmed, mfccs, mel_spec_db = extract_features(y, sr)
                    
                    # Make prediction
                    result = predict_emotion(model, features)
                    
                    # Create plots
                    fig = plot_results(result, y_trimmed, sr, mfccs, mel_spec_db)
                    sentiment_fig = plot_sentiment_probs(result)
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                
                st.markdown("### Intelligence Analysis Results:")
                
                # Display alert based on sentiment
                if result['category'] == 'DANGER':
                    st.markdown("""
                    <div class='danger-alert'>
                        ⚠️ ALERT: HOSTAGE DISTRESS DETECTED! POSSIBLE INTERVENTION REQUIRED!
                    </div>
                    """, unsafe_allow_html=True)
                elif result['category'] == 'WARNING':
                    st.warning("⚠️ CAUTION: Subject in unpredictable state. Monitor closely.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Mental State", result['emotion'].capitalize())
                    st.metric("Detection Confidence", f"{result['confidence']:.2f}")
                
                with col2:
                    st.metric("Status Assessment", result['sentiment'])
                    st.metric("Threat Level", result['category'])
                
                # Show visualizations
                st.pyplot(fig)
                st.pyplot(sentiment_fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Real-time Analysis Tab
    with tab3:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>CONTINUOUS SURVEILLANCE</p>", unsafe_allow_html=True)
        st.write("Monitor hostage mental state in real-time using surveillance microphones.")
        
        # Initialize session state for real-time processor
        if 'realtime_processor' not in st.session_state:
            st.session_state.realtime_processor = RealtimeAudioProcessor(model)
            st.session_state.realtime_active = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.realtime_active:
                if st.button("ACTIVATE SURVEILLANCE"):
                    st.session_state.realtime_processor.start()
                    st.session_state.realtime_active = True
                    st.rerun()
            else:
                if st.button("TERMINATE SURVEILLANCE"):
                    st.session_state.realtime_processor.stop()
                    st.session_state.realtime_active = False
                    st.rerun()
        
        with col2:
            if st.session_state.realtime_active:
                st.markdown("<p style='color:#e50914; font-weight:bold; font-size:1.2rem;'>🔴 SURVEILLANCE ACTIVE</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#d4af37; font-weight:bold; font-size:1.2rem;'>⚪ STANDBY MODE</p>", unsafe_allow_html=True)
        
        # Create placeholders for real-time updates
        alert_status = st.empty()
        current_emotion = st.empty()
        current_sentiment = st.empty()
        graph_placeholder = st.empty()
        trend_placeholder = st.empty()
        
        # If real-time is active, update with the latest results
        if st.session_state.realtime_active:
            while True:
                # Get latest result
                latest = st.session_state.realtime_processor.get_latest_result()
                
                if latest:
                    result, y_trimmed, mfccs, mel_spec_db, sr = latest
                    
                    # Get alert status
                    alert_level, alert_message = st.session_state.realtime_processor.get_alert_status()
                    
                    # Display alert
                    if alert_level == "CRITICAL" or alert_level == "HIGH":
                        alert_status.markdown(f"""
                        <div class='danger-alert'>
                            ⚠️ THREAT LEVEL {alert_level}: {alert_message}
                        </div>
                        """, unsafe_allow_html=True)
                    elif alert_level == "ELEVATED":
                        alert_status.warning(f"⚠️ THREAT LEVEL {alert_level}: {alert_message}")
                    else:
                        alert_status.info(f"ℹ️ THREAT LEVEL {alert_level}: {alert_message}")
                    
                    # Update metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        current_emotion.metric("Current Mental State", result['emotion'].capitalize(), 
                                              delta=f"{result['confidence']:.2f}")
                    with col2:
                        current_sentiment.metric("Threat Assessment", result['category'])
                    
                    # Create the plot
                    fig = plot_results(result, y_trimmed, sr, mfccs, mel_spec_db)
                    graph_placeholder.pyplot(fig)
                    plt.close(fig)
                    
                    # Get emotion trend stats
                    stats = st.session_state.realtime_processor.get_emotion_stats()
                    if stats:
                        emotion_counts, category_counts = stats
                        
                        # Create trend plot with Money Heist theme
                        trend_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='black')
                        ax1.set_facecolor('black')
                        ax2.set_facecolor('black')
                        
                        # Plot emotion trends
                        emotions = list(emotion_counts.keys())
                        values = list(emotion_counts.values())
                        emotion_bars = ax1.bar(emotions, values)
                        
                        # Color emotion bars with Money Heist theme
                        emotion_colors = {
                            'sad': '#e50914',      # Red for negative
                            'surprised': '#d4af37', # Gold for neutral
                            'joyfully': '#00b300',  # Green for positive
                            'euphoric': '#00b300'   # Green for positive
                        }
                        
                        for i, emotion in enumerate(emotions):
                            emotion_bars[i].set_color(emotion_colors[emotion])
                        
                        ax1.set_title("Mental State Trend", color='#e50914', fontweight='bold')
                        ax1.set_ylabel("Frequency", color='#d4af37')
                        ax1.set_ylim(0, 1)
                        ax1.tick_params(colors='white')
                        ax1.grid(True, color='#333333', linestyle='--', linewidth=0.5)
                        
                        # Plot sentiment trend
                        categories = list(category_counts.keys())
                        cat_values = list(category_counts.values())
                        category_bars = ax2.bar(categories, cat_values)
                        
                        # Color category bars
                        category_colors = {
                            'STABLE': '#00b300',  # Green
                            'WARNING': '#d4af37', # Gold
                            'DANGER': '#e50914'   # Red
                        }
                        
                        for i, category in enumerate(categories):
                            category_bars[i].set_color(category_colors[category])
                        
                        ax2.set_title("Threat Level Trend", color='#e50914', fontweight='bold')
                        ax2.set_ylabel("Frequency", color='#d4af37')
                        ax2.set_ylim(0, 1)
                        ax2.tick_params(colors='white')
                        ax2.grid(True, color='#333333', linestyle='--', linewidth=0.5)
                        
                        plt.tight_layout()
                        trend_placeholder.pyplot(trend_fig)
                        plt.close(trend_fig)
                
                time.sleep(0.1)  # Small delay to prevent UI freezing
                
                # Check if still active (handles the case where stop was pressed)
                if not st.session_state.realtime_active:
                    break
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # About section with Money Heist theme
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>OPERATION DETAILS</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("""
        This surveillance system uses advanced neural network technology to analyze hostage mental states.
        
        **Capabilities:**
        - Detection of 4 mental states (distressed, surprised, cooperative, compliant)
        - Classification into 3 threat levels (DANGER, WARNING, STABLE)
        - Real-time voice pattern analysis
        - Automated alert system for hostage distress detection
        
        All audio signals are processed with noise reduction algorithms to improve detection accuracy.
        """)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; border: 1px solid #e50914; padding: 10px; border-radius: 5px;">
            <p style="color: #d4af37; font-weight: bold;">SECURITY LEVEL</p>
            <h3 style="color: #e50914;">PROFESSOR<br>ACCESS ONLY</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; font-style: italic; color: #d4af37;">
        "A war is not lost until you surrender." - The Professor
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()