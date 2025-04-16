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
    page_title="Speech Emotion Analyzer",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f7;
    }
    .header {
        color: #1E1E1E;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
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
        st.error(f"Error loading model: {e}")
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
    sentiment_mapping = {
        'sad': 'Negative', 'surprised': 'Neutral', 
        'joyfully': 'Positive', 'euphoric': 'Positive'
    }
    
    # Make prediction
    prediction = model.predict(features, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    predicted_emotion = idx_to_emotion[predicted_class]
    predicted_sentiment = sentiment_mapping[predicted_emotion]
    
    # Create probability dictionaries
    all_probs = {idx_to_emotion[i]: float(prob) for i, prob in enumerate(prediction)}
    sentiment_probs = {
        'Positive': sum(all_probs[e] for e, s in sentiment_mapping.items() if s == 'Positive'),
        'Negative': sum(all_probs[e] for e, s in sentiment_mapping.items() if s == 'Negative'),
        'Neutral': sum(all_probs[e] for e, s in sentiment_mapping.items() if s == 'Neutral')
    }
    
    return {
        'emotion': predicted_emotion,
        'sentiment': predicted_sentiment,
        'confidence': float(confidence),
        'all_probs': all_probs,
        'sentiment_probs': sentiment_probs
    }

def plot_results(result, y_trimmed, sr, mfccs, mel_spec_db):
    """Create visualization plots for the results"""
    # Set up subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot waveform
    axs[0].set_title(f"Waveform - Emotion: {result['emotion']} ({result['confidence']:.2f}) - Sentiment: {result['sentiment']}")
    time = np.arange(0, len(y_trimmed)) / sr
    axs[0].plot(time, y_trimmed, color='b')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    
    # Plot MFCC
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axs[1])
    axs[1].set_title("MFCC Features")
    fig.colorbar(axs[1].collections[0], ax=axs[1], format='%+2.0f dB')
    axs[1].set_ylabel("MFCC Coeffs")
    
    # Plot Mel Spectrogram
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axs[2])
    axs[2].set_title("Mel Spectrogram")
    fig.colorbar(axs[2].collections[0], ax=axs[2], format='%+2.0f dB')
    axs[2].set_ylabel("Frequency (Hz)")
    
    # Plot emotion probabilities
    emotions = list(result['all_probs'].keys())
    values = list(result['all_probs'].values())
    bars = axs[3].bar(emotions, values)
    axs[3].set_title("Emotion Prediction Probabilities")
    axs[3].set_ylabel("Probability")
    axs[3].set_ylim(0, 1)
    
    # Define colors based on sentiment
    sentiment_mapping = {
        'sad': 'Negative', 'surprised': 'Neutral', 
        'joyfully': 'Positive', 'euphoric': 'Positive'
    }
    
    # Color the bars
    for i, emotion in enumerate(emotions):
        if sentiment_mapping[emotion] == 'Positive':
            bars[i].set_color('green')
        elif sentiment_mapping[emotion] == 'Negative':
            bars[i].set_color('red')
        else:  # Neutral
            bars[i].set_color('blue')
    
    # Add probability values as text
    for i, prob in enumerate(values):
        axs[3].text(i, prob + 0.02, f"{prob:.2f}", ha='center')
    
    plt.tight_layout()
    return fig

def plot_sentiment_probs(result):
    """Create a separate plot for sentiment probabilities"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sentiments = list(result['sentiment_probs'].keys())
    values = list(result['sentiment_probs'].values())
    
    # Create bars with colors
    bars = ax.bar(sentiments, values)
    for i, sentiment in enumerate(sentiments):
        if sentiment == 'Positive':
            bars[i].set_color('green')
        elif sentiment == 'Negative':
            bars[i].set_color('red')
        else:  # Neutral
            bars[i].set_color('blue')
    
    ax.set_title("Combined Sentiment Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    
    # Add values as text
    for i, prob in enumerate(values):
        ax.text(i, prob + 0.02, f"{prob:.2f}", ha='center')
    
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
        st.error(f"Error processing audio: {e}")
        return None, None, None

def record_audio(duration=5, sample_rate=22050):
    """Record audio from microphone"""
    try:
        # Record audio
        st.write(f"Recording for {duration} seconds...")
        progress_bar = st.progress(0)
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        
        # Show progress
        for i in range(100):
            time.sleep(duration/100)
            progress_bar.progress(i + 1)
        
        sd.wait()
        st.success("Recording complete!")
        
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
        st.error(f"Error recording audio: {e}")
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
                    
                    # Add result to queue
                    self.results_queue.put((result, y_trimmed, mfccs, mel_spec_db))
                    
                except Exception as e:
                    print(f"Error in processing: {e}")
            
            # Small sleep to avoid CPU hogging
            time.sleep(0.1)
    
    def get_latest_result(self):
        """Get latest processing result if available"""
        try:
            result, y_trimmed, mfccs, mel_spec_db = self.results_queue.get_nowait()
            return result, y_trimmed, mfccs, mel_spec_db
        except queue.Empty:
            return None
    
    def get_emotion_stats(self):
        """Get emotion and sentiment statistics from history"""
        if not self.emotion_history:
            return None
        
        emotion_counts = {}
        for emotion in set(self.emotion_history):
            emotion_counts[emotion] = self.emotion_history.count(emotion) / len(self.emotion_history)
            
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        sentiment_mapping = {
            'sad': 'Negative', 'surprised': 'Neutral', 
            'joyfully': 'Positive', 'euphoric': 'Positive'
        }
        
        for emotion in self.emotion_history:
            sentiment = sentiment_mapping[emotion]
            sentiment_counts[sentiment] += 1 / len(self.emotion_history)
            
        return emotion_counts, sentiment_counts

# Main app
def main():
    st.markdown("<h1 class='header'>🎭 Speech Emotion Analyzer</h1>", unsafe_allow_html=True)
    
    # Load model
    model_path = 'speech_emotion_model_final_final.h5'
    model = load_model(model_path)
    
    if not model:
        st.error(f"Could not load model from {model_path}. Please check the path.")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["File Upload", "Quick Recording", "Real-time Analysis"])
    
    # File Upload Tab
    with tab1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Upload Audio File")
        st.write("Upload an audio file to analyze the emotion and sentiment.")
        
        audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])
        
        if audio_file is not None:
            st.audio(audio_file, format='audio/wav')
            
            if st.button("Analyze", key="analyze_upload"):
                with st.spinner("Processing audio..."):
                    result, fig, sentiment_fig = process_audio_file(model, audio_file)
                    
                if result:
                    st.markdown("### Analysis Results:")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Detected Emotion", result['emotion'].capitalize())
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                    
                    with col2:
                        st.metric("Sentiment Category", result['sentiment'])
                        
                        # Display sentiment emoji
                        if result['sentiment'] == 'Positive':
                            st.markdown("### 😊")
                        elif result['sentiment'] == 'Negative':
                            st.markdown("### 😢")
                        else:
                            st.markdown("### 😐")
                    
                    # Show visualizations
                    st.pyplot(fig)
                    st.pyplot(sentiment_fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick Recording Tab
    with tab2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Quick 5-Second Recording")
        st.write("Record a 5-second audio clip and analyze the emotion.")
        
        if st.button("Start Recording", key="record_btn"):
            audio_bytes, tmp_path = record_audio(duration=5)
            
            if audio_bytes and tmp_path:
                st.audio(audio_bytes, format='audio/wav')
                
                with st.spinner("Analyzing recording..."):
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
                
                st.markdown("### Analysis Results:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Detected Emotion", result['emotion'].capitalize())
                    st.metric("Confidence", f"{result['confidence']:.2f}")
                
                with col2:
                    st.metric("Sentiment Category", result['sentiment'])
                    
                    # Display sentiment emoji
                    if result['sentiment'] == 'Positive':
                        st.markdown("### 😊")
                    elif result['sentiment'] == 'Negative':
                        st.markdown("### 😢")
                    else:
                        st.markdown("### 😐")
                
                # Show visualizations
                st.pyplot(fig)
                st.pyplot(sentiment_fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Real-time Analysis Tab
    with tab3:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Real-time Emotion Analysis")
        st.write("Analyze emotions in real-time using your microphone.")
        
        # Initialize session state for real-time processor
        if 'realtime_processor' not in st.session_state:
            st.session_state.realtime_processor = RealtimeAudioProcessor(model)
            st.session_state.realtime_active = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.realtime_active:
                if st.button("Start Real-time Analysis"):
                    st.session_state.realtime_processor.start()
                    st.session_state.realtime_active = True
                    st.experimental_rerun()
            else:
                if st.button("Stop Real-time Analysis"):
                    st.session_state.realtime_processor.stop()
                    st.session_state.realtime_active = False
                    st.experimental_rerun()
        
        with col2:
            if st.session_state.realtime_active:
                st.markdown("#### 🔴 Recording")
            else:
                st.markdown("#### ⚪ Idle")
        
        # Create placeholders for real-time updates
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
                    result, y_trimmed, mfccs, mel_spec_db = latest
                    
                    # Update metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        current_emotion.metric("Current Emotion", result['emotion'].capitalize(), 
                                              delta=f"{result['confidence']:.2f}")
                    with col2:
                        current_sentiment.metric("Sentiment", result['sentiment'])
                    
                    # Create the plot
                    fig = plot_results(result, y_trimmed, sr, mfccs, mel_spec_db)
                    graph_placeholder.pyplot(fig)
                    plt.close(fig)
                    
                    # Get emotion trend stats
                    stats = st.session_state.realtime_processor.get_emotion_stats()
                    if stats:
                        emotion_counts, sentiment_counts = stats
                        
                        # Create trend plot
                        trend_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                        
                        # Plot emotion trends
                        emotions = list(emotion_counts.keys())
                        values = list(emotion_counts.values())
                        emotion_bars = ax1.bar(emotions, values)
                        
                        # Color emotion bars
                        sentiment_mapping = {
                            'sad': 'Negative', 'surprised': 'Neutral', 
                            'joyfully': 'Positive', 'euphoric': 'Positive'
                        }
                        emotion_colors = {'sad': 'red', 'surprised': 'blue', 
                                          'joyfully': 'green', 'euphoric': 'green'}
                        
                        for i, emotion in enumerate(emotions):
                            emotion_bars[i].set_color(emotion_colors[emotion])
                        
                        ax1.set_title("Emotion Trend")
                        ax1.set_ylim(0, 1)
                        
                        # Plot sentiment trend
                        sentiments = list(sentiment_counts.keys())
                        sent_values = list(sentiment_counts.values())
                        sentiment_bars = ax2.bar(sentiments, sent_values)
                        
                        # Color sentiment bars
                        for i, sentiment in enumerate(sentiments):
                            if sentiment == 'Positive':
                                sentiment_bars[i].set_color('green')
                            elif sentiment == 'Negative':
                                sentiment_bars[i].set_color('red')
                            else:
                                sentiment_bars[i].set_color('blue')
                        
                        ax2.set_title("Sentiment Trend")
                        ax2.set_ylim(0, 1)
                        
                        plt.tight_layout()
                        trend_placeholder.pyplot(trend_fig)
                        plt.close(trend_fig)
                
                time.sleep(0.1)  # Small delay to prevent UI freezing
                
                # Check if still active (handles the case where stop was pressed)
                if not st.session_state.realtime_active:
                    break
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # About section
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("About")
    st.write("""
    This application uses a deep learning model to analyze emotions in speech. 
    It can detect 4 different emotions (sad, surprised, joyfully, euphoric) and categorize them into 3 sentiment categories (Positive, Negative, Neutral).
    
    The analysis includes:
    - Emotion detection
    - Sentiment classification
    - Audio feature visualization
    - Confidence scores
    
    All audio is processed with noise reduction to improve accuracy.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()