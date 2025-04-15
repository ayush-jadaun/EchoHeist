import os
import time
import threading
import queue
import random
from datetime import datetime

from flask import Flask, jsonify
from flask_socketio import SocketIO, emit

import numpy as np
import tensorflow as tf
import librosa

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

class RealtimeSpeechSentimentDetector:
    """
    A simulated real-time speech sentiment detection system that pushes updates using SocketIO.
    In a real implementation, this class would capture audio, process it, extract features,
    and predict sentiment via a deep learning model.
    """
    def __init__(self, model_path='speech_emotion_model_final.h5', sample_rate=22050, 
                 chunk_duration=3, overlap_duration=1):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.result_queue = queue.Queue()
        self.latest_prediction = None
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Using dummy predictions.")
            self.model = None
        else:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        self.running = False
        self.audio_thread = None
        self.process_thread = None

    def start(self):
        self.running = True        
        self.audio_thread = threading.Thread(target=self._simulate_audio_capture, daemon=True)
        self.audio_thread.start()
        self.process_thread = threading.Thread(target=self._simulate_processing, daemon=True)
        self.process_thread.start()
        print("Realtime speech sentiment detection started.")

    def stop(self):
        self.running = False
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=1.0)
        if self.process_thread is not None:
            self.process_thread.join(timeout=1.0)
        print("Realtime speech sentiment detection stopped.")

    def _simulate_audio_capture(self):
        while self.running:
            time.sleep(self.chunk_duration - self.overlap_duration)

    def _simulate_processing(self):
        sentiment_options = [
            {"emotion": "joyfully", "sentiment": "Positive"},
            {"emotion": "sad", "sentiment": "Negative"},
            {"emotion": "surprised", "sentiment": "Neutral"}
        ]
        while self.running:
            time.sleep(self.chunk_duration)
            selected = random.choice(sentiment_options)
            confidence = round(random.uniform(0.5, 0.99), 2)
            prediction = {
                "emotion": selected["emotion"],
                "sentiment": selected["sentiment"],
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            self.latest_prediction = prediction
            self.result_queue.put(prediction)
            socketio.emit('sentiment_update', prediction)
            print(f"Simulated prediction: {prediction}")

detector = RealtimeSpeechSentimentDetector()

def start_detector():
    detector.start()

detector_thread = threading.Thread(target=start_detector, daemon=True)
detector_thread.start()

@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
    if detector.latest_prediction:
        return jsonify(detector.latest_prediction)
    else:
        return jsonify({
            "message": "No sentiment analysis result available yet. Please try again shortly."
        })

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    if detector.latest_prediction:
        emit('sentiment_update', detector.latest_prediction)

if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        detector.stop()
        print("Microservice stopped.")
