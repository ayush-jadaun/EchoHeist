# EchoHeist: Professor's Sentiment Surveillance System

## Project Overview

EchoHeist was developed by team EchoRobbers for BotRush 3.0, organized by the Robotics Club of MNNIT. It's a sentiment analysis system styled after the popular TV show "Money Heist" (La Casa de Papel), designed to monitor and analyze voice recordings to detect emotional states.

## Key Features

- **Surveillance Records**: Upload pre-recorded audio files (.wav, .mpeg, .amr) to analyze voice patterns and detect emotional states
- **Quick Interrogation**: Record a 7-second audio sample using your microphone for immediate analysis
- **Live Monitoring**: Continuous real-time voice analysis with trend tracking and alert system that works through your device's microphone

## Technical Components

- Voice emotion recognition using deep learning (TensorFlow)
- Audio processing and feature extraction (Librosa)
- Noise reduction for clear signal processing
- Visualization of voice patterns and emotional states
- Real-time audio processing and analysis
- Streamlit-based interactive UI with Money Heist theme

## Emotion Detection Categories

The system detects the following emotional states:

- Sad → "Alert: Professor Distress"
- Surprised → "Caution: Unpredictable State"
- Joyful → "Stable: Cooperative"
- Euphoric → "Optimal: Compliant"

## Live Monitoring System

The live monitoring feature implements a sophisticated real-time audio processing pipeline:

1. **Continuous Audio Capture**: Uses `sounddevice` to create an audio stream from the microphone
2. **Buffer Management**: Maintains a rolling buffer of audio data for analysis
3. **Threading Implementation**: Processes audio in a separate thread to maintain UI responsiveness
4. **Real-time Feature Extraction**: Continuously extracts MFCC and mel-spectrogram features
5. **Emotion Trend Analysis**: Tracks emotion patterns over time with visual trend graphs
6. **Alert System**: Generates warnings based on detected patterns of distress
7. **Visualizations**: Updates waveform and spectrogram visualizations in real-time

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

Install the required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

Note: On Windows, you might need to use pipwin to install some audio processing libraries:

```bash
pip install pipwin
pipwin install pyaudio
```

### Running the Application

1. Clone the repository
2. Navigate to the project directory
3. Install dependencies
4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Usage

1. Choose one of the three tabs based on your monitoring needs
2. Upload audio files or record new samples to analyze
3. View the detailed analysis, including:
   - Voice waveform patterns
   - MFCC (Mel-frequency cepstral coefficients) features
   - Mel-spectrogram analysis
   - Emotion probability distribution
   - Threat assessment metrics

## Project Background

This project was developed for BotRush 3.0, a hackathon conducted by the Robotics Club of MNNIT. It combines audio processing, machine learning, and an engaging UI inspired by the Money Heist series to create an interactive voice sentiment analysis system.

## Model Information

The system uses a custom CNN (Convolutional Neural Network) model created by the team members (myself and Abhishek Pandey). The model (`finishing_model.h5`) was trained to classify emotions from audio features by analyzing MFCCs and mel-spectrograms to determine emotional states. The CNN architecture was specifically designed to identify patterns in speech that correlate with different emotional states.

## Team EchoRobbers

- Team members( Me and Abhishek Pandey) who developed this project for BotRush 3.0 competition with a Money Heist theme

## Acknowledgments

- The Robotics Club of MNNIT for organizing BotRush 3.0
- The creators of the libraries used in this project
- "La Casa de Papel" for the inspirational theme

Bella Ciao! 🎭
