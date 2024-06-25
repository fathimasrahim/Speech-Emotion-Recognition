import streamlit as st
from transformers import pipeline
import nemo.collections.asr as nemo_asr
from tempfile import NamedTemporaryFile

# Initialize ASR model
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Initialize emotion recognition model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def transcribe_audio(audio_path):
    transcript = asr_model.transcribe([audio_path])[0]
    return transcript

def recognize_emotion(text):
    emotion_predictions = emotion_classifier(text)
    return emotion_predictions

st.title("Speech Emotion Recognition")

# File uploader
st.header("Upload an audio file")
uploaded_file = st.file_uploader("Choose a file", type=["wav"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        audio_path = temp_audio.name
    
    # Display the audio player
    st.audio(audio_path, format="audio/wav")

    # Transcription
    st.write("Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    st.write("Transcript:")
    st.write(transcript)

    # Emotion Recognition
    st.write("Recognizing emotion...")
    emotion_predictions = recognize_emotion(transcript)
    st.write("Emotion Predictions:")
    for emotion in emotion_predictions[0]:
        st.write(f"{emotion['label']}: {emotion['score']:.4f}")
