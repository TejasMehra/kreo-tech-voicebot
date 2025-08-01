import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import wave
import os
import whisper
import re
import google.generativeai as genai
import asyncio
import edge_tts
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# -------------------- App Configuration --------------------
st.set_page_config(page_title="Kreo Assistant", layout="wide", page_icon="�")
st.title("🎮 Kreo Tech Voice Assistant")

# -------------------- API & Model Initialization --------------------

# Get API key from Streamlit secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except (KeyError, FileNotFoundError):
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
    st.stop()


# System prompt defining the assistant's personality and role
SYSTEM_PROMPT = """
You are Kreo, an energetic and helpful voice assistant for Kreo Tech, a company that sells high-end custom gaming PCs.
Your primary goal is to answer questions about Kreo Tech's products and services.
You must only answer questions related to Kreo Tech, gaming PCs, PC components, and gaming culture.
If a user asks a question outside of this scope, politely decline to answer and steer the conversation back to Kreo Tech.
Keep your replies concise and friendly, ideally under 50 words.
Adopt a quirky, energetic tone that would appeal to gamers. Use gaming slang where appropriate, but don't overdo it.
Always be helpful and positive. Let's get this bread!
"""

# Use caching to load models only once
@st.cache_resource
def load_models():
    """Loads the Gemini and Whisper models."""
    logging.info("Loading Gemini and Whisper models...")
    # Add the system prompt directly to the Gemini model configuration
    gemini_model = genai.GenerativeModel(
        "models/gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT
    )
    whisper_model = whisper.load_model("base")
    logging.info("Models loaded successfully.")
    return gemini_model, whisper_model

gemini_model, whisper_model = load_models()

# -------------------- Utility Functions --------------------

def clean_input(text):
    """Cleans the transcribed text to correct common misspellings of 'Kreo'."""
    # A list of common mispronunciations or transcription errors
    replacements = ["creo", "krio", "curetech", "curotech", "kurio", "kreotech", "crayo"]
    for word in replacements:
        # Use case-insensitive regex substitution
        text = re.sub(r'\b' + word + r'\b', "Kreo", text, flags=re.IGNORECASE)
    return text.strip()

def transcribe_audio(file_path):
    """Transcribes an audio file using the pre-loaded Whisper model."""
    logging.info(f"Transcribing audio file: {file_path}")
    try:
        result = whisper_model.transcribe(file_path)
        transcription = result["text"]
        logging.info(f"Transcription result: {transcription}")
        return transcription
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

async def text_to_speech(text):
    """Converts text to speech using Edge TTS and plays it automatically."""
    logging.info("Generating speech for text...")
    voice = "en-US-ChristopherNeural"  # A friendly, energetic male voice
    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_path)

    # Read the audio file and encode it in base64
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    
    b64_audio = base64.b64encode(audio_bytes).decode()
    
    # Embed the audio using HTML with autoplay
    audio_html = f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    
    # Clean up the temporary file
    os.remove(tmp_path)
    logging.info("Speech generated and played.")

# -------------------- Session State Initialization --------------------

if "chat" not in st.session_state:
    # Start a new chat session with the Gemini model
    st.session_state.chat = gemini_model.start_chat(history=[])
    st.session_state.history = []

# -------------------- Main Application UI --------------------

# Display the chat history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

# Function to handle a new message (from text or voice)
def handle_message(user_input):
    cleaned_input = clean_input(user_input)
    
    # Add user message to history and display it
    st.session_state.history.append(("user", cleaned_input))
    with st.chat_message("user"):
        st.write(cleaned_input)

    # Send message to Gemini and get the response
    try:
        response = st.session_state.chat.send_message(cleaned_input).text
    except Exception as e:
        response = f"Sorry, I ran into an error: {e}"
        logging.error(f"Gemini API error: {e}")

    # Add assistant response to history and display it
    st.session_state.history.append(("assistant", response))
    with st.chat_message("assistant"):
        st.write(response)

    # Run the text-to-speech function
    # asyncio.run() can cause issues if an event loop is already running.
    # This is a safer way to run an async function from a sync context.
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(text_to_speech(response))
        loop.close()
    except Exception as e:
        logging.error(f"Error running TTS: {e}")


st.divider()
st.subheader("Talk or Type to Kreo")

# Text Input
user_text = st.chat_input("Ask me about Kreo Tech...")
if user_text:
    handle_message(user_text)

# Voice Input using audio_recorder_streamlit
st.markdown("### Or record your voice:")
audio_bytes = audio_recorder(
    text="Click to Record",
    recording_color="#e84040",
    neutral_color="#6a6a6a",
    icon_size="2x",
    pause_threshold=120.0, # Increase pause threshold to avoid premature stopping
)

if audio_bytes:
    st.info("Audio recorded! Processing...")
    # Save the audio bytes to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        # The audio_recorder returns a WAV file in bytes
        tmp_wav.write(audio_bytes)
        wav_path = tmp_wav.name

    # Transcribe the audio
    transcribed_text = transcribe_audio(wav_path)

    # Clean up the temp file
    os.remove(wav_path)

    if transcribed_text:
        st.caption(f"🎙️ You said: \"{transcribed_text}\"")
        # Process the transcribed text
        handle_message(transcribed_text)
    else:
        st.warning("Could not understand the audio. Please try again.")

�