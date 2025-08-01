import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
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
st.set_page_config(page_title="Kreo Assistant", layout="wide", page_icon="🎮")
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

def save_buffer_to_wav(audio_buffer):
    """Saves the audio buffer from session state to a temporary WAV file."""
    if not audio_buffer:
        return None
    
    logging.info("Saving audio buffer to WAV file.")
    # Concatenate all audio frames
    raw_audio_data = np.concatenate([frame.to_ndarray() for frame in audio_buffer], axis=1)
    
    # Create a temporary file
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # Configure and write the WAV file
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(48000) # Sample rate from WebRTC
        # Convert float audio to 16-bit PCM format
        wf.writeframes((raw_audio_data.flatten() * 32767).astype(np.int16).tobytes())
        
    logging.info(f"WAV file saved at: {path}")
    return path

# -------------------- Session State Initialization --------------------

if "chat" not in st.session_state:
    # Start a new chat session with the Gemini model
    st.session_state.chat = gemini_model.start_chat(history=[])
    st.session_state.history = []
    st.session_state.audio_buffer = []

# -------------------- Audio Processor Class --------------------

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Receives audio frames from WebRTC and appends them to the session buffer."""
        # We store the frames in the session state to persist them across reruns
        st.session_state.audio_buffer.append(frame)
        return frame

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

# Voice Input
webrtc_ctx = webrtc_streamer(
    key="voice-input",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    send_audio_frame_by_frame=False, # Send frames in chunks
    audio_receiver_size=1024
)

# Logic to handle the state of the voice recorder
if webrtc_ctx.state.playing:
    # This block runs when the user has clicked "start" and is recording.
    # We clear the buffer at the start of a new recording session.
    if "is_recording" not in st.session_state or not st.session_state.is_recording:
        st.session_state.audio_buffer = []
        st.session_state.is_recording = True
    st.info("🎙️ Recording... Press the 'stop' button in the component above when you're done.")
else:
    # This block runs when the recorder is stopped.
    st.session_state.is_recording = False
    # Check if there is audio in the buffer to be processed.
    if st.session_state.get("audio_buffer"):
        st.info("Recording stopped. Ready to process.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Submit Voice", use_container_width=True, type="primary"):
                # Save the buffer to a WAV file
                wav_path = save_buffer_to_wav(st.session_state.audio_buffer)
                if wav_path:
                    # Transcribe the audio
                    transcribed_text = transcribe_audio(wav_path)
                    if transcribed_text:
                        st.caption(f"🎙️ You said: \"{transcribed_text}\"")
                        # Process the transcribed text like a regular message
                        handle_message(transcribed_text)
                    else:
                        st.warning("Could not understand audio. Please try again.")
                    # Clean up the temp file
                    os.remove(wav_path)
                
                # Clear the buffer and rerun to reset the UI
                st.session_state.audio_buffer = []
                st.rerun()

        with col2:
            if st.button("🗑️ Discard", use_container_width=True):
                # Clear the buffer and rerun to reset the UI
                st.session_state.audio_buffer = []
                st.rerun()

