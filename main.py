import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
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

# -------------------- Config --------------------
st.set_page_config(page_title="Kreo Assistant", layout="wide", page_icon="🎮")
st.title("🎮 Kreo Tech Voice Assistant")

# Get API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]

# Initialize Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# System prompt
SYSTEM_PROMPT = """
You are Kreo, an energetic and helpful voice assistant for Kreo Tech.
Only answer questions related to Kreo Tech's products, gaming PCs, or services.
Keep replies short (max 50 words). Use a friendly, quirky tone.
Always be helpful and energetic, with a touch of gaming culture.
"""

# Clean input
def clean_input(text):
    replacements = ["creo", "krio", "curetech", "curotech", "kurio", "kreotech"]
    for word in replacements:
        text = re.sub(word, "Kreo", text, flags=re.IGNORECASE)
    return text.strip()

# Transcribe from WAV using Whisper
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

# Text-to-speech
async def speak(text):
    voice = "en-US-ChristopherNeural"
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_path)
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

# -------------------- Audio Processor --------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

    def get_wav_file(self):
        if not self.frames:
            return None
        audio = np.concatenate(self.frames, axis=1).flatten()
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        return path

# -------------------- Chat Session --------------------
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.history = []

# Display past chat
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

# -------------------- Input UI --------------------
st.divider()
st.subheader("Talk or Type to Kreo")

# Text Input
col1, col2 = st.columns([5, 1])
with col1:
    user_text = st.text_input("Your question", placeholder="Ask me about Kreo Tech...", label_visibility="collapsed")
with col2:
    send_pressed = st.button("➡️", use_container_width=True)

# Voice Input
st.markdown("### Or record your voice:")
webrtc_ctx = webrtc_streamer(
    key="voice",
    mode="SENDONLY",
    audio_receiver_size=1024,
    in_audio_enabled=True,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Process text input
if send_pressed and user_text:
    cleaned_input = clean_input(user_text)
    st.session_state.history.append(("user", cleaned_input))

    prompt = f"{SYSTEM_PROMPT}\n\nUser: {cleaned_input}"
    response = model.generate_content(prompt).text
    st.session_state.history.append(("assistant", response))

    st.chat_message("user").write(cleaned_input)
    st.chat_message("assistant").write(response)

    asyncio.run(speak(response))

# Process voice input
if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
    if st.button("🔴 Stop Recording & Submit"):
        wav_path = webrtc_ctx.audio_processor.get_wav_file()
        if wav_path:
            transcribed = transcribe_audio(wav_path)
            cleaned = clean_input(transcribed)
            st.caption(f"🎙️ You said: {cleaned}")

            st.session_state.history.append(("user", cleaned))
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {cleaned}"
            response = model.generate_content(prompt).text
            st.session_state.history.append(("assistant", response))

            st.chat_message("user").write(cleaned)
            st.chat_message("assistant").write(response)

            asyncio.run(speak(response))
