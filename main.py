import streamlit as st
from streamlit_audiorecorder import audiorecorder
import google.generativeai as genai
import tempfile
import os
import torch
import whisper
import base64
import re
import edge_tts
import asyncio

# Page config
st.set_page_config(page_title="Kreo Assistant", layout="wide", page_icon="🎮")
st.title("🎮 Kreo Tech Voice Assistant")

# Get Gemini API key securely from secrets
api_key = st.secrets["GEMINI_API_KEY"]

# System prompt
SYSTEM_PROMPT = """
You are Kreo, an energetic and helpful voice assistant for Kreo Tech.
Only answer questions related to Kreo Tech's products, gaming PCs, or services.
Keep replies short (max 50 words). Use a friendly, quirky tone.
Do not answer questions unrelated to Kreo Tech. If unsure, say “I’m only trained on Kreo Tech info!”
"""

# Clean up user audio input
def clean_input(text):
    replacements = ["creo", "krio", "curetech", "curotech", "kurio", "kreotech"]
    for word in replacements:
        text = re.sub(word, "Kreo", text, flags=re.IGNORECASE)
    return text

# Transcribe audio bytes using Whisper
def transcribe_audio_bytes(audio_bytes):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    model = whisper.load_model("base")
    result = model.transcribe(temp_path)
    return result["text"]

# Speak response using edge-tts
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

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.history = []

# Show chat history
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

# Layout for input
col1, col2 = st.columns([8, 2])

with col1:
    user_input = st.text_input("Ask something", placeholder="Ask me about Kreo Tech...", label_visibility="collapsed")

with col2:
    audio = audiorecorder("🎤 Click to speak", "Recording...", key="recorder")

# Handle audio input
if len(audio) > 0:
    with st.spinner("Transcribing..."):
        user_input = transcribe_audio_bytes(audio)
        user_input = clean_input(user_input)
        st.caption(f"🎙️ You said: {user_input}")

# Generate and respond
if user_input:
    user_input = clean_input(user_input)
    st.session_state.history.append(("user", user_input))

    prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}"
    response = model.generate_content(prompt).text
    st.session_state.history.append(("assistant", response))

    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)
    asyncio.run(speak(response))
