import streamlit as st
import google.generativeai as genai
import tempfile
import os
import torch
import whisper
import sounddevice as sd
import numpy as np
import wave
import asyncio
import edge_tts
import base64
import re

st.set_page_config(page_title="Kreo Assistant", layout="wide", page_icon="🎮")
st.title("🎮 Kreo Tech Voice Assistant")

# --- API Key ---
api_key = st.text_input("Gemini API Key", type="password", label_visibility="collapsed")

# --- System Prompt ---
SYSTEM_PROMPT = """
You are Kreo, an energetic and helpful voice assistant for Kreo Tech.
Only answer questions related to Kreo Tech's products, gaming PCs, or services.
Keep replies short (max 50 words). Use a friendly, quirky tone.
"""

# --- Helper Functions ---
def clean_input(text):
    for word in ["creo", "krio", "curetech", "curotech", "kurio", "kreotech"]:
        text = re.sub(word, "Kreo", text, flags=re.IGNORECASE)
    return text

def record_audio(filename, duration=5, fs=44100):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes((recording * 32767).astype(np.int16).tobytes())
    wf.close()

def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

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

# --- Main Chat Logic ---
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    if "chat" not in st.session_state:
        st.session_state.chat = model.start_chat(history=[])
        st.session_state.history = []

    # Display past messages
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    # Input row with aligned mic + arrow
    input_col, mic_col, send_col = st.columns([8, 1, 1])
    with input_col:
        user_input = st.text_input("Your Question", placeholder="Ask me about Kreo Tech...", label_visibility="collapsed", key="text_input")

    with mic_col:
        mic_pressed = st.button("🎙️", help="Speak", use_container_width=True)

    with send_col:
        send_pressed = st.button("➡️", help="Send", use_container_width=True)

    # --- Handle Mic Input ---
    if mic_pressed:
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with st.spinner("🎙️ Recording..."):
            record_audio(audio_file)
        user_input = transcribe_audio(audio_file)
        user_input = clean_input(user_input)
        st.caption(f"🎧 You said: _{user_input}_")

    # --- Handle Send ---
    if send_pressed and user_input:
        user_input = clean_input(user_input)
        st.session_state.history.append(("user", user_input))

        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}"
        response = model.generate_content(prompt).text
        st.session_state.history.append(("assistant", response))

        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(speak(response))
            else:
                loop.run_until_complete(speak(response))
        except RuntimeError:
            asyncio.run(speak(response))
