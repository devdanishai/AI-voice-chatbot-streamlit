import os
import speech_recognition as sr
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import tempfile
import json
from datetime import datetime
import uuid
import asyncio
import edge_tts
import pygame
import traceback


# Configuration
class Config:
    LLM_MODEL = "llama3-8b-8192"
    MAX_TOKENS = 20
    TEMPERATURE = 0.3
    MAX_HISTORY_LENGTH = 50
    LISTEN_TIMEOUT = 5
    TEMP_DIR = os.path.join(os.getcwd(), "temp_audio")
    DEFAULT_VOICE = "en-US-JennyNeural"


# Initialize session state for voice selection
if 'selected_voice' not in st.session_state:
    st.session_state.selected_voice = Config.DEFAULT_VOICE
if 'available_voices' not in st.session_state:
    st.session_state.available_voices = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_tested' not in st.session_state:
    st.session_state.system_tested = False

# Create temp directory if it doesn't exist
if not os.path.exists(Config.TEMP_DIR):
    os.makedirs(Config.TEMP_DIR)

# Initialize pygame mixer only once at startup
if not st.session_state.system_tested:
    try:
        pygame.mixer.init()
        st.write("Pygame mixer initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize pygame mixer: {str(e)}")


async def get_available_voices():
    """Get list of available voices"""
    try:
        voices = await edge_tts.list_voices()
        return voices
    except Exception as e:
        st.error(f"Error getting voices: {str(e)}")
        return []


async def generate_speech(text, output_file):
    """Generate speech using edge-tts with selected voice"""
    try:
        communicate = edge_tts.Communicate(text, st.session_state.selected_voice)
        await communicate.save(output_file)
        return True
    except edge_tts.exceptions.NoAudioReceived:
        st.error(f"Voice {st.session_state.selected_voice} failed to generate speech")
        return False
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return False


def speak(text):
    """Text-to-speech conversion using edge-tts"""
    try:
        temp_file = os.path.join(Config.TEMP_DIR, f"tts_{uuid.uuid4()}.mp3")

        # Run edge-tts
        asyncio.run(generate_speech(text, temp_file))

        # Verify file exists
        if not os.path.exists(temp_file):
            st.error("Speech file was not generated")
            return

        # Play audio
        try:
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except pygame.error as pe:
            st.error(f"Pygame error: {str(pe)}")

        # Cleanup
        pygame.mixer.music.unload()
        os.remove(temp_file)

    except Exception as e:
        st.error(f"Speech error: {str(e)}")
        st.error(traceback.format_exc())


async def test_audio_system():
    """Test the audio system"""
    try:
        test_text = "Test audio system."
        temp_file = os.path.join(Config.TEMP_DIR, "test_audio.mp3")
        success = await generate_speech(test_text, temp_file)
        if success and os.path.exists(temp_file):
            os.remove(temp_file)
            return True
        return False
    except Exception as e:
        st.error(f"Audio system test failed: {str(e)}")
        return False


async def initialize_voices():
    """Initialize available voices"""
    voices = await get_available_voices()
    st.session_state.available_voices = [
        {
            'name': voice['ShortName'],
            'display': f"{voice['Locale']} - {voice['ShortName']}"
        }
        for voice in voices
    ]


def limit_history():
    """Limit chat history to prevent memory issues"""
    if len(st.session_state.chat_history) > Config.MAX_HISTORY_LENGTH:
        st.session_state.chat_history = st.session_state.chat_history[-Config.MAX_HISTORY_LENGTH:]


def download_chat_history():
    """Export chat history as JSON"""
    history_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.chat_history
    }
    return json.dumps(history_data, indent=2)


def cleanup_old_files():
    """Clean up old temporary files"""
    for file in os.listdir(Config.TEMP_DIR):
        file_path = os.path.join(Config.TEMP_DIR, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error deleting {file_path}: {e}")


def transcribe_and_generate_response():
    """Main function to handle voice input and generate responses"""
    try:
        # Record audio
        with st.spinner("Listening..."):
            with sr.Microphone() as source:
                st.write("Please speak something...")
                try:
                    audio = recognizer.listen(source, timeout=Config.LISTEN_TIMEOUT)
                except sr.WaitTimeoutError:
                    st.warning("No speech detected. Please try again.")
                    return

        audio_data = audio.get_wav_data()
        if not audio_data:
            st.warning("No audio was recorded. Please try again.")
            return

        # Process audio
        with st.spinner("Processing..."):
            temp_filename = os.path.join(Config.TEMP_DIR, f"audio_{uuid.uuid4()}.wav")

            try:
                with open(temp_filename, 'wb') as temp_file:
                    temp_file.write(audio_data)

                transcribed_text = recognizer.recognize_google(audio)
                user_input_text = transcribed_text
                st.session_state.chat_history.append({"role": "user", "content": user_input_text})

            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        # Generate response
        messages = [
            {"role": "system", "content": "You are a concise assistant. Always respond in 40 words or less."}
        ] + st.session_state.chat_history

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=Config.LLM_MODEL,
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE
        )

        llm_response = chat_completion.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": llm_response})

        # Limit history and speak response
        limit_history()
        speak(llm_response)

        st.rerun()

    except sr.RequestError as e:
        st.error(f"Could not request results: {str(e)}")
    except sr.UnknownValueError:
        st.error("Could not understand audio. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(traceback.format_exc())
    finally:
        cleanup_old_files()


# Load environment variables
load_dotenv()

# Validate environment variables
if not os.getenv('GROQ_API_KEY'):
    st.error("GROQ_API_KEY not found in environment variables")
    st.stop()

# Initialize the Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize voices at startup
asyncio.run(initialize_voices())

# Run audio system test only once at startup
if not st.session_state.system_tested:
    if asyncio.run(test_audio_system()):
        st.success("Audio system is working properly")
    else:
        st.error("Audio system is not working properly")
    st.session_state.system_tested = True

# Streamlit UI
st.title("Voice-Activated Chatbot with History")

# Sidebar controls
with st.sidebar:
    st.subheader("Settings")

    if st.session_state.available_voices:
        # Voice selection dropdown
        voice_options = [voice['display'] for voice in st.session_state.available_voices]
        voice_names = [voice['name'] for voice in st.session_state.available_voices]

        try:
            default_idx = voice_names.index(Config.DEFAULT_VOICE)
        except ValueError:
            default_idx = 0

        selected_display = st.selectbox(
            "Select Voice",
            options=voice_options,
            index=default_idx
        )

        selected_idx = voice_options.index(selected_display)
        st.session_state.selected_voice = voice_names[selected_idx]
    else:
        st.warning("No voices available")

    if st.button("Download Chat History"):
        st.download_button(
            label="Download JSON",
            data=download_chat_history(),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Main chat interface
st.subheader("Chat History:")
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

# Control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Talking"):
        transcribe_and_generate_response()
with col2:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        cleanup_old_files()
        st.rerun()