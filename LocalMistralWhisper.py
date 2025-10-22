import torch
import sounddevice as sd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from TTS.api import TTS
from transformers import pipeline
import time
from threading import Thread

# --- 1. Mistral 7B Setup (Text Generation) ---
# Ensure your model path is correct
model_path = "/home/tanish/LocalLLMS/mistral-7b-bnb-4bit-local"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

print("Mistral 7B loaded successfully!")

# --- 2. Whisper Setup (Speech-to-Text) ---
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="/home/tanish/LocalLLMS/whisper-turbo-large-local",
    torch_dtype=torch.float16,
    device="cuda:0",
)
print("Whisper ASR pipeline loaded successfully!")

# --- 3. Coqui TTS Setup (Text-to-Speech) ---
tts_model_name = "/home/tanish/LocalLLMS/Coqui-local/model.pth"
tts = TTS(tts_model_name).to("cuda")
print("Coqui TTS loaded successfully!")

# --- 4. Main Conversational Loop ---
def speak(text):
    """Generates audio from text and plays it."""
    print(f"AI is speaking: {text}")
    wav = tts.tts(text=text, speaker=tts.speakers[0])
    sd.play(np.array(wav), samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()

def listen_for_input():
    """Listens for a user's voice input and transcribes it."""
    print("Listening... (Press Ctrl+C to exit)")
    fs = 16000  # Sample rate for Whisper
    duration = 5  # Listen for 5 seconds

    while True:
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait() # Wait for the recording to finish
        
        # Check if the audio is mostly silence
        if np.max(np.abs(audio_data)) < 0.1:  # Adjust threshold as needed
            continue
        
        audio_input = {"raw": audio_data, "sampling_rate": fs}
        transcription = whisper_pipe(audio_input)["text"]
        print(f"You said: {transcription}")
        return transcription

def main_loop():
    while True:
        try:
            # Step 1: Listen and Transcribe
            user_input = listen_for_input()
            if not user_input.strip():
                continue

            # Step 2: Generate Response with Mistral
            prompt = f"User: {user_input}\nAI:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
            output = model.generate(**inputs, max_new_tokens=100)
            response = tokenizer.decode(output[0], skip_special_tokens=True).split("AI:")[-1].strip()
            
            # Step 3: Speak the Response with Coqui
            speak(response)

        except KeyboardInterrupt:
            print("\nExiting the conversational loop.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main_loop()
