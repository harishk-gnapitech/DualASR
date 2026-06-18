from faster_whisper import WhisperModel

# 1. Choose your model size and device (use "cpu" if you do not have a GPU)
model_size = "base"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# 2. Transcribe the audio file
print("Transcribing audio...")
segments, info = model.transcribe("Recording.wav", beam_size=5)

# 3. Print the detected language
print(f"Detected language: '{info.language}' with probability {info.language_probability:.2f}")

# 4. Print the text chunks with their timestamps
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
