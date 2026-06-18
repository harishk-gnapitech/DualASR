from transformers import pipeline
import librosa

audio_data, sample_rate = librosa.load("Recording.wav", sr=16000, mono=True)
            

# Load MedASR model
medasr = pipeline("automatic-speech-recognition", model="google/medasr")

# Pass the clean numeric audio data directly to the pipeline
result = medasr(audio_data)
print(result["text"])
