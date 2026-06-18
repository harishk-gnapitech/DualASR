import os
import whisperx
from dotenv import load_dotenv

load_dotenv()

# Configuration
device = "cpu"  # Use "cpu" if you do not have an NVIDIA GPU
compute_type = "int8"  # Use "int8" on CPU
audio_file = "conversation.mp3"

# Retrieve your existing token from your machine environment setup
hf_token = os.getenv("MedASR_Read")

print("1. Transcribing Audio...")
model = whisperx.load_model("base.en", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=16, language="en")

print("2. Aligning Word Timestamps...")
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device=device
)
aligned_result = whisperx.align(
    result["segments"], 
    model_a, 
    metadata, 
    audio, 
    device, 
    return_char_alignments=False
)

print("3. Running Speaker Diarization...")
# This automatically calls pyannote using your Hugging Face permission token
diarize_model = whisperx.diarize.DiarizationPipeline(
    token=hf_token, 
    device=device
)
diarize_segments = diarize_model(audio)

print("4. Assigning Speakers to Transcript Text...")
# Merges the word timestamps with the speaker maps
final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

# Print organized, clean dialogue results
print("\n--- Final Speaker-Attributed Transcript ---")
for segment in final_result["segments"]:
    # Safely get speaker name, default to UNKNOWN if overlapping/unclear
    speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {speaker}: {segment['text']}")

