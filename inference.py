import torch
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import cmudict
from model.fastspeech2 import FastSpeech2
from data.dataset import phoneme_to_id

# === CONFIG ===
VOCAB_SIZE = 75
CHECKPOINT_PATH = "checkpoints/fastspeech2_epoch50.pt"

# === LOAD MODEL ===
model = FastSpeech2(vocab_size=VOCAB_SIZE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()
print("✅ Model loaded.")

# === LOAD CMU DICTIONARY ===
try:
    cmu = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    cmu = cmudict.dict()

def word_to_phonemes(word):
    word = word.lower()
    return cmu[word][0] if word in cmu else []

def text_to_sequence(text):
    words = text.strip().split()
    phonemes = []
    for word in words:
        phonemes.extend(word_to_phonemes(word))
    phonemes = [p for p in phonemes if p in phoneme_to_id]
    sequence = [phoneme_to_id[p] for p in phonemes]
    
    # Debug logs
    print("Input text:", text)
    print("Phonemes:", phonemes)
    print("Phoneme IDs:", sequence)
    
    return torch.LongTensor(sequence).unsqueeze(0)

# === INFERENCE ===
input_text = "The quick brown fox jumps over the lazy dog"
phoneme_ids = text_to_sequence(input_text)

with torch.no_grad():
    mel_output, predicted_durations = model(phoneme_ids)

mel_output = mel_output.squeeze(0).cpu().numpy().T  # (80, time)

# === VISUALIZE MEL SPECTROGRAM ===
plt.figure(figsize=(10, 4))
plt.imshow(mel_output, aspect='auto', origin='lower', interpolation='none')
plt.title("Generated Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.colorbar()
plt.tight_layout()
plt.savefig("generated_mel.png")
plt.show()

print("✅ Inference complete. Mel spectrogram saved and displayed.")
