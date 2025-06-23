
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf

class MelAudioDataset(Dataset):
    def __init__(self, mel_dir, wav_dir):
        self.mel_files = sorted([f for f in os.listdir(mel_dir) if f.endswith('.npy')])
        self.wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
        self.mel_dir = mel_dir
        self.wav_dir = wav_dir

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel = np.load(os.path.join(self.mel_dir, self.mel_files[idx]))
        audio, _ = sf.read(os.path.join(self.wav_dir, self.wav_files[idx]))
        mel = torch.tensor(mel).float()
        audio = torch.tensor(audio).float()
        return mel, audio
