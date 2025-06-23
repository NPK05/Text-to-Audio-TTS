
import os
import torch
import numpy as np
from torch.utils.data import Dataset

# === Phoneme Symbol Table ===
# This is a basic English phoneme set with stress markers
PHONEMES = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2',
            'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2',
            'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2',
            'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
            'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1',
            'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ',', '.', '!', '?', '-', ' ']

phoneme_to_id = {p: i for i, p in enumerate(PHONEMES)}

def text_to_sequence(phoneme_str):
    sequence = []
    for p in phoneme_str.split():
        if p in phoneme_to_id:
            sequence.append(phoneme_to_id[p])
    return sequence


class LJSpeechDataset(Dataset):
    def __init__(self, phoneme_file, mel_dir):
        super().__init__()
        self.samples = []
        self.mel_dir = mel_dir

        with open(phoneme_file, 'r', encoding='utf-8') as f:
            for line in f:
                audio_id, phoneme_str = line.strip().split('|')
                phoneme_seq = text_to_sequence(phoneme_str)
                mel_path = os.path.join(mel_dir, f"{audio_id}.npy")

                if os.path.exists(mel_path):
                    self.samples.append((phoneme_seq, mel_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        phoneme_seq, mel_path = self.samples[idx]
        mel = np.load(mel_path)

        return {
            'phonemes': torch.LongTensor(phoneme_seq),
            'mel': torch.FloatTensor(mel.T)  # (T, 80)
        }


# === Collate Function for Padding ===
def collate_fn(batch):
    phoneme_seqs = [item['phonemes'] for item in batch]
    mel_specs = [item['mel'] for item in batch]

    phoneme_lens = [len(seq) for seq in phoneme_seqs]
    mel_lens = [mel.shape[0] for mel in mel_specs]

    phonemes_padded = torch.nn.utils.rnn.pad_sequence(phoneme_seqs, batch_first=True, padding_value=0)
    mel_padded = torch.nn.utils.rnn.pad_sequence(mel_specs, batch_first=True, padding_value=0.0)

    return {
        'phonemes': phonemes_padded,
        'mel': mel_padded,
        'phoneme_lens': torch.LongTensor(phoneme_lens),
        'mel_lens': torch.LongTensor(mel_lens)
    }
