
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class FeedForwardTransformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, ff_mult=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * ff_mult,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, in_dim, filter_size=256, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, filter_size, kernel_size, padding=1)
        self.ln1 = nn.LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=1)
        self.ln2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv1(x)
        x = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.ln2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear(x.transpose(1, 2)).squeeze(-1)
        return x  # (B, T)


class LengthRegulator(nn.Module):
    def forward(self, x, durations):
        output = []
        for batch, dur in zip(x, durations):
            expanded = [batch[i].repeat(int(dur[i]), 1) for i in range(len(dur))]
            output.append(torch.cat(expanded, dim=0))
        max_len = max([o.size(0) for o in output])
        output_padded = torch.stack([F.pad(o, (0, 0, 0, max_len - o.size(0))) for o in output])
        return output_padded


class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=2, num_layers=4, mel_bins=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = FeedForwardTransformer(d_model, n_heads, num_layers)
        self.duration_predictor = DurationPredictor(d_model)
        self.length_regulator = LengthRegulator()
        self.decoder = FeedForwardTransformer(d_model, n_heads, num_layers)
        self.mel_proj = nn.Linear(d_model, mel_bins)

    def forward(self, phoneme_ids, durations=None):
        x = self.embedding(phoneme_ids)
        x = self.pos_encoding(x)
        x = self.encoder(x)

        if durations is None:
            log_durations = self.duration_predictor(x)
            durations = torch.clamp(torch.round(torch.exp(log_durations) - 1), min=1).long()
        else:
            durations = durations.long()

        x = self.length_regulator(x, durations)
        x = self.pos_encoding(x)
        x = self.decoder(x)
        mel_output = self.mel_proj(x)

        return mel_output, durations
