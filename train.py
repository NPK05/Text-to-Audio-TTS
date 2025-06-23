import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.fastspeech2 import FastSpeech2
from data.dataset import LJSpeechDataset, collate_fn
from tqdm import tqdm
import matplotlib.pyplot as plt

# === CONFIG ===
PHONEME_FILE = r"C:\\Users\\welcome\\Downloads\\Text to Audio\\phoneme_data.txt"
MEL_DIR = r"C:\\Users\\welcome\\Downloads\\Text to Audio\\LJSpeech-1.1\\LJSpeech-1.1\\mel_spectrograms"
VOCAB_SIZE = 75  # Based on phoneme list size
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = "checkpoints"
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# === SETUP ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
dataset = LJSpeechDataset(PHONEME_FILE, MEL_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = FastSpeech2(vocab_size=VOCAB_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)  # Mixed precision only if CUDA

loss_history = []

# === TRAIN LOOP ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress:
        phonemes = batch['phonemes'].to(DEVICE)
        mel_targets = batch['mel'].to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_CUDA):
            mel_outputs, durations = model(phonemes)

        # Match length for temporary fix
        min_len = min(mel_outputs.size(1), mel_targets.size(1))
        mel_outputs = mel_outputs[:, :min_len, :]
        mel_targets = mel_targets[:, :min_len, :]

        loss = criterion(mel_outputs, mel_targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    scheduler.step()

    # === Save Checkpoint ===
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fastspeech2_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), ckpt_path)

    print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}. Model saved to {ckpt_path}")

# === Plot Loss ===
plt.figure(figsize=(8, 4))
plt.plot(loss_history, label="MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()
