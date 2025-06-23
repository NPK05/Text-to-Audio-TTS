import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.generator import HiFiGANGenerator
from models.discriminator import MultiPeriodDiscriminator
from dataset import MelAudioDataset
from loss import generator_loss, discriminator_loss, feature_loss
from tqdm import tqdm
import traceback

# === CONFIG ===
with open("config.json") as f:
    config = json.load(f)

mel_dir = config["mel_dir"]
wav_dir = config["wav_dir"]
BATCH_SIZE = config.get("batch_size", 4)
EPOCHS = config.get("epochs", 100)
LEARNING_RATE = config.get("learning_rate", 1e-4)
CHECKPOINT_DIR = config.get("checkpoint_dir", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === COLLATE FUNCTION ===
def collate_fn(batch):
    mels, audios = zip(*batch)
    try:
        # Check shapes
        max_mel_len = max([mel.shape[1] for mel in mels])
        max_audio_len = max([audio.shape[0] for audio in audios])

        mels_padded = torch.stack([
            F.pad(mel, (0, max_mel_len - mel.shape[1])) for mel in mels
        ])
        audios_padded = torch.stack([
            F.pad(audio, (0, max_audio_len - audio.shape[0])) for audio in audios
        ])

        return mels_padded, audios_padded
    except Exception as e:
        print("❌ Error in collate_fn:", e)
        traceback.print_exc()
        return torch.tensor([]), torch.tensor([])

# === TRAINING LOOP ===
def main():
    print("🚀 Initializing training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MelAudioDataset(mel_dir, wav_dir)
    print(f"📊 Total samples in dataset: {len(dataset)}")
    if len(dataset) == 0:
        print("❌ ERROR: Dataset is empty! Check mel_dir and wav_dir paths.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    generator = HiFiGANGenerator().to(device)
    discriminator = MultiPeriodDiscriminator().to(device)

    opt_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    opt_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        generator.train()
        discriminator.train()

        print(f"\n🔁 Starting Epoch {epoch}/{EPOCHS}")
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", dynamic_ncols=True)
        batch_count = 0

        for batch_idx, (mel, audio) in enumerate(progress):
            if mel.nelement() == 0 or audio.nelement() == 0:
                continue

            try:
                mel = mel.to(device)
                audio = audio.to(device)

                # === Train Discriminator ===
                fake_audio = generator(mel).detach()
                d_loss = discriminator_loss(discriminator, audio, fake_audio)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                # === Train Generator ===
                fake_audio = generator(mel)
                g_loss = generator_loss(discriminator, audio, fake_audio)
                feat_loss = feature_loss(discriminator, audio, fake_audio)
                total_g_loss = g_loss + feat_loss

                opt_g.zero_grad()
                total_g_loss.backward()
                opt_g.step()

                batch_count += 1
                progress.set_postfix({
                    "loss_D": round(d_loss.item(), 4),
                    "loss_G": round(total_g_loss.item(), 4),
                    "batch": batch_count
                })

            except Exception as batch_err:
                print(f"❌ Exception in batch {batch_idx + 1}: {str(batch_err)}")
                traceback.print_exc()

        print(f"✅ Epoch {epoch} completed. Total batches processed: {batch_count}")
        torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch}.pt"))
        print(f"✅ Models saved for Epoch {epoch}.")

if __name__ == "__main__":
    main()
