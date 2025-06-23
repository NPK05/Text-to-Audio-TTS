
# 🎙️ Text-to-Audio: FastSpeech2 + HiFi-GAN Based TTS System

Welcome to the **Text-to-Audio-TTS** repository — a custom-built, modular, and research-inspired project that transforms raw English text into natural-sounding speech using a combination of **FastSpeech2** for spectrogram generation and **HiFi-GAN** as a vocoder for waveform synthesis.

> 🔧 Entire pipeline handcrafted in Python, with no third-party UI tools or automated dashboards. This repository reflects a complete hands-on implementation.

---

## 📁 Repository Overview

```bash
Text-to-Audio-TTS/
│
├── inference.py                  # Real-time TTS inference pipeline
├── train.py                      # HiFi-GAN training script
├── dataset.py                    # Dataset loader and preprocessor
├── Text_to_Audio_Notebook.ipynb # Jupyter notebook for end-to-end experimentation
│
├── model/                        # FastSpeech2 & HiFi-GAN model architectures
├── checkpoints/                 # Pretrained model weights (local only)
├── data_hifigan_validated/      # Sample wav/mel data used for inference
├── outputs/                      # Generated spectrograms and .wav outputs
│
├── training_loss.png             # Training visualization
├── Figure_1.png                  # Sample spectrogram output
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation file
```

---

## 🔍 Objective

To build a robust, real-time Text-to-Speech (TTS) system that combines:

- 🧠 **FastSpeech2** — for fast, robust, and accurate mel-spectrogram generation.
- 🔊 **HiFi-GAN** — for high-fidelity audio waveform synthesis from spectrograms.

---

## 🧪 Technologies Used

- `Python 3.x`
- `PyTorch` for deep learning model training
- `NumPy`, `Librosa` for audio processing
- `Matplotlib` and `Seaborn` for training and output visualizations

---

## 🧾 Dataset Used

This project utilizes a subset of the open-source [LJSpeech-1.1 dataset](https://keithito.com/LJ-Speech-Dataset/) (⭢ **Kaggle mirror available** [here](https://www.kaggle.com/datasets/mervemenekse/ecommerce-dataset)). Due to GitHub's file size restrictions, only a small set of **mel-spectrograms** and **wav files** are uploaded in the `data_hifigan_validated/` folder.

To train or evaluate the full system:

1. Download the full dataset (LJSpeech-1.1).
2. Preprocess into aligned `wavs/` and `mel/` pairs.
3. Place them inside the appropriate `/data` folder.

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/NPK05/Text-to-Audio-TTS.git
cd Text-to-Audio-TTS
pip install -r requirements.txt
```

---

## 🚀 Run Inference

```python
from inference import text_to_speech

text_to_speech("This is a sample text-to-speech output using FastSpeech2 and HiFi-GAN.")
```

---

## 📊 Output Samples

| 🧠 Feature                  | 📈 Output |
|---------------------------|-----------|
| Training Curve            | ![training_loss](training_loss.png) |
| Mel Spectrogram Sample    | ![mel](Figure_1.png) |
| Final Audio Waveform      | Stored in `/outputs` as `.wav` files |

---

## 🔐 Checkpoints & Models

Due to the 25MB GitHub file limit:
- 🔒 Full pretrained model weights are **excluded** from the repo.
- You can download trained checkpoints or generate your own using `train.py`.

Place any trained `.pt` or `.ckpt` models under:

```
Text-to-Audio-TTS/
└── checkpoints/
    └── generator.pt
    └── discriminator.pt
```

---

## 🧠 Core Features

- 🔤 **Text Preprocessing** — Converts raw text to phoneme-based input
- 📈 **Mel Spectrogram Generator** — FastSpeech2 decoder
- 🔊 **Waveform Generation** — HiFi-GAN generator architecture
- 🎯 **Modular Inference** — Easily test and modify
- 🛠️ **Custom Training Pipeline** — Manual control over epochs, loss, and batch size

---

## 📌 Design Philosophy

This project was built as a complete learning implementation with no drag-and-drop UI tools or auto-generated dashboards. Every component from model loading to spectrogram visualization was manually crafted in Python.

📚 **Inspired by**:
- _Neural Speech Synthesis with Transformer Network_ – Microsoft
- _HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis_
- Kaggle shared notebooks and open GitHub repositories

---

## 🧠 Suggested Enhancements

- Integrate Tacotron2 as an optional frontend.
- Use Griffin-Lim as an alternative vocoder for fast experiments.
- Improve phoneme-to-spectrogram alignment and duration modeling.

---

## 📘 References & Learning Sources

- 🔗 [HiFi-GAN GitHub](https://github.com/jik876/hifi-gan)
- 📘 *Deep Learning for Speech and Language* – Li Deng
- 📕 *Python Machine Learning* – Sebastian Raschka
- 🧪 Blog posts on [Medium](https://medium.com/), [Kaggle](https://www.kaggle.com/), and official [PyTorch tutorials](https://pytorch.org/tutorials/)

---

## 🙌 Acknowledgments

Special thanks to:
- The **Kaggle** community for shared insights and baseline models.
- **GPT** for initial code restructuring and formatting assistance.
- All research authors and contributors to TTS systems.

---

## 📄 License

This project is available under the **MIT License** for educational and non-commercial use.

---

### 🚀 Enjoy building next-gen audio AI from scratch!
