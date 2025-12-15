import torch
import torchaudio
import torchaudio.functional as F

import matplotlib.pyplot as plt

from pathlib import Path
import urllib.request
import sounddevice as sd

# Carpeta local donde se guardarán los assets
ASSETS_DIR = Path.home() / ".cache" / "torchaudio" / "tutorial-assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://download.pytorch.org/torchaudio/tutorial-assets"


def download_asset(filename: str) -> str:
    url = f"{BASE_URL}/{filename}"
    filepath = ASSETS_DIR / filename
    if not filepath.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    return str(filepath)


SAMPLE_WAV = download_asset("steam-train-whistle-daniel_simon.wav")
SAMPLE_RIR = download_asset("Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
SAMPLE_SPEECH = download_asset("Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")
SAMPLE_NOISE = download_asset("Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")


waveform, sr_waveform = torchaudio.load(SAMPLE_WAV)  # shape: (channels, samples)
waveform = waveform / waveform.abs().max()  # normalizar
print(f"Reproduciendo WAV original, shape: {waveform.shape}")
sd.play(waveform.T.numpy(), sr_waveform)  # .T → (samples, channels)
sd.wait()

# Cargar audios en (channels, samples)
speech, sr_speech = torchaudio.load(SAMPLE_SPEECH)  # shape: (channels, samples)
noise, sr_noise = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, : speech.shape[1]]  # recortar al mismo largo
noise = noise / noise.abs().max()  # normalizar ruido

# Definir SNRs
snr_dbs = [20, 10, 3]
noisy_speeches = []

# Agregar ruido con SNR controlado
for snr in snr_dbs:
    snr_tensor = torch.tensor([snr], dtype=torch.float32)
    noisy = F.add_noise(speech, noise, snr_tensor)
    # Normalizar a [-1,1] para reproducir
    noisy = noisy / noisy.abs().max()
    noisy_speeches.append(noisy)


# Función para preparar audio para sounddevice
def prepare_audio_for_sd(tensor):
    tensor = tensor.float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    elif tensor.shape[0] <= 2 and tensor.ndim == 2:
        tensor = tensor.T
    if tensor.shape[1] > 2:
        tensor = tensor[:, :2]
    return tensor


# Reproducir audios ruidosos
for i, noisy in enumerate(noisy_speeches):
    audio = prepare_audio_for_sd(noisy)
    print(f"SNR: {snr_dbs[i]} dB, shape: {audio.shape}")
    sd.play(audio.numpy(), sr_speech)
    sd.wait()
